//! Pm4CmdBuilder + Pm4Queue — PM4 command buffer construction and dispatch.

use std::sync::Arc;
use std::os::unix::io::RawFd;
use super::device::KfdDevice;
use super::buffer::GpuBuffer;
use super::ioctl::*;
pub struct Pm4CmdBuilder {
    cmds: Vec<u32>,
}

impl Pm4CmdBuilder {
    pub fn new() -> Self {
        Self { cmds: Vec::with_capacity(64) }
    }

    /// Emit PACKET3: [31:30]=type3, [29:16]=count-1, [15:8]=opcode
    /// AMD PM4 spec: count field = number of body dwords - 1
    fn pkt3(&mut self, opcode: u32, body: &[u32]) {
        let header = (3u32 << 30) | (((body.len() as u32 - 1) & 0x3FFF) << 16) | (opcode << 8);
        self.cmds.push(header);
        self.cmds.extend_from_slice(body);
    }

    /// SET_SH_REG: write consecutive shader registers
    pub fn set_sh_reg(&mut self, reg_addr: u32, values: &[u32]) {
        let reg_offset = (reg_addr - SH_REG_BASE) >> 2;
        let mut body = Vec::with_capacity(1 + values.len());
        body.push(reg_offset);
        body.extend_from_slice(values);
        self.pkt3(PM4_SET_SH_REG, &body);
    }

    /// ACQUIRE_MEM for GFX10+ (with GCR_CNTL for cache invalidation)
    pub fn acquire_mem_gfx10(&mut self) {
        // GFX10+ ACQUIRE_MEM format: 7 body dwords
        // [0] = 0
        // [1:2] = coherence size (u64, all memory)
        // [3:4] = coherence base (u64, 0)
        // [5] = poll interval (0)
        // [6] = GCR_CNTL flags (invalidate all caches)
        let gcr_cntl: u32 =
            (1 << 0)  |  // GLI_INV
            (1 << 1)  |  // GLM_INV
            (1 << 2)  |  // GLM_WB
            (1 << 3)  |  // GLK_INV
            (1 << 4)  |  // GLK_WB
            (1 << 5)  |  // GLV_INV
            (1 << 6)  |  // GL1_INV
            (1 << 7)  |  // GL2_INV
            (1 << 8);    // GL2_WB
        self.pkt3(PM4_ACQUIRE_MEM, &[
            0,                  // cp_coher_cntl
            0xFFFF_FFFF, 0xFF,  // coherence size (large)
            0, 0,               // coherence base
            0,                  // poll interval
            gcr_cntl,           // GCR cache control flags
        ]);
    }

    /// DISPATCH_DIRECT: launch workgroups
    pub fn dispatch_direct(&mut self, grid: [u32; 3], dispatch_initiator: u32) {
        self.pkt3(PM4_DISPATCH_DIRECT, &[grid[0], grid[1], grid[2], dispatch_initiator]);
    }

    /// EVENT_WRITE: various GPU events
    pub fn event_write(&mut self, event_type: u32, event_index: u32) {
        let event_dw = event_type | (event_index << 8);
        self.pkt3(PM4_EVENT_WRITE, &[event_dw]);
    }

    /// RELEASE_MEM: write value to memory upon pipeline completion
    pub fn release_mem(&mut self, addr: u64, value: u32, data_sel: u32, int_sel: u32, cache_flush: bool) {
        let cache_flags = if cache_flush {
            (1 << 12) |  // GLV_INV
            (1 << 13) |  // GL1_INV
            (1 << 14) |  // GL2_INV
            (1 << 15) |  // GLM_WB
            (1 << 16) |  // GLM_INV
            (1 << 17) |  // GL2_WB
            (1 << 18)    // SEQ
        } else {
            0
        };
        let event_dw = CACHE_FLUSH_AND_INV_TS_EVENT | (5 << 8) | cache_flags; // event_index=5 for MEC end_of_pipe
        let data_dw = (data_sel << 29) | (int_sel << 24);
        self.pkt3(PM4_RELEASE_MEM, &[
            event_dw,
            data_dw,
            addr as u32,
            (addr >> 32) as u32,
            value,
            0,   // ctxid
            0,   // padding
        ]);
    }

    /// WRITE_DATA: write 32-bit value to GPU-visible memory address
    /// Opcode 0x37. dst_sel=2 (memory mapped via TC L2), wr_confirm=1, engine=ME
    pub fn write_data(&mut self, addr: u64, value: u32) {
        const PM4_WRITE_DATA: u32 = 0x37;
        // dst_sel=2 = "memory mapped" (used by tinygrad COPY_DATA with (2<<8)|4)
        // wr_confirm=bit20: wait for write to be acknowledged before proceeding
        // engine=MEC(1<<30): use MEC engine for compute
        let control_dw: u32 = (2 << 8) | (1 << 20);
        self.pkt3(PM4_WRITE_DATA, &[
            control_dw,
            addr as u32,
            (addr >> 32) as u32,
            value,
        ]);
    }

    /// Finish and return the PM4 dword sequence
    pub fn finish(self) -> Vec<u32> {
        self.cmds
    }

    /// Strict compute barrier: wait for all prior dispatches to complete + flush L1/L2 caches.
    /// PM4 equivalent of AQL barrier=1. Use between dependent kernel dispatches in an IB.
    pub fn compute_barrier(&mut self) {
        // 1. Wait for all compute shaders to finish
        self.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
        // 2. Flush/invalidate all GPU caches (same flags as acquire_mem_gfx10)
        let gcr_cntl: u32 =
            (1 << 0)  |  // GLI_INV
            (1 << 1)  |  // GLM_INV
            (1 << 2)  |  // GLM_WB
            (1 << 3)  |  // GLK_INV
            (1 << 4)  |  // GLK_WB
            (1 << 5)  |  // GLV_INV
            (1 << 6)  |  // GL1_INV
            (1 << 7)  |  // GL2_INV
            (1 << 8);    // GL2_WB
        self.pkt3(PM4_ACQUIRE_MEM, &[
            0,                  // cp_coher_cntl
            0xFFFF_FFFF, 0xFF,  // coherence size
            0, 0,               // coherence base
            0,                  // poll interval
            gcr_cntl,           // GCR flags
        ]);
    }
}

pub struct Pm4Queue {
    pub queue_id: u32,
    pub ring_buffer: GpuBuffer,
    pub ring_size: u32,
    pub write_ptr_host: *mut u64,
    pub read_ptr_host: *mut u64,
    pub doorbell_ptr: *mut u64, // KFD doorbell is u64 (same as AQL)
    pub write_offset: u32,      // current byte offset into ring
    /// Original mmap base for doorbell (needed for correct munmap)
    pub(crate) doorbell_mmap_base: *mut u8,
    pub _wr_ptrs: GpuBuffer,
    pub _eop_buffer: GpuBuffer,
    pub _cwsr_buffer: Option<GpuBuffer>,
    pub device: Arc<KfdDevice>,
}

unsafe impl Send for Pm4Queue {}
unsafe impl Sync for Pm4Queue {}

// PM4 PACKET3 opcodes
const PACKET3_SET_SH_REG: u32        = 0x76;
const PACKET3_DISPATCH_DIRECT: u32   = 0x15;
const PACKET3_RELEASE_MEM: u32       = 0x49;
const PACKET3_ACQUIRE_MEM: u32       = 0x58;

// GFX11 Compute SH register offsets (relative to 0x2C00)
const COMPUTE_PGM_LO: u32          = 0x2C0C;
const COMPUTE_PGM_HI: u32          = 0x2C10;
const COMPUTE_PGM_RSRC1: u32       = 0x2C44;
const COMPUTE_PGM_RSRC2: u32       = 0x2C48;
const COMPUTE_USER_DATA_0: u32     = 0x2C4C;
const COMPUTE_NUM_THREAD_X: u32    = 0x2C78;
const COMPUTE_NUM_THREAD_Y: u32    = 0x2C7C;
const COMPUTE_NUM_THREAD_Z: u32    = 0x2C80;
const COMPUTE_RESOURCE_LIMITS: u32 = 0x2C14;

impl Pm4Queue {
    /// Write a u32 dword to the ring buffer at current offset
    fn emit(&mut self, dword: u32) {
        let off = (self.write_offset as usize) % self.ring_size as usize;
        unsafe {
            let ptr = self.ring_buffer.host_ptr.add(off) as *mut u32;
            std::ptr::write_volatile(ptr, dword);
        }
        self.write_offset += 4;
    }

    /// Emit PACKET3 header: [31:30]=type3, [29:16]=body_dword_count, [15:8]=opcode
    /// AMD spec: count field = number of body dwords (NOT -1)
    /// tinygrad pkt3(op, *args): op | len(args)<<16  — len(args) = body dwords
    fn emit_packet3(&mut self, opcode: u32, body_dwords: u32) {
        let header = (3u32 << 30) | ((body_dwords & 0x3FFF) << 16) | (opcode << 8);
        self.emit(header);
    }

    /// SET_SH_REG: write consecutive registers starting at reg_addr
    fn emit_set_sh_reg(&mut self, reg_addr: u32, values: &[u32]) {
        let reg_offset = (reg_addr - 0x2C00) >> 2; // convert to dword offset from SH base
        self.emit_packet3(PACKET3_SET_SH_REG, values.len() as u32 + 1);
        self.emit(reg_offset);
        for &v in values {
            self.emit(v);
        }
    }

    /// DISPATCH_DIRECT: launch compute workgroups
    fn emit_dispatch_direct(&mut self, dim_x: u32, dim_y: u32, dim_z: u32) {
        self.emit_packet3(PACKET3_DISPATCH_DIRECT, 4);
        self.emit(dim_x);
        self.emit(dim_y);
        self.emit(dim_z);
        // dispatch_initiator: bit 0 = compute_shader_en
        self.emit(1u32);
    }

    /// RELEASE_MEM: write a value to memory after all prior work completes
    /// Used for completion signaling
    fn emit_release_mem(&mut self, dst_addr: u64, value: u64) {
        self.emit_packet3(PACKET3_RELEASE_MEM, 6);
        // event_cntl: EOP event, cache policy
        // [5:0] = event_type = 0x2F (BOTTOM_OF_PIPE_TS for compute)
        // [11:8] = event_index = 5 (write confirmation)
        let event_cntl = 0x2F | (5 << 8);
        self.emit(event_cntl);
        // data_cntl: [28:26]=data_sel(1=32bit,2=64bit,3=timestamp), [24:22]=int_sel
        // data_sel=2 (send 64-bit data), int_sel=0 (no interrupt)
        let data_cntl = 2 << 26;
        self.emit(data_cntl);
        // dst address (low, high)
        self.emit(dst_addr as u32);
        self.emit((dst_addr >> 32) as u32);
        // data (low, high)
        self.emit(value as u32);
        self.emit((value >> 32) as u32);
    }

    /// ACQUIRE_MEM: invalidate GPU caches (L1/L2) to see fresh data
    fn emit_acquire_mem(&mut self) {
        self.emit_packet3(PACKET3_ACQUIRE_MEM, 6);
        self.emit(0); // cp_coher_cntl (all caches)
        self.emit(0xFFFFFFFF); // coher_size (everything)
        self.emit(0); // coher_size_hi
        self.emit(0); // coher_base
        self.emit(0); // coher_base_hi
        self.emit(0); // poll_interval
    }

    /// Dispatch a kernel via PM4 commands
    pub fn dispatch_nop(&mut self, code_addr: u64, rsrc1: u32, rsrc2: u32,
                        wg_size: [u32; 3], grid: [u32; 3],
                        signal_buf: Option<&GpuBuffer>) -> Result<(), String> {
        // Record start offset
        self.write_offset = unsafe { std::ptr::read_volatile(self.write_ptr_host) } as u32 * 4;

        // 1. Acquire memory (flush GPU caches)
        self.emit_acquire_mem();

        // 2. Set shader program address (code_addr >> 8 because COMPUTE_PGM_LO only stores top bits)
        self.emit_set_sh_reg(COMPUTE_PGM_LO, &[
            (code_addr >> 8) as u32,  // PGM_LO
            (code_addr >> 40) as u32, // PGM_HI
        ]);

        // 3. Set resource limits (allow 1 CU)
        self.emit_set_sh_reg(COMPUTE_RESOURCE_LIMITS, &[0]);

        // 4. Set rsrc1/rsrc2
        self.emit_set_sh_reg(COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]);

        // 5. Set workgroup dimensions
        self.emit_set_sh_reg(COMPUTE_NUM_THREAD_X, &[wg_size[0], wg_size[1], wg_size[2]]);

        // 6. Dispatch
        self.emit_dispatch_direct(grid[0], grid[1], grid[2]);

        // 7. Release mem (signal completion)
        if let Some(sig) = signal_buf {
            self.emit_release_mem(sig.gpu_addr(), 0x12345678DEADBEEF);
        }

        // 8. Dump ring buffer for debugging
        eprintln!("[PM4] Ring buffer ({} dwords, {} bytes):", self.write_offset / 4, self.write_offset);
        for i in 0..(self.write_offset / 4) {
            let dword = unsafe {
                let ptr = self.ring_buffer.host_ptr.add(i as usize * 4) as *const u32;
                std::ptr::read_volatile(ptr)
            };
            if i % 4 == 0 { eprint!("  [{:3}]:", i); }
            eprint!(" {:08X}", dword);
            if i % 4 == 3 || i == self.write_offset / 4 - 1 { eprintln!(); }
        }

        // 9. Ring doorbell
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let new_wptr = self.write_offset / 4; // convert byte offset to dword count
        eprintln!("[PM4] Ringing doorbell with wptr={}", new_wptr);
        unsafe {
            std::ptr::write_volatile(self.write_ptr_host, new_wptr as u64);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_wptr as u64);
        }

        // 9. Wait for completion
        if let Some(sig) = signal_buf {
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(5);
            loop {
                let val: u64 = sig.read_val(0);
                if val == 0x12345678DEADBEEF {
                    return Ok(());
                }
                if start.elapsed() > timeout {
                    let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                    return Err(format!("PM4 timeout: signal=0x{:X} rp={} wp={}", val, rp, new_wptr));
                }
                std::hint::spin_loop();
            }
        } else {
            // Poll read pointer
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(5);
            loop {
                let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                if rp >= new_wptr as u64 {
                    return Ok(());
                }
                if start.elapsed() > timeout {
                    return Err(format!("PM4 read_ptr timeout: rp={} wp={}", rp, new_wptr));
                }
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for Pm4Queue {
    fn drop(&mut self) {
        // Drain queue: wait for read_ptr to catch up with write_ptr (500ms timeout)
        let target_bytes = self.write_offset;
        if target_bytes > 0 {
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_millis(500);
            loop {
                let rp = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
                if rp >= target_bytes as u64 { break; }
                if start.elapsed() > timeout {
                    eprintln!("[KFD] WARNING: Pm4Queue {} drain timeout (read={} target={})",
                        self.queue_id, rp, target_bytes);
                    break;
                }
                std::hint::spin_loop();
            }
        }
        let mut args = KfdDestroyQueueArgs { queue_id: self.queue_id, pad: 0 };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_DESTROY_QUEUE,
            &mut args as *mut _ as *mut u8);
        // Unmap doorbell — use original mmap base, not the offset pointer
        unsafe { munmap(self.doorbell_mmap_base, 8192); }
    }
}

// =============================================================================
// GpuKernel — loaded kernel ready for dispatch
// =============================================================================

