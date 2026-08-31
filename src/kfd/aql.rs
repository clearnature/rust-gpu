//! AqlQueue — HSA AQL compute queue with doorbell dispatch.

use std::sync::Arc;
use std::os::unix::io::RawFd;
use super::device::KfdDevice;
use super::buffer::{GpuBuffer, validate_kernargs_bytes};
use super::ioctl::*;
use super::kernel::{GpuKernel, KernelLoadConfig};
use super::pm4::Pm4CmdBuilder;
// =============================================================================
// AqlQueue — AQL compute queue with doorbell dispatch
// =============================================================================

/// AQL hardware compute queue
pub struct AqlQueue {
    pub queue_id: u32,
    pub ring_buffer: GpuBuffer,
    pub ring_size: u32,
    pub write_ptr_host: *mut u64,
    pub read_ptr_host: *mut u64,
    pub doorbell_ptr: *mut u64,
    /// Original mmap base for doorbell (needed for correct munmap)
    pub(crate) doorbell_mmap_base: *mut u8,
    /// Size of doorbell mmap region
    pub(crate) doorbell_mmap_size: usize,
    // PM4-in-AQL: indirect buffer for PM4 commands
    pub(crate) pm4_ib: Option<GpuBuffer>,
    pub(crate) pm4_ib_offset: usize,
    // PM4-in-AQL: completion buffer — GPU writes seqno here after kernel finishes
    pub(crate) completion_buf: GpuBuffer,
    pub(crate) completion_seqno: u32,
    // Keep these alive (RAII)
    pub _wr_ptrs: GpuBuffer,
    pub _eop_buffer: GpuBuffer,
    pub _cwsr_buffer: Option<GpuBuffer>,
    pub device: Arc<KfdDevice>,
}

unsafe impl Send for AqlQueue {}
unsafe impl Sync for AqlQueue {}

impl AqlQueue {
    /// Dispatch a kernel. Returns after GPU completes execution.
    ///
    /// `kernel` = loaded GPU kernel
    /// `grid` = [grid_x, grid_y, grid_z] in threads (NOT workgroups)
    /// `kernargs` = kernel argument data (will be copied to GPU)
    pub fn dispatch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) -> Result<(), String> {
        self.dispatch_signal(kernel, grid, kernargs, None)
    }

    /// Dispatch with explicit signal buffer for completion tracking.
    /// Validates kernarg size and ensures ring space before dispatch.
    pub fn dispatch_signal(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        signal: Option<&GpuBuffer>,
    ) -> Result<(), String> {
        // Validate kernarg size matches kernel's declared requirement
        assert!(
            kernargs.size >= kernel.kernarg_size as usize,
            "kernarg too small: buffer={}B, kernel expects {}B",
            kernargs.size, kernel.kernarg_size
        );
        // Ensure ring buffer has space (prevents overflow)
        self.ensure_ring_space();
        // Get current write pointer
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1; // number of slots - 1
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // Build AQL dispatch packet (write header LAST for atomicity)
        let pkt_ptr = unsafe { self.ring_buffer.host_ptr.add(pkt_offset) as *mut AqlDispatchPacket };

        // Prepare completion signal using amd_signal_t layout:
        //   offset 0x00: kind (u64)   — must be 1 (AMD_SIGNAL_KIND_USER)
        //   offset 0x08: value (i64)  — CP will atomic_sub(1) upon completion
        //   offset 0x10: event_mailbox_ptr (u64) — must be 0 (no event)
        //   Total: 64 bytes, must be zeroed first
        let signal_va = if let Some(sig) = signal {
            // Zero entire 64-byte signal struct to clear any garbage event pointers
            unsafe { std::ptr::write_bytes(sig.host_ptr, 0, 64); }
            // kind = 1 (AMD_SIGNAL_KIND_USER) at offset 0
            sig.write_val::<u64>(0, 1);
            // value = 1 at offset 8 (CP will atomic_sub to make it 0)
            sig.write_val::<i64>(8, 1);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            // PCIe readback drain: ensure signal reaches GPU VRAM before doorbell
            let _ = unsafe { std::ptr::read_volatile(sig.host_ptr) };
            sig.gpu_addr()
        } else {
            0
        };

        // Build header: type=DISPATCH(2), barrier=1, acquire=SYSTEM(2), release=SYSTEM(2)
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |       // bits 0:7 = type
            (1 << 8) |                                        // bit 8 = barrier
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |         // bits 9:10 = acquire fence
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);         // bits 11:12 = release fence

        // DEBUG: dump AQL packet fields before writing
        if std::env::var("T0_DUMP_PKT").is_ok() {
            eprintln!("[AQL] submit: wg=[{},{},{}] grid=[{},{},{}] lds={} desc_va=0x{:X} ka=0x{:X} header=0x{:04X}",
                kernel.workgroup_size[0], kernel.workgroup_size[1], kernel.workgroup_size[2],
                grid[0], grid[1], grid[2], kernel.lds_size,
                kernel.descriptor_va, kernargs.gpu_addr(), header);
            eprintln!("[AQL] kernel.kernarg_size={} kernargs.size={}", kernel.kernarg_size, kernargs.size);
        }

        unsafe {
            // Write all fields EXCEPT header first (use addr_of_mut! for packed-safe access)
            let base = pkt_ptr as *mut u8;
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16); // setup: 3D (always, unused dims=1)
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16); // reserved0
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32); // private_segment_size
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size); // group_segment_size
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va); // kernel_object
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr()); // kernarg_address
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64); // reserved2
            std::ptr::write_volatile(base.add(0x38) as *mut u64, signal_va); // completion_signal

            // Memory fence before writing header
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

            // Write header LAST (makes packet visible to CP atomically)
            std::ptr::write_volatile(base.add(0x00) as *mut u16, header);

            // Memory fence after header write
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // Update write pointer (points PAST the last packet)
            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);

            // Memory fence to ensure write pointer is visible before doorbell
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // doorbell = new_write_idx - 1 (index of the just-written packet)
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }

        // Wait for completion
        if let Some(sig) = signal {
            self.wait_signal(sig)?;
        } else {
            // No signal: poll read_ptr (less safe but functional)
            self.wait_read_ptr(write_idx + 1)?;
        }

        Ok(())
    }

    /// Submit a kernel without waiting — pipelined dispatch.
    /// 
    /// Writes AQL packet and rings doorbell, returns immediately.
    /// Call `wait_idle()` after submitting a batch to drain the queue.
    /// No signal overhead, no waiting — maximum throughput.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        if std::env::var("T0_DUMP_PKT").is_ok() {
            eprintln!("[AQL] submit: wg=[{},{},{}] grid=[{},{},{}] lds={} desc_va=0x{:X} ka=0x{:X} kernarg_size={} ka_size={}",
                kernel.workgroup_size[0], kernel.workgroup_size[1], kernel.workgroup_size[2],
                grid[0], grid[1], grid[2], kernel.lds_size,
                kernel.descriptor_va, kernargs.gpu_addr(), kernel.kernarg_size, kernargs.size);
        }
        // Pre-dispatch kernarg validation (debug builds only — zero cost in release)
        // Catches NULL pointers and invalid GPU VAs BEFORE they reach the GPU.
        #[cfg(debug_assertions)]
        {
            let ka_size = (kernel.kernarg_size as usize).min(kernargs.size);
            if ka_size >= 8 {
                let ka_bytes = kernargs.read_bytes(0, ka_size);
                validate_kernargs_bytes(&ka_bytes, ka_size);
            }
        }

        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // barrier=1 ensures previous kernel completes before this one starts
        // (critical for data dependencies between consecutive kernels)
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |                                        // barrier bit
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        // DEBUG: dump AQL packet fields

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr());
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64); // no signal

            // SeqCst = mfence on x86: flush WC buffers before doorbell
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Submit kernel dispatch using raw kernargs GPU address (no signal, no wait).
    /// Similar to `submit()` but takes a `u64` address directly, enabling
    /// DispatchPool's single-buffer offset addressing.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit_at(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernarg_addr: u64,
    ) {
        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernarg_addr);
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64);

            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(base as *mut u16, header);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Batch-optimized submit: write AQL packet WITHOUT ringing doorbell.
    ///
    /// Call this N times, then call `ring_doorbell()` once.
    /// Avoids N MMIO writes to doorbell (each ~0.5-2μs via PCIe).
    /// Also optimizes packet construction: 1 memcpy instead of 13 write_volatile,
    /// and only 1 fence instead of 3.
    ///
    /// ```ignore
    /// for i in 0..100 {
    ///     queue.submit_batch(&kernel, grid, pool.get_kernargs(i));
    /// }
    /// queue.ring_doorbell();  // single MMIO write
    /// queue.wait_idle()?;
    /// ```
    pub fn submit_batch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        self.submit_batch_addr(kernel, grid, kernargs.gpu_addr());
    }

    /// Batch-optimized submit with raw kernarg address.
    ///
    /// Ring buffer overflow protection: spin-waits if ring is nearly full.
    pub fn submit_batch_addr(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernarg_addr: u64,
    ) {
        self.ensure_ring_space();
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // Build entire 64-byte AQL packet in a stack buffer, then memcpy once
        #[repr(C, packed)]
        struct AqlPkt {
            header: u16,        // 0x00
            setup: u16,         // 0x02
            wg_x: u16,          // 0x04
            wg_y: u16,          // 0x06
            wg_z: u16,          // 0x08
            reserved0: u16,     // 0x0A
            grid_x: u32,        // 0x0C
            grid_y: u32,        // 0x10
            grid_z: u32,        // 0x14
            private_seg: u32,   // 0x18
            group_seg: u32,     // 0x1C
            kernel_obj: u64,    // 0x20
            kernarg: u64,       // 0x28
            reserved2: u64,     // 0x30
            signal: u64,        // 0x38
        }

        // Header with INVALID type first — CP won't process until we flip header
        let real_header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        let pkt = AqlPkt {
            header: 1u16,   // INVALID — placeholder, overwritten below
            setup: 3,
            wg_x: kernel.workgroup_size[0] as u16,
            wg_y: kernel.workgroup_size[1] as u16,
            wg_z: kernel.workgroup_size[2] as u16,
            reserved0: 0,
            grid_x: grid[0],
            grid_y: grid[1],
            grid_z: grid[2],
            private_seg: 0,
            group_seg: kernel.lds_size,
            kernel_obj: kernel.descriptor_va,
            kernarg: kernarg_addr,
            reserved2: 0,
            signal: 0,
        };

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);

            // Single memcpy for the whole packet (header=INVALID so CP ignores it)
            std::ptr::copy_nonoverlapping(
                &pkt as *const AqlPkt as *const u8,
                base,
                64,
            );

            // SeqCst = mfence: flush WC buffers before making packet valid
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            // Atomically make packet valid by writing the real header
            std::ptr::write_volatile(base as *mut u16, real_header);

            // Update write pointer (no fence needed — x86 TSO guarantees
            // stores are visible in order, and Release fence above is sufficient)
            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // NO doorbell write — caller must call ring_doorbell()
        }
    }

    /// Ring the doorbell once after a batch of submit_batch() calls.
    /// This triggers CP to process all queued packets.
    pub fn ring_doorbell(&self) {
        unsafe {
            let write_idx = std::ptr::read_volatile(self.write_ptr_host);
            // fence to ensure all packet writes + write_ptr are visible before doorbell MMIO
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(self.doorbell_ptr, write_idx - 1);
        }
    }

    // =========================================================================
    // Ring buffer overflow protection
    // =========================================================================

    /// Ensure there is space in the AQL ring buffer before writing a new packet.
    /// 
    /// Spin-waits if `write_idx - read_idx >= ring_slots - 64`.
    /// The 64-slot margin provides headroom so we never overwrite an unprocessed
    /// packet. With a 4MB ring (65536 slots), this effectively never triggers
    /// under normal workloads, but prevents hard hangs if thousands of kernels
    /// are submitted faster than the GPU can consume them.
    ///
    /// On timeout (5s): panics instead of exit(99) so callers can catch_unwind.
    fn ensure_ring_space(&self) {
        let ring_slots = self.ring_size as u64 / 64;
        // Leave 64 slots of headroom to avoid overwriting in-flight packets
        let max_inflight = ring_slots - 64;
        let start = std::time::Instant::now();
        let mut last_log = 0u64;
        loop {
            let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if write_idx.wrapping_sub(read_idx) < max_inflight {
                return;
            }
            let elapsed_s = start.elapsed().as_secs();
            // Periodic progress log every 2s so we can see if GPU is still alive
            if elapsed_s >= last_log + 2 {
                last_log = elapsed_s;
                eprintln!(
                    "[KFD] ensure_ring_space: waiting {elapsed_s}s — \
                     write={write_idx} read={read_idx} inflight={} ring_slots={ring_slots}",
                    write_idx.wrapping_sub(read_idx)
                );
            }
            // Timeout after 5 seconds — panic (catchable) instead of exit(99)
            // to avoid leaving GPU in hung state after process death.
            if elapsed_s >= 5 {
                let msg = format!(
                    "[KFD] ensure_ring_space TIMEOUT (5s): GPU likely hung!\n\
                     write_idx={} read_idx={} inflight={} ring_slots={} max_inflight={}\n\
                     This indicates a GPU page fault or kernel hang.",
                    write_idx, read_idx, write_idx.wrapping_sub(read_idx),
                    ring_slots, max_inflight
                );
                eprintln!("{}", msg);
                panic!("{}", msg);
            }
            std::hint::spin_loop();
        }
    }

    pub fn wait_idle(&self) -> Result<(), String> {
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        self.wait_read_ptr(target)
    }

    /// Wait for all pending dispatches + memory fence.
    /// This is the SAFE way to synchronize before dropping GPU buffers.
    /// Ensures all GPU stores (including L2 writeback) are complete.
    pub fn synchronize(&self) -> Result<(), String> {
        self.wait_idle()?;
        // Memory fence to ensure CPU sees GPU writes
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // Small sleep to allow L2 cache writeback to complete
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }

    /// Wait for completion signal (poll amd_signal_t.value at offset 8)
    fn wait_signal(&self, signal: &GpuBuffer) -> Result<(), String> {
        let timeout_ns: u64 = 60_000_000_000; // 60 seconds (large GEMMs can be slow on first dispatch)
        let start = std::time::Instant::now();
        loop {
            // Read amd_signal_t.value at offset 8 (not offset 0 which is kind!)
            let val: i64 = signal.read_val(8);
            if val <= 0 {
                return Ok(());
            }
            if start.elapsed().as_nanos() as u64 > timeout_ns {
                return Err(format!("Kernel execution timeout (>{}s)", timeout_ns / 1_000_000_000));
            }
            std::hint::spin_loop();
        }
    }

    /// Fallback: wait by polling read pointer.
    /// Returns Err on timeout (5s) instead of exit(99) to allow graceful recovery.
    fn wait_read_ptr(&self, target: u64) -> Result<(), String> {
        let start = std::time::Instant::now();
        let mut last_log_s = 0u64;
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                return Ok(());
            }
            let elapsed = start.elapsed();
            let elapsed_s = elapsed.as_secs();
            // Progress log every 1s so we can see if GPU is making progress
            if elapsed_s >= last_log_s + 1 {
                last_log_s = elapsed_s;
                let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
                eprintln!(
                    "[KFD] wait_read_ptr: {elapsed_s}s — read={read_idx} target={target} \
                     write={write_idx} pending={}",
                    write_idx.wrapping_sub(read_idx)
                );
            }
            // 2026-08-31 动态超时（再次放宽，正确性优先）：60s + 每 pending dispatch 60s。
            // 用户指引：全部视为超时误报处理，正确性优先通过。pending=1 → 120s；
            // 10 并发 async → ~660s。代价：真卡时检测很慢，但避免误报优先。
            let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
            let pending = write_idx.wrapping_sub(read_idx);
            let timeout_ns: u64 = 60_000_000_000 + pending.saturating_mul(60_000_000_000);
            if elapsed.as_nanos() as u64 > timeout_ns {
                let msg = format!(
                    "[KFD] wait_read_ptr TIMEOUT ({}s): GPU hung! \
                     read={}, target={}, write={}, pending={}",
                    timeout_ns / 1_000_000_000, read_idx, target, write_idx, pending
                );
                eprintln!("{}", msg);
                // T0_DBG_TRAP=1: 读队列快照确认例外（MEMVIOL 等）——定位卡因。
                if std::env::var("T0_DBG_TRAP").is_ok() {
                    crate::kfd::ioctl::snapshot_queue_exceptions(&self.device);
                }
                // Return Err instead of exit(99) so tests can continue
                // and GPU resources can potentially be reclaimed by KFD driver.
                return Err(msg);
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // Optimized dispatch path — minimal overhead
    // =========================================================================

    /// Ultra-low-latency submit: skips ensure_ring_space, uses single Release fence.
    ///
    /// **Safety**: caller must guarantee ring buffer won't overflow (typical for
    /// benchmarks or pre-checked workloads). For production code, use `submit()`.
    ///
    /// Optimizations vs `submit()`:
    /// - 1× Release fence instead of 3× SeqCst (eliminates 2 x86 mfence @ ~33ns each)
    /// - No `ensure_ring_space()` check
    /// - Direct field writes without intermediate volatile reads where possible
    pub fn submit_fast(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
    ) {
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let slot_idx = write_idx & ring_mask;
        let pkt_offset = (slot_idx * 64) as usize;

        // AQL header: dispatch + barrier + AGENT fences (GPU-internal only, no PCIe sync)
        // AGENT scope saves ~10-20μs per dispatch vs SYSTEM scope by avoiding L2 writeback.
        let header: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_AGENT as u16) << 9) |
            ((HSA_FENCE_SCOPE_AGENT as u16) << 11);

        unsafe {
            let base = self.ring_buffer.host_ptr.add(pkt_offset);

            // Write packet body first (header last = atomic activation)
            // setup = dims(3) at +0x02
            std::ptr::write_volatile(base.add(0x02) as *mut u16, 3u16);
            // workgroup_size at +0x04, +0x06, +0x08
            std::ptr::write_volatile(base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(base.add(0x0A) as *mut u16, 0u16);
            // grid_size at +0x0C, +0x10, +0x14
            std::ptr::write_volatile(base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(base.add(0x14) as *mut u32, grid[2]);
            // private_segment_size + group_segment_size at +0x18, +0x1C
            std::ptr::write_volatile(base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(base.add(0x1C) as *mut u32, kernel.lds_size);
            // kernel_object (descriptor VA) at +0x20
            std::ptr::write_volatile(base.add(0x20) as *mut u64, kernel.descriptor_va);
            // kernarg_address at +0x28
            std::ptr::write_volatile(base.add(0x28) as *mut u64, kernargs.gpu_addr());
            // reserved + completion signal = 0
            std::ptr::write_volatile(base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(base.add(0x38) as *mut u64, 0u64);

            // Single Release fence: ensures all packet body writes are visible
            // before activating the header. On x86, this compiles to nothing
            // (x86 stores are naturally ordered) — only the final SeqCst for
            // the doorbell is needed.
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

            // Activate packet (header write)
            std::ptr::write_volatile(base as *mut u16, header);

            // SeqCst fence before doorbell write: this is the one fence we
            // truly need — it ensures the WC buffer is drained to memory
            // before the doorbell ring reaches the GPU's CP.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

            let new_write_idx = write_idx + 1;
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // Doorbell: GPU CP reads this to discover new packets
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
    }

    /// Ultra-low-latency wait: tight spin on read_dispatch_id.
    ///
    /// No `Instant::now()`, no `.elapsed()`, no progress logging.
    /// Pure volatile read + spin_loop_hint. Typical exit: < 1μs for empty kernels.
    ///
    /// **Warning**: hangs forever if GPU is stuck. For production, use `wait_idle()`.
    #[inline]
    pub fn wait_idle_spin(&self) {
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target {
                return;
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // GPU-accelerated buffer zeroing
    // =========================================================================

    /// Zero a GPU buffer using a GPU memset kernel.
    ///
    /// **~38x faster** than `buf.zero()` (CPU PCIe writes at ~25 GB/s):
    /// Uses `global_store_dwordx4` at VRAM bandwidth (~960 GB/s).
    ///
    /// The memset kernel is lazily built and cached on first call.
    ///
    /// ```ignore
    /// queue.gpu_zero(&d_output);  // zeros entire buffer on GPU
    /// queue.wait_idle()?;         // wait for completion
    /// ```
    pub fn gpu_zero(&self, buf: &GpuBuffer) {
        use std::sync::OnceLock;
        use crate::rdna3_asm::gfx11;
        use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

        let is_gfx1200 = buf.device.gfx_target_version >= 120000;
        let target_gfx = if is_gfx1200 { "gfx1200" } else { "gfx1100" };

        // Separate statics per target (kernel binary differs due to wait instructions)
        static MEMSET_KERNEL_GFX1100: OnceLock<GpuKernel> = OnceLock::new();
        static MEMSET_KA_BUF_GFX1100: OnceLock<GpuBuffer> = OnceLock::new();
        static MEMSET_KERNEL_GFX1200: OnceLock<GpuKernel> = OnceLock::new();
        static MEMSET_KA_BUF_GFX1200: OnceLock<GpuBuffer> = OnceLock::new();

        let (kernel_src, ka_buf_src) = if is_gfx1200 {
            (&MEMSET_KERNEL_GFX1200, &MEMSET_KA_BUF_GFX1200)
        } else {
            (&MEMSET_KERNEL_GFX1100, &MEMSET_KA_BUF_GFX1100)
        };

        let kernel = kernel_src.get_or_init(|| {
            let mut asm = crate::rdna3_asm::Rdna3Assembler::new();
            asm.set_target(if is_gfx1200 { "gfx1200" } else { "gfx1100" });

            // Kernarg: [ptr: u64(0), n_dwords: u32(8), pad: u32(12)]
            // SGPR: s[0:1]=kernarg_ptr, s2=workgroup_id_x (TGID_X_EN=1)
            // Load kernargs into s[4:7]
            let words = if is_gfx1200 {
                gfx11::s_load_dwordx4_gfx1200(4, 0, 0)
            } else {
                gfx11::s_load_dwordx4(4, 0, 0)
            };
            asm.emit(words[0]); asm.emit(words[1]);
            asm.wait_kmcnt(0);

            // global_id = workgroup_id_x * 256 + thread_id
            asm.emit(gfx11::v_mov_b32_from_sgpr(1, 2));  // v1 = wg_id_x
            asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));     // v1 *= 256
            let add = gfx11::v_add_co_u32_vcc(1, 1, 0);  // v1 += v0
            asm.emit(add[0]); asm.emit(add[1]);

            // Bounds: if global_id * 4 >= n_dwords → mask off
            asm.emit(gfx11::v_lshlrev_b32(2, 2, 1));     // v2 = global_id * 4
            asm.emit(gfx11::v_mov_b32_from_sgpr(3, 6));   // v3 = n_dwords
            asm.emit(gfx11::v_cmp_lt_u32(2, 3));          // vcc = v2 < v3
            asm.emit(gfx11::s_and_saveexec_b32_vcc(8));   // mask off OOB lanes

            // addr = ptr + global_id * 16
            asm.emit(gfx11::v_lshlrev_b32(4, 4, 1));     // v4 = byte offset
            asm.emit(gfx11::v_mov_b32_from_sgpr(5, 4));   // v5 = ptr_lo
            asm.emit(gfx11::v_mov_b32_from_sgpr(6, 5));   // v6 = ptr_hi
            let al = gfx11::v_add_co_u32_vcc(5, 5, 4);
            asm.emit(al[0]); asm.emit(al[1]);
            let ah = gfx11::v_add_co_ci_u32_zero_vcc(6, 6);
            asm.emit(ah[0]); asm.emit(ah[1]);

            // Write 16 bytes of zeros
            asm.emit(gfx11::v_mov_b32_imm(10, 0));
            asm.emit(gfx11::v_mov_b32_imm(11, 0));
            asm.emit(gfx11::v_mov_b32_imm(12, 0));
            asm.emit(gfx11::v_mov_b32_imm(13, 0));
            asm.global_store_dwordx4(5, 10, 0);

            asm.wait_loadcnt(0);
            asm.wait_vscnt(0);
            asm.emit(gfx11::S_ENDPGM);

            let co = AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
                name: "gpu_memset_zero".into(),
                lds_size: 0, kernarg_size: 16,
                vgpr_count: 16, sgpr_count: 16,
                workgroup_size_x: 256, workgroup_size_y: 1, workgroup_size_z: 1,
                scratch_size: 0,
                target_gfx: target_gfx.into(),
            });
            let hsaco = co.to_code_object_llvm().expect("gpu_memset LLVM build");
            GpuKernel::load(&buf.device, &hsaco, &KernelLoadConfig {
                lds_size: 0,
                workgroup_size: [256, 1, 1],
            }).expect("gpu_memset kernel load")
        });

        // Prepare kernargs: [ptr(8), n_dwords(4), pad(4)]
        let n_dwords = (buf.size / 4) as u32;
        ka_buf_src.get_or_init(|| {
            buf.device.alloc_uncached(256).expect("memset ka buf")
        });
        let ka_buf = ka_buf_src.get().unwrap();

        // Write kernargs directly
        let mut ka_data = [0u8; 16];
        ka_data[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka_data[8..12].copy_from_slice(&n_dwords.to_le_bytes());
        ka_buf.write(&ka_data);

        // Grid: ceil(n_dwords / 4 / 256) * 256 threads
        let threads_needed = ((n_dwords as usize + 3) / 4 + 255) / 256 * 256;
        let grid = [threads_needed as u32, 1, 1];

        self.submit(kernel, grid, ka_buf);
    }

    // =========================================================================
    // PM4 hardware synchronization primitives (Mega-IB pipeline)
    // =========================================================================

    /// Submit pure PM4 commands wrapped in a VENDOR_SPECIFIC AQL packet.
    /// Used for compute_barrier() + release_mem() synchronization
    /// that replaces the old wait_idle() pattern.
    pub fn submit_pm4(&mut self, pm4_cmds: &[u32]) -> Result<(), String> {
        if pm4_cmds.is_empty() {
            return Ok(());
        }
        // Ensure PM4 IB buffer is allocated
        if self.pm4_ib.is_none() {
            self.pm4_ib = Some(self.device.alloc_uncached(PM4_IB_SIZE)?);
            self.pm4_ib_offset = 0;
        }
        let pm4_byte_size = pm4_cmds.len() * 4;
        if self.pm4_ib_offset + pm4_byte_size > PM4_IB_SIZE {
            self.pm4_ib_offset = 0; // wrap around
        }
        let ib = self.pm4_ib.as_ref().unwrap();
        let ib_byte_offset = self.pm4_ib_offset;

        // Write PM4 commands to IB
        unsafe {
            let dst = ib.host_ptr.add(ib_byte_offset) as *mut u32;
            for (i, &dword) in pm4_cmds.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), dword);
            }
            // Read-back to flush write-combine buffers
            let _ = std::ptr::read_volatile(dst.add(pm4_cmds.len() - 1));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let ib_va = ib.gpu_addr() + ib_byte_offset as u64;
        self.pm4_ib_offset += pm4_byte_size;

        // Write VENDOR_SPECIFIC AQL packet
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let slot = write_idx & ring_mask;
        let base = unsafe { self.ring_buffer.host_ptr.add((slot * 64) as usize) };

        // Header: barrier=1 (wait for prior compute), fence_scope=SYSTEM
        let header: u16 = HSA_PACKET_TYPE_VENDOR_SPECIFIC |
            (1 << 8) |  // barrier
            (HSA_FENCE_SCOPE_SYSTEM << 9) |
            (HSA_FENCE_SCOPE_SYSTEM << 11);

        // IB PACKET3 command: INDIRECT_BUFFER pointing to our PM4 commands
        let ib_pkt3 = (3u32 << 30) |
            (((3u32 - 1) & 0x3FFF) << 16) |  // 3 body dwords for IB: addr_lo, addr_hi, size|valid
            (PACKET3_INDIRECT_BUFFER << 8);

        unsafe {
            std::ptr::write_volatile(base as *mut u16, 1u16); // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_bytes(base.add(2), 0, 62);
            // VS packet layout: [0]=header, [2]=?, [4]=IB_pkt3, [8]=addr_lo, [12]=addr_hi, [16]=size|valid
            std::ptr::write_volatile(base.add(2)  as *mut u16, 1u16);
            std::ptr::write_volatile(base.add(4)  as *mut u32, ib_pkt3);
            std::ptr::write_volatile(base.add(8)  as *mut u32, ib_va as u32);
            std::ptr::write_volatile(base.add(12) as *mut u32, (ib_va >> 32) as u32);
            std::ptr::write_volatile(base.add(16) as *mut u32,
                pm4_cmds.len() as u32 | INDIRECT_BUFFER_VALID);
            std::ptr::write_volatile(base.add(20) as *mut u32, 10u32); // padding
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(base as *mut u16, header); // header LAST (atomically valid)

            // Ring doorbell — CRITICAL: write new_write_idx (NOT -1) to wake CP
            // after queue drain. When read_ptr == old_write_ptr, doorbell must be 
            // strictly greater than what MEC last consumed.
            let new_write_idx = write_idx + 1;
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }
        Ok(())
    }

    /// Lock-free VRAM polling: wait for GPU to write seqno >= target via RELEASE_MEM.
    /// Replaces wait_idle() — never drains the AQL queue, avoids CP sleep/wakeup bug.
    pub fn wait_vram_seqno(sync_buf: &GpuBuffer, target_seqno: u32) -> Result<(), String> {
        let ptr = sync_buf.host_ptr as *const std::sync::atomic::AtomicU32;
        let atomic_ptr = unsafe { &*ptr };
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);
        loop {
            let current = atomic_ptr.load(std::sync::atomic::Ordering::Acquire);
            if current >= target_seqno {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(format!(
                    "VRAM seqno timeout: expected >= {}, got {} after {:?}",
                    target_seqno, current, start.elapsed()
                ));
            }
            std::hint::spin_loop();
        }
    }

    // =========================================================================
    // PM4-in-AQL: embed PM4 PACKET3 commands inside AQL VENDOR_SPECIFIC packets
    // =========================================================================

    /// Dispatch a kernel via PM4-in-AQL hybrid approach.
    ///
    /// Submits TWO AQL packets to the queue:
    ///   1. VENDOR_SPECIFIC: Contains INDIRECT_BUFFER with register setup PM4
    ///      (ACQUIRE_MEM + SET_SH_REG for all compute registers)
    ///   2. KERNEL_DISPATCH: Native AQL dispatch packet for actual kernel launch
    ///      (Uses MES's proven dispatch path + amd_signal_t completion)
    ///
    /// This hybrid approach is more reliable than pure PM4 dispatch because:
    /// - Register setup uses PM4 for fine-grained control (matching tinygrad)
    /// - Kernel launch uses native AQL for proper MES scheduling and completion
    /// - No IB-reuse race: VENDOR_SPECIFIC IB only contains register writes
    pub fn dispatch_pm4(
        &mut self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        signal: Option<&GpuBuffer>,
    ) -> Result<(), String> {
        // Ensure PM4 IB buffer is allocated
        if self.pm4_ib.is_none() {
            self.pm4_ib = Some(self.device.alloc_uncached(PM4_IB_SIZE)?);
            self.pm4_ib_offset = 0;
        }

        // ── Step 1: Build register setup PM4 commands ───────────────────────
        let mut pm4 = Pm4CmdBuilder::new();

        // ACQUIRE_MEM — invalidate GPU caches (GFX10+ format with GCR_CNTL)
        // DIAG (T0_NO_ACQUIRE=1): skip ACQUIRE_MEM — suspect it blocks KD start
        // (VS executes ACQUIRE_MEM, which may stall waiting for cache flush).
        if std::env::var("T0_NO_ACQUIRE").is_err() {
            pm4.acquire_mem_gfx10();
        }

        // Set shader program address: COMPUTE_PGM_LO / COMPUTE_PGM_HI
        pm4.set_sh_reg(REG_COMPUTE_PGM_LO, &[
            (kernel.code_entry_va >> 8) as u32,
            (kernel.code_entry_va >> 40) as u32,
        ]);

        // Set RSRC1/RSRC2 (force rsrc1.priv=1 on GFX11 to workaround CWSR — tinygrad pattern)
        pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC1, &[
            kernel.rsrc1 | (1 << 20),  // rsrc1.priv = 1 (GFX11 CWSR workaround)
            kernel.rsrc2,
        ]);

        // Set RSRC3 (GFX11 specific — wave slots / scratch)
        pm4.set_sh_reg(REG_COMPUTE_PGM_RSRC3, &[0]);

        // Set TMPRING_SIZE
        pm4.set_sh_reg(REG_COMPUTE_TMPRING_SIZE, &[0]);

        // Set RESTART_X/Y/Z = 0,0,0
        pm4.set_sh_reg(REG_COMPUTE_RESTART_X, &[0, 0, 0]);

        // Set USER_DATA_0 = kernargs pointer (lo, hi)
        pm4.set_sh_reg(REG_COMPUTE_USER_DATA_0, &[
            kernargs.gpu_addr() as u32,
            (kernargs.gpu_addr() >> 32) as u32,
        ]);

        // Set RESOURCE_LIMITS = 0
        pm4.set_sh_reg(REG_COMPUTE_RESOURCE_LIMITS, &[0]);

        // Set START_X/Y/Z = 0,0,0 + workgroup sizes (NUM_THREAD_X/Y/Z) + padding
        pm4.set_sh_reg(REG_COMPUTE_START_X, &[
            0, 0, 0,                        // start x, y, z
            kernel.workgroup_size[0],        // num_thread_x
            kernel.workgroup_size[1],        // num_thread_y
            kernel.workgroup_size[2],        // num_thread_z
            0, 0,                            // padding
        ]);

        let pm4_cmds = pm4.finish();

        // ── Step 2: Write PM4 register setup to IB buffer ───────────────────
        let ib_byte_offset = self.pm4_ib_offset;
        let pm4_byte_size = pm4_cmds.len() * 4;

        if ib_byte_offset + pm4_byte_size > PM4_IB_SIZE {
            self.pm4_ib_offset = 0; // wrap around
        }
        let ib_byte_offset = self.pm4_ib_offset;
        let ib = self.pm4_ib.as_ref().unwrap();

        unsafe {
            let dst = ib.host_ptr.add(ib_byte_offset) as *mut u32;
            for (i, &dword) in pm4_cmds.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), dword);
            }
            // Read-back to flush write-combine buffers before ringing doorbell
            let _ = std::ptr::read_volatile(dst.add(pm4_cmds.len() - 1));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let ib_va = ib.gpu_addr() + ib_byte_offset as u64;
        self.pm4_ib_offset += pm4_byte_size;

        // ── Step 3+4: Write VS (reg setup) and KD (kernel launch) together ──
        // Pre-write both AQL slots, ring doorbell once for KD, wait for KD completion.
        // This avoids intermediate wait on VS read_ptr which can stall after 12+ packets.
        let ring_mask = (self.ring_size as u64 / 64) - 1;
        let vs_write_idx = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let kd_write_idx = vs_write_idx + 1;

        // ── Write VENDOR_SPECIFIC slot ────────────────────────────────────────
        let vs_slot = vs_write_idx & ring_mask;
        let vs_base = unsafe { self.ring_buffer.host_ptr.add((vs_slot * 64) as usize) };

        let vs_aql_hdr: u16 =
            HSA_PACKET_TYPE_VENDOR_SPECIFIC |
            (1 << 8) |
            (HSA_FENCE_SCOPE_SYSTEM << 9) |
            (HSA_FENCE_SCOPE_SYSTEM << 11);

        let ib_pkt3 = (3u32 << 30) | ((2u32 & 0x3FFF) << 16) | (PACKET3_INDIRECT_BUFFER << 8);

        unsafe {
            std::ptr::write_volatile(vs_base as *mut u16, 1u16);  // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_bytes(vs_base.add(2), 0, 62);
            std::ptr::write_volatile(vs_base.add(2)  as *mut u16, 1u16);
            std::ptr::write_volatile(vs_base.add(4)  as *mut u32, ib_pkt3);
            std::ptr::write_volatile(vs_base.add(8)  as *mut u32, ib_va as u32);
            std::ptr::write_volatile(vs_base.add(12) as *mut u32, (ib_va >> 32) as u32);
            std::ptr::write_volatile(vs_base.add(16) as *mut u32,
                pm4_cmds.len() as u32 | INDIRECT_BUFFER_VALID);
            std::ptr::write_volatile(vs_base.add(20) as *mut u32, 10u32);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(vs_base as *mut u16, vs_aql_hdr); // header LAST
        }

        // ── Write KERNEL_DISPATCH slot ────────────────────────────────────────
        let kd_slot = kd_write_idx & ring_mask;
        let kd_base = unsafe { self.ring_buffer.host_ptr.add((kd_slot * 64) as usize) };

        let signal_va = if let Some(sig) = signal {
            unsafe { std::ptr::write_bytes(sig.host_ptr, 0, 64); }
            sig.write_val::<u64>(0, 1);
            sig.write_val::<i64>(8, 1);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            sig.gpu_addr()
        } else {
            0u64
        };

        let kd_hdr: u16 =
            (HSA_PACKET_TYPE_KERNEL_DISPATCH as u16) |
            (1 << 8) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 9) |
            ((HSA_FENCE_SCOPE_SYSTEM as u16) << 11);

        unsafe {
            std::ptr::write_volatile(kd_base.add(0x00) as *mut u16, 1u16);  // INVALID first
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(kd_base.add(0x02) as *mut u16, 3u16);
            std::ptr::write_volatile(kd_base.add(0x04) as *mut u16, kernel.workgroup_size[0] as u16);
            std::ptr::write_volatile(kd_base.add(0x06) as *mut u16, kernel.workgroup_size[1] as u16);
            std::ptr::write_volatile(kd_base.add(0x08) as *mut u16, kernel.workgroup_size[2] as u16);
            std::ptr::write_volatile(kd_base.add(0x0A) as *mut u16, 0u16);
            std::ptr::write_volatile(kd_base.add(0x0C) as *mut u32, grid[0]);
            std::ptr::write_volatile(kd_base.add(0x10) as *mut u32, grid[1]);
            std::ptr::write_volatile(kd_base.add(0x14) as *mut u32, grid[2]);
            std::ptr::write_volatile(kd_base.add(0x18) as *mut u32, 0u32);
            std::ptr::write_volatile(kd_base.add(0x1C) as *mut u32, kernel.lds_size);
            std::ptr::write_volatile(kd_base.add(0x20) as *mut u64, kernel.descriptor_va);
            std::ptr::write_volatile(kd_base.add(0x28) as *mut u64, kernargs.gpu_addr());
            std::ptr::write_volatile(kd_base.add(0x30) as *mut u64, 0u64);
            std::ptr::write_volatile(kd_base.add(0x38) as *mut u64, signal_va);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            std::ptr::write_volatile(kd_base.add(0x00) as *mut u16, kd_hdr); // header LAST
        }

        // ── Ring doorbell: update write_ptr to cover BOTH VS+KD ──────────────
        let new_write_idx = kd_write_idx + 1;
        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.write_ptr_host, new_write_idx);
            // tinygrad: write_ptr = put_value, doorbell = put_value - 1
            std::ptr::write_volatile(self.doorbell_ptr, new_write_idx - 1);
        }

        // ── Wait for kernel completion ────────────────────────────────────────
        if let Some(sig) = signal {
            self.wait_signal(sig)?;
        } else {
            // No signal: poll read_ptr until both VS+KD consumed
            let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
            self.wait_read_ptr(target)?;
        }

        Ok(())
    }
}


// =============================================================================
// Pm4CmdBuilder — builds PM4 PACKET3 command sequences
// =============================================================================


/// PM4 command builder for constructing PACKET3 sequences
impl Drop for AqlQueue {
    fn drop(&mut self) {
        // Drain queue: wait for all in-flight dispatches to complete (500ms timeout)
        // This prevents DESTROY_QUEUE from killing active GPU work, which can
        // leave the KFD driver in a dirty state for the next KfdDevice::open().
        let target = unsafe { std::ptr::read_volatile(self.write_ptr_host) };
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(500);
        loop {
            let read_idx = unsafe { std::ptr::read_volatile(self.read_ptr_host) };
            if read_idx >= target { break; }
            if start.elapsed() > timeout {
                eprintln!("[KFD] WARNING: AqlQueue {} drain timeout (read={} target={})",
                    self.queue_id, read_idx, target);
                break;
            }
            std::hint::spin_loop();
        }
        let mut args = KfdDestroyQueueArgs { queue_id: self.queue_id, pad: 0 };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_DESTROY_QUEUE,
            &mut args as *mut _ as *mut u8);
        // Unmap doorbell — must use original mmap base address and size!
        // doorbell_ptr is offset within the mmap region, NOT the mmap return value.
        unsafe { munmap(self.doorbell_mmap_base, self.doorbell_mmap_size); }
    }
}

// =============================================================================
// Pm4Queue — PM4 compute queue with raw PACKET3 dispatch
// =============================================================================

