//! DispatchPool + GpuMemset — dispatch pool and GPU memory operations.

use std::sync::Arc;
use super::device::KfdDevice;
use super::buffer::GpuBuffer;
use super::aql::AqlQueue;
use super::kernel::{GpuKernel, KernelLoadConfig};
pub struct DispatchPool {
    /// Single reusable signal buffer
    pub signal: GpuBuffer,
    /// Auto-growing ring of kernargs buffers (Mutex for thread-safe interior mutability)
    kernargs_ring: std::sync::Mutex<Vec<GpuBuffer>>,
    /// Device reference for on-demand allocation
    device: Arc<KfdDevice>,
}

impl DispatchPool {
    /// Create a pool. `initial_slots` kernargs buffers are pre-allocated.
    /// Additional slots are allocated on-demand when accessed.
    /// Pass 0 for default (1024 initial slots, auto-grows beyond that).
    pub fn new(device: &Arc<KfdDevice>, initial_slots: usize) -> Result<Self, String> {
        let signal = device.alloc_signal()?;
        let n = if initial_slots == 0 { 1024 } else { initial_slots };
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            ring.push(device.alloc_uncached(256)?);
        }
        Ok(Self {
            signal,
            kernargs_ring: std::sync::Mutex::new(ring),
            device: Arc::clone(device),
        })
    }

    /// Ensure slot `idx` exists, growing the pool if necessary.
    fn ensure_slot(&self, idx: usize) {
        let mut ring = self.kernargs_ring.lock().unwrap();
        while idx >= ring.len() {
            // Allocate new slot on demand
            match self.device.alloc_uncached(256) {
                Ok(buf) => ring.push(buf),
                Err(e) => panic!("DispatchPool: failed to grow to slot {}: {}", idx, e),
            }
        }
    }

    /// Get kernargs buffer for slot `idx`. Auto-allocates if slot doesn't exist.
    pub fn get_kernargs(&self, idx: usize) -> &GpuBuffer {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        // Safety: buffer lives as long as the pool (never removed from Vec)
        unsafe { &*(ring.get(idx).unwrap() as *const GpuBuffer) }
    }

    /// Write kernargs data to slot `idx` and return the buffer ref.
    /// Auto-allocates if slot doesn't exist yet.
    ///
    /// **Direction 2 (buffer-reuse fix)**: rewrites the ENTIRE slot every
    /// dispatch. The slot is zeroed first so bytes beyond `data.len()` can
    /// never retain stale pointer values from a previous dispatch (a kernel
    /// whose declared kernarg_size exceeds the current write would otherwise
    /// read a stale VA → page fault / KD stall). Kernargs always carry the
    /// current GPU VAs and zeros elsewhere.
    pub fn write_kernargs(&self, idx: usize, data: &[u8]) -> &GpuBuffer {
        self.ensure_slot(idx);
        let ring = self.kernargs_ring.lock().unwrap();
        let buf = unsafe { &*(ring.get(idx).unwrap() as *const GpuBuffer) };
        debug_assert!(data.len() <= buf.size, "kernargs overflow: {} > {}", data.len(), buf.size);
        // 1) wipe the whole slot (256B) — kills any stale tail from prior dispatches
        buf.zero();
        // 2) write the fresh kernargs (current GPU VAs)
        buf.write(data);
        buf
    }

    /// Dispatch with pre-allocated signal. Resets signal, dispatches, waits.
    pub fn dispatch(
        &self,
        queue: &AqlQueue,
        kernel: &GpuKernel,
        grid: [u32; 3],
        ka_idx: usize,
    ) -> Result<(), String> {
        self.signal.write_val::<u64>(0, 1);
        self.signal.write_val::<i64>(8, 1);
        // WC readback drain: ensure signal reaches GPU VRAM before doorbell
        // geisYaO: "SFENCE 不保证 PCIe 可见性，必须 readback drain"
        unsafe { let _ = std::ptr::read_volatile(self.signal.host_ptr); }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let ka = self.get_kernargs(ka_idx);
        queue.dispatch_signal(kernel, grid, ka, Some(&self.signal))
    }

    /// Current number of allocated slots.
    pub fn len(&self) -> usize { self.kernargs_ring.lock().unwrap().len() }

    /// Capacity in slots (same as len since auto-growing).
    pub fn capacity(&self) -> usize { self.kernargs_ring.lock().unwrap().capacity() }
}

// =============================================================================
// GpuMemset — GPU-side memory zeroing (replaces CPU PCIe writes)
// =============================================================================

/// GPU memset: zeros memory at VRAM bandwidth (~960 GB/s) vs CPU→PCIe (~25 GB/s).
/// For 16 MB: GPU ≈ 0.017 ms vs CPU ≈ 0.64 ms → 37× faster.
pub struct GpuMemset {
    kernel: GpuKernel,
    ka_buf: GpuBuffer,
}

impl GpuMemset {
    /// Build and load the GPU memset kernel.
    pub fn new(device: &Arc<KfdDevice>) -> Result<Self, String> {
        use crate::rdna3_asm::{Rdna3Assembler, gfx11};
        use crate::rdna3_code_object::{AmdGpuCodeObject, KernelConfig};

        let is_gfx1200 = device.gfx_target_version >= 120000;
        let target_gfx = if is_gfx1200 { "gfx1200" } else { "gfx1100" };

        let mut asm = Rdna3Assembler::new();
        asm.set_target(target_gfx);
        // s[0:1] = kernarg_ptr, s2 = wg_id_x, v0 = thread_id_x
        // Kernarg: [ptr:u64, n_dw4:u32, pad:u32]

        // Load kernargs -> s[4:7]
        let [w0, w1] = if is_gfx1200 {
            gfx11::s_load_dwordx4_gfx1200(4, 0, 0)
        } else {
            gfx11::s_load_dwordx4(4, 0, 0)
        };
        asm.emit(w0); asm.emit(w1);
        asm.wait_kmcnt(0);

        // global_id = wg_id_x * 256 + thread_id
        asm.emit(gfx11::v_mov_b32_from_sgpr(1, 2));
        asm.emit(gfx11::v_lshlrev_b32(1, 8, 1));
        asm.emit(gfx11::v_add_u32(1, 1, 0));

        // EXEC mask: active if global_id < n_dw4
        asm.emit(gfx11::v_mov_b32_from_sgpr(2, 6));
        asm.emit(gfx11::v_cmp_lt_u32(1, 2));

        // addr = ptr + global_id * 16
        asm.emit(gfx11::v_mov_b32_from_sgpr(3, 4));
        asm.emit(gfx11::v_mov_b32_from_sgpr(4, 5));
        asm.emit(gfx11::v_lshlrev_b32(5, 4, 1));
        let [a0, a1] = gfx11::v_add_co_u32_vcc(3, 3, 5);
        asm.emit(a0); asm.emit(a1);
        let [b0, b1] = gfx11::v_add_co_ci_u32_zero_vcc(4, 4);
        asm.emit(b0); asm.emit(b1);

        // v[6:9] = 0
        asm.emit(gfx11::v_mov_b32_imm(6, 0));
        asm.emit(gfx11::v_mov_b32_imm(7, 0));
        asm.emit(gfx11::v_mov_b32_imm(8, 0));
        asm.emit(gfx11::v_mov_b32_imm(9, 0));

        // store zeros (only active EXEC lanes)
        asm.global_store_dwordx4(3, 6, 0);

        asm.wait_vscnt(0);
        asm.emit(gfx11::S_ENDPGM);

        let co = AmdGpuCodeObject::from_assembler(&asm, KernelConfig {
            name: "gpu_memset_zero".to_string(),
            lds_size: 0,
            kernarg_size: 16,
            vgpr_count: 10,
            sgpr_count: 8,
            workgroup_size_x: 256,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            scratch_size: 0,
            target_gfx: target_gfx.into(),
        });

        let hsaco = co.to_code_object_llvm().map_err(|e| format!("memset LLVM: {e}"))?;
        let kernel = GpuKernel::load(device, &hsaco, &KernelLoadConfig {
            lds_size: 0,
            workgroup_size: [256, 1, 1],
        })?;
        let ka_buf = device.alloc_uncached(256)?;
        Ok(Self { kernel, ka_buf })
    }

    /// Zero `n_bytes` of `buf`. Waits for completion via signal.
    pub fn zero(
        &self,
        queue: &AqlQueue,
        buf: &GpuBuffer,
        n_bytes: usize,
        signal: &GpuBuffer,
    ) -> Result<(), String> {
        let n_dw4 = ((n_bytes + 15) / 16) as u32;
        let n_wg = (n_dw4 + 255) / 256;
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&n_dw4.to_le_bytes());
        self.ka_buf.write(&ka);
        signal.write_val::<u64>(0, 1);
        signal.write_val::<i64>(8, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        queue.dispatch_signal(&self.kernel, [n_wg * 256, 1, 1], &self.ka_buf, Some(signal))
    }

    /// Zero without signal — call queue.wait_idle() after.
    pub fn zero_async(&self, queue: &AqlQueue, buf: &GpuBuffer, n_bytes: usize) {
        let n_dw4 = ((n_bytes + 15) / 16) as u32;
        let n_wg = (n_dw4 + 255) / 256;
        let mut ka = [0u8; 16];
        ka[0..8].copy_from_slice(&buf.gpu_addr().to_le_bytes());
        ka[8..12].copy_from_slice(&n_dw4.to_le_bytes());
        self.ka_buf.write(&ka);
        queue.submit(&self.kernel, [n_wg * 256, 1, 1], &self.ka_buf);
    }
}
