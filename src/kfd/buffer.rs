//! GpuBuffer — RAII GPU memory with automatic cleanup.

use std::sync::Arc;
use super::device::KfdDevice;
use super::ioctl::{munmap, ioctl_safe, KfdUnmapMemoryArgs, KfdFreeMemoryArgs,
    AMDKFD_IOC_UNMAP_MEMORY, AMDKFD_IOC_FREE_MEMORY};

/// GPU memory buffer with automatic lifecycle management
pub struct GpuBuffer {
    pub(crate) handle: u64,
    pub va_addr: u64,
    pub host_ptr: *mut u8,
    pub size: usize,
    pub device: Arc<KfdDevice>,
}

// GpuBuffer is Send+Sync because it wraps GPU memory accessed via mmap
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Write data from CPU to GPU buffer
    pub fn write(&self, data: &[u8]) {
        assert!(data.len() <= self.size, "write overflow: {} > {}", data.len(), self.size);
        unsafe {
            let dst = self.host_ptr;
            let src = data.as_ptr();
            let n8 = data.len() / 8;
            let rem = data.len() % 8;
            for i in 0..n8 {
                let val = std::ptr::read_unaligned(src.add(i * 8) as *const u64);
                std::ptr::write_volatile(dst.add(i * 8) as *mut u64, val);
            }
            let base = n8 * 8;
            for i in 0..rem {
                std::ptr::write_volatile(dst.add(base + i), *src.add(base + i));
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let _ = std::ptr::read_volatile(self.host_ptr);
        }
    }

    /// Read data from GPU buffer to CPU
    pub fn read(&self, buf: &mut [u8]) {
        assert!(buf.len() <= self.size, "read overflow: {} > {}", buf.len(), self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_ptr, buf.as_mut_ptr(), buf.len());
        }
    }

    /// Write typed data
    pub fn write_val<T: Copy>(&self, offset: usize, val: T) {
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe {
            let ptr = self.host_ptr.add(offset) as *mut T;
            std::ptr::write_volatile(ptr, val);
        }
    }

    /// Read typed data
    pub fn read_val<T: Copy>(&self, offset: usize) -> T {
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe {
            let ptr = self.host_ptr.add(offset) as *const T;
            std::ptr::read_volatile(ptr)
        }
    }

    /// GPU virtual address
    pub fn gpu_addr(&self) -> u64 {
        self.va_addr
    }

    /// Write bytes at a specific offset with bounds checking.
    pub fn write_bytes(&self, offset: usize, data: &[u8]) {
        assert!(offset + data.len() <= self.size,
            "write_bytes overflow: offset={} len={} size={}", offset, data.len(), self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.host_ptr.add(offset), data.len());
        }
    }

    /// Read bytes from a specific offset with bounds checking.
    pub fn read_bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(offset + len <= self.size,
            "read_bytes overflow: offset={} len={} size={}", offset, len, self.size);
        let mut buf = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_ptr.add(offset), buf.as_mut_ptr(), len);
        }
        buf
    }

    /// Zero the buffer (volatile writes + PCIe readback drain)
    pub fn zero(&self) {
        unsafe {
            let p = self.host_ptr as *mut u64;
            let n = self.size / 8;
            for i in 0..n {
                std::ptr::write_volatile(p.add(i), 0u64);
            }
            let rem_base = n * 8;
            for i in 0..(self.size % 8) {
                std::ptr::write_volatile(self.host_ptr.add(rem_base + i), 0u8);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let _ = std::ptr::read_volatile(self.host_ptr);
        }
    }

    /// Create a sub-region view of this buffer (for pool allocation).
    /// The sub-region has handle=0, so Drop won't call KFD free.
    pub fn sub_region(parent: &GpuBuffer, offset: usize, size: usize) -> GpuBuffer {
        assert!(offset + size <= parent.size, "sub_region overflow: {}+{} > {}", offset, size, parent.size);
        GpuBuffer {
            handle: 0,
            va_addr: parent.va_addr + offset as u64,
            host_ptr: unsafe { parent.host_ptr.add(offset) },
            size,
            device: parent.device.clone(),
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        unsafe {
            munmap(self.host_ptr, self.size);
        }
        let mut gpu_ids = [self.device.gpu_id];
        let mut unmap = KfdUnmapMemoryArgs {
            handle: self.handle,
            device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_UNMAP_MEMORY,
            &mut unmap as *mut _ as *mut u8);
        let mut free = KfdFreeMemoryArgs { handle: self.handle };
        let _ = ioctl_safe(self.device.kfd_fd, AMDKFD_IOC_FREE_MEMORY,
            &mut free as *mut _ as *mut u8);
    }
}

/// Scan kernarg bytes for obviously invalid GPU pointers.
#[cfg(debug_assertions)]
pub(crate) fn validate_kernargs_bytes(ka: &[u8], ka_size: usize) {
    let mut offset = 0;
    while offset + 8 <= ka_size {
        let val = u64::from_le_bytes(ka[offset..offset + 8].try_into().unwrap());
        if val <= 0xFFFF {
            offset += 8;
            continue;
        }
        if val > 0xFFFF && val < 0x1000_0000 {
            eprintln!(
                "[KFD VALIDATE] Warning: kernarg offset {} has value 0x{:016X} \
                 — suspicious (too small for GPU VA, too large for u32 param)",
                offset, val
            );
        }
        if val > 0x0000_FFFF_FFFF_FFFF {
            // 2026-08-30: 收紧 panic 条件——u32 标量对（如 split_k_shift +
            // y_split_stride）组合成 8 字节值时会 > 48 位（例：y_split_stride
            // = 0x10000 → 组合 0x0001000000000000），被误判为非法指针而 panic
            // （vs_gemm_gen 的 gemm ka 合法值触发）。真 GPU 指针的低 32 位
            // 几乎总是 ≥ 0x10000（VA 高地址）；低 32 位极小的 >48 位值按
            // 标量组合处理（warning 而非 panic）。
            if (val & 0xFFFF_FFFF) >= 0x10000 {
                panic!(
                    "[KFD SAFETY] Kernarg offset {} contains 0x{:016X} — \
                     outside 48-bit VA space. Likely a host pointer or uninitialized memory!\n\
                     This WILL cause a GPU page fault and hard hang.",
                    offset, val
                );
            } else {
                eprintln!(
                    "[KFD VALIDATE] Warning: kernarg offset {} has 0x{:016X} \
                     (low32=0x{:08X}) — likely a u32 scalar pair, not a pointer",
                    offset, val, (val & 0xFFFF_FFFF)
                );
            }
        }
        offset += 8;
    }
}
