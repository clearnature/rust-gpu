//! KFD ioctl constants, argument structs, and low-level syscall wrappers.
//!
//! All ioctl numbers and C-compatible structs that map directly to kernel's `kfd_ioctl.h`.

use std::os::unix::io::RawFd;

// =============================================================================
// KFD IOCTL numbers (Linux _IOC encoding, type='K'=0x4B)
// =============================================================================

pub(crate) const AMDKFD_IOC_GET_VERSION: u64       = 0x80084B01;
pub(crate) const AMDKFD_IOC_CREATE_QUEUE: u64      = 0xC0604B02; // sizeof=96 (matches tinygrad)
pub(crate) const AMDKFD_IOC_DESTROY_QUEUE: u64     = 0xC0084B03;
pub(crate) const AMDKFD_IOC_ACQUIRE_VM: u64        = 0x40084B15;
pub(crate) const AMDKFD_IOC_ALLOC_MEMORY: u64      = 0xC0284B16;
pub(crate) const AMDKFD_IOC_FREE_MEMORY: u64       = 0x40084B17;
pub(crate) const AMDKFD_IOC_MAP_MEMORY: u64        = 0xC0184B18;
pub(crate) const AMDKFD_IOC_UNMAP_MEMORY: u64      = 0xC0184B19;
pub(crate) const AMDKFD_IOC_CREATE_EVENT: u64      = 0xC02C4B08;
pub(crate) const AMDKFD_IOC_WAIT_EVENTS: u64       = 0xC0204B0B;
pub(crate) const AMDKFD_IOC_RUNTIME_ENABLE: u64    = 0xC0104B25; // sizeof=16

// Memory allocation flags
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_VRAM: u32       = 1 << 0;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32        = 1 << 1;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32   = 1 << 31;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC: u32     = 1 << 29;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE: u32 = 1 << 28;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM: u32 = 1 << 27;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_COHERENT: u32   = 1 << 26;
pub(crate) const KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED: u32   = 1 << 25;

// Queue types
pub(crate) const KFD_IOC_QUEUE_TYPE_COMPUTE: u32 = 0x0;     // PM4 compute queue
pub(crate) const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;  // AQL compute queue

// AQL packet types
pub(crate) const HSA_PACKET_TYPE_VENDOR_SPECIFIC: u16 = 0x0;
pub(crate) const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 0x2;

// Fence scopes
#[allow(dead_code)]
pub(crate) const HSA_FENCE_SCOPE_AGENT: u16 = 1;   // GPU-internal only (no PCIe sync)
pub(crate) const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;  // Full system coherency (PCIe writeback)

// PM4-in-AQL constants
pub(crate) const PACKET3_INDIRECT_BUFFER: u32 = 0x3F;
pub(crate) const INDIRECT_BUFFER_VALID: u32 = 1 << 23;
pub(crate) const PM4_IB_SIZE: usize = 256 * 1024; // 256KB for PM4 indirect buffers

// PM4 opcodes for compute dispatch
pub(crate) const PM4_SET_SH_REG: u32         = 0x76;
pub(crate) const PM4_DISPATCH_DIRECT: u32    = 0x15;
pub(crate) const PM4_RELEASE_MEM: u32        = 0x49;
pub(crate) const PM4_ACQUIRE_MEM: u32        = 0x58;
pub(crate) const PM4_EVENT_WRITE: u32        = 0x46;

// GFX11 Compute SH register offsets
pub(crate) const SH_REG_BASE: u32              = 0x2C00;
pub(crate) const REG_COMPUTE_PGM_LO: u32       = 0x2C0C;
pub(crate) const REG_COMPUTE_PGM_RSRC1: u32    = 0x2C44;
pub(crate) const REG_COMPUTE_PGM_RSRC3: u32    = 0x2C94;
pub(crate) const REG_COMPUTE_USER_DATA_0: u32  = 0x2C4C;
pub(crate) const REG_COMPUTE_NUM_THREAD_X: u32 = 0x2C78;
pub(crate) const REG_COMPUTE_RESOURCE_LIMITS: u32 = 0x2C14;
pub(crate) const REG_COMPUTE_START_X: u32      = 0x2C98;
pub(crate) const REG_COMPUTE_TMPRING_SIZE: u32 = 0x2C18;
pub(crate) const REG_COMPUTE_RESTART_X: u32    = 0x2C88;

// GFX11 event types
pub(crate) const CS_PARTIAL_FLUSH: u32     = 0x07;
pub(crate) const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;
pub(crate) const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 0x14;

// mmap constants
pub(crate) const PROT_READ: i32 = 1;
pub(crate) const PROT_WRITE: i32 = 2;
pub(crate) const MAP_SHARED: i32 = 1;
#[allow(dead_code)]
pub(crate) const MAP_PRIVATE: i32 = 2;
pub(crate) const MAP_ANONYMOUS: i32 = 0x20;
pub(crate) const MAP_FIXED: i32 = 0x10;
pub(crate) const MAP_NORESERVE: i32 = 0x4000;
pub(crate) const MAP_FAILED: *mut u8 = usize::MAX as *mut u8;

// =============================================================================
// KFD IOCTL structs (must match kernel's kfd_ioctl.h exactly)
// =============================================================================

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdGetVersionArgs {
    pub(crate) major_version: u32,
    pub(crate) minor_version: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdAcquireVmArgs {
    pub(crate) drm_fd: u32,
    pub(crate) gpu_id: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdAllocMemoryArgs {
    pub(crate) va_addr: u64,
    pub(crate) size: u64,
    pub(crate) handle: u64,
    pub(crate) mmap_offset: u64,
    pub(crate) gpu_id: u32,
    pub(crate) flags: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdFreeMemoryArgs {
    pub(crate) handle: u64,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdMapMemoryArgs {
    pub(crate) handle: u64,
    pub(crate) device_ids_array_ptr: u64,
    pub(crate) n_devices: u32,
    pub(crate) n_success: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdUnmapMemoryArgs {
    pub(crate) handle: u64,
    pub(crate) device_ids_array_ptr: u64,
    pub(crate) n_devices: u32,
    pub(crate) n_success: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdCreateQueueArgs {
    pub(crate) ring_base_address: u64,
    pub(crate) write_pointer_address: u64,
    pub(crate) read_pointer_address: u64,
    pub(crate) doorbell_offset: u64,
    pub(crate) ring_size: u32,
    pub(crate) gpu_id: u32,
    pub(crate) queue_type: u32,
    pub(crate) queue_percentage: u32,
    pub(crate) queue_priority: u32,
    pub(crate) queue_id: u32,
    pub(crate) eop_buffer_address: u64,
    pub(crate) eop_buffer_size: u64,
    pub(crate) ctx_save_restore_address: u64,
    pub(crate) ctx_save_restore_size: u32,
    pub(crate) ctl_stack_size: u32,
    pub(crate) sdma_engine_id: u32,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdDestroyQueueArgs {
    pub(crate) queue_id: u32,
    pub(crate) pad: u32,
}

#[repr(C)]
#[derive(Default, Debug)]
pub(crate) struct KfdRuntimeEnableArgs {
    pub(crate) r_debug: u64,
    pub(crate) mode_mask: u32,
    pub(crate) capabilities_mask: u32,
}

// =============================================================================
// AQL Dispatch Packet (64 bytes, hardware format)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AqlDispatchPacket {
    pub header: u16,               // 0x00: type + barrier + fences (write LAST)
    pub setup: u16,                // 0x02: dimensions
    pub workgroup_size_x: u16,     // 0x04
    pub workgroup_size_y: u16,     // 0x06
    pub workgroup_size_z: u16,     // 0x08
    pub reserved0: u16,            // 0x0A: must be 0
    pub grid_size_x: u32,          // 0x0C
    pub grid_size_y: u32,          // 0x10
    pub grid_size_z: u32,          // 0x14
    pub private_segment_size: u32, // 0x18
    pub group_segment_size: u32,   // 0x1C: LDS size
    pub kernel_object: u64,        // 0x20: VA of kernel descriptor (NOT .text!)
    pub kernarg_address: u64,      // 0x28: VA of kernel arguments
    pub reserved2: u64,            // 0x30: must be 0
    pub completion_signal: u64,    // 0x38: VA of u64 for completion (0 = no signal)
}

const _: () = assert!(std::mem::size_of::<AqlDispatchPacket>() == 64);

// =============================================================================
// libc FFI
// =============================================================================

extern "C" {
    pub(crate) fn open(pathname: *const u8, flags: i32) -> i32;
    pub(crate) fn close(fd: i32) -> i32;
    pub(crate) fn mmap(addr: *mut u8, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    pub(crate) fn munmap(addr: *mut u8, length: usize) -> i32;
    // Use syscall to avoid variadic ioctl() issues
    fn syscall(number: i64, ...) -> i64;
}

pub(crate) const SYS_IOCTL: i64 = 16; // x86-64: __NR_ioctl = 16

pub(crate) fn ioctl_safe(fd: RawFd, request: u64, arg: *mut u8) -> Result<(), String> {
    let ret = unsafe { syscall(SYS_IOCTL, fd as i64, request as i64, arg as i64) };
    if ret < 0 {
        let errno = std::io::Error::last_os_error();
        Err(format!("ioctl 0x{:X} failed: {} (ret={})", request, errno, ret))
    } else {
        Ok(())
    }
}

/// KFD debug trap ioctl (GET_QUEUE_SNAPSHOT / ENABLE — T0_DBG_TRAP 调试钩子)
pub(crate) const AMDKFD_IOC_DBG_TRAP: u64 = 0xc0204b26;
pub(crate) const KFD_IOC_DBG_TRAP_ENABLE: u32 = 0;
pub(crate) const KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT: u32 = 13;
pub(crate) const KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION: u64 = 256;

/// 队列快照：读所有队列的 exception_status（卡时确认 MEMVIOL 例外）。
/// 仅在 T0_DBG_TRAP=1 时由 wait_read_ptr 超时路径调用。
#[cfg(feature = "rocm")]
pub(crate) fn snapshot_queue_exceptions(device: &crate::kfd::device::KfdDevice) {
    use crate::kfd::device::KfdDevice;
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct QSnap { exception_mask: u64, buf: u64, num: u32, entry_size: u32 }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Args { pid: u32, op: u32, data: QSnap }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Entry {
        exception_status: u64, ring_base: u64, wptr: u64, rptr: u64, cwsr: u64,
        queue_id: u32, gpu_id: u32, ring_size: u32, qtype: u32, cwsr_size: u32, reserved: u32,
    }
    let mut entries = [Entry::default(); 8];
    let mut args = Args {
        pid: std::process::id(), op: KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT,
        data: QSnap { exception_mask: 0, buf: entries.as_mut_ptr() as u64, num: 8, entry_size: 64 },
    };
    let _ = ioctl_safe(device.kfd_fd, AMDKFD_IOC_DBG_TRAP, &mut args as *mut _ as *mut u8);
    let mut any = false;
    for e in &entries {
        if e.exception_status != 0 {
            any = true;
            eprintln!("[DBG] queue_id={} gpu_id={} exception_status=0x{:016X} (MEMVIOL={})",
                e.queue_id, e.gpu_id, e.exception_status,
                (e.exception_status >> 5) & 1); // EC_QUEUE_WAVE_MEMORY_VIOLATION = 5
        }
    }
    if !any { eprintln!("[DBG] 快照无 exception（{} 队列）", entries.iter().filter(|e| e.ring_base != 0).count()); }
}
