//! RDNA3 Bare-Metal KFD Runtime
//!
//! Direct GPU control via /dev/kfd ioctl, bypassing HIP/ROCm entirely.
//! Implements: VRAM allocation, AQL queue dispatch, doorbell ring, completion polling.
//!
//! Target: AMD RX 7900 XTX (GFX1100, RDNA3), KFD v1.14, Linux 6.17
//!
//! Architecture:
//!   /dev/kfd  → KFD ioctl (memory, queues, events)
//!   /dev/dri/renderD128 → DRM fd for acquire_vm
//!   AQL ring buffer → 64-byte dispatch packets → doorbell → GPU execution

pub(crate) mod ioctl;
mod buffer;
mod kernel;
mod aql;
mod device;
mod pool;

// Re-export public types
pub use ioctl::AqlDispatchPacket;
// HSA/uapi 契约常量（tests/aql_contract.rs 锁定）
pub use ioctl::{
    AMDKFD_IOC_ACQUIRE_VM, AMDKFD_IOC_ALLOC_MEMORY, AMDKFD_IOC_CREATE_QUEUE,
    AMDKFD_IOC_GET_VERSION, HSA_FENCE_SCOPE_SYSTEM, HSA_PACKET_TYPE_KERNEL_DISPATCH,
    PM4_ACQUIRE_MEM, PM4_DISPATCH_DIRECT, PM4_EVENT_WRITE, PM4_SET_SH_REG,
};
pub use buffer::GpuBuffer;
pub use kernel::{GpuKernel, KernelLoadConfig};
pub use aql::AqlQueue;
pub mod pm4;
pub use pm4::{Pm4CmdBuilder, Pm4Queue};
pub use device::KfdDevice;
pub use pool::{DispatchPool, GpuMemset};

// SIGPIPE defense — prevent pipe-broken signals from killing the process
pub(crate) fn ignore_sigpipe() {
    extern "C" { fn signal(sig: i32, handler: usize) -> usize; }
    unsafe { signal(13, 1); }
}
