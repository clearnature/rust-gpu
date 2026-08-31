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
mod pm4;
mod device;
mod pool;

// Re-export public types
pub use ioctl::AqlDispatchPacket;
pub use buffer::GpuBuffer;
pub use kernel::{GpuKernel, KernelLoadConfig};
pub use aql::AqlQueue;
pub use pm4::{Pm4CmdBuilder, Pm4Queue};
pub use device::KfdDevice;
pub use pool::{DispatchPool, GpuMemset};

// SIGPIPE defense — prevent pipe-broken signals from killing the process
pub(crate) fn ignore_sigpipe() {
    extern "C" { fn signal(sig: i32, handler: usize) -> usize; }
    unsafe { signal(13, 1); }
}
