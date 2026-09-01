use crate::universal::core::{
    Arch, Block, ComputeQueue, CopyQueue, DeviceInfo, DriverFactory, GpuDevice,
    GpuMemory, Grid, Kernel, MemType, QueueConfig, Signal, Vendor,
};
use std::sync::Arc;
use std::time::Duration;

// ═══════════════════════════════════════════════════════
// AmdDriver — 桥接到 t0-gpu 现有 KfdDevice
// ═══════════════════════════════════════════════════════

pub struct AmdDriver {
    #[cfg(feature = "rocm")]
    inner: Option<Arc<crate::kfd::KfdDevice>>,
}

impl AmdDriver {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "rocm")]
            inner: crate::kfd::KfdDevice::open().ok(),
        }
    }

    pub fn is_available_fn() -> bool {
        std::path::Path::new("/dev/kfd").exists()
    }
}

impl DriverFactory for AmdDriver {
    fn enumerate(&self) -> Vec<DeviceInfo> {
        #[cfg(feature = "rocm")]
        {
            let mut devices = Vec::new();
            let topology_path = "/sys/class/kfd/kfd/topology/nodes";

            if let Ok(entries) = std::fs::read_dir(topology_path) {
                for entry in entries.flatten() {
                    let props_path = entry.path().join("properties");
                    if let Ok(props) = std::fs::read_to_string(&props_path) {
                        let mut simd_count = 0u32;
                        let mut wave_size = 32u32;
                        let mut gfx_target_version = 0u32;
                        let mut lds_size = 0u32;

                        for line in props.lines() {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() < 2 { continue; }
                            match parts[0] {
                                "simd_count" => { simd_count = parts[1].parse().unwrap_or(0); }
                                "wave_front_size" => { wave_size = parts[1].parse().unwrap_or(32); }
                                "gfx_target_version" => { gfx_target_version = parts[1].parse().unwrap_or(0); }
                                "lds_size_in_kb" => { lds_size = parts[1].parse::<u32>().unwrap_or(0) * 1024; }
                                _ => {}
                            }
                        }

                        // 只处理 GPU 节点 (simd_count > 0)
                        if simd_count > 0 {
                            let node_id: u32 = entry.file_name().to_string_lossy().parse().unwrap_or(0);
                            let arch = detect_arch(gfx_target_version);
                            devices.push(DeviceInfo {
                                id: node_id,
                                name: format!("AMD {:?} ({} CUs)", arch, simd_count),
                                vendor: Vendor::AMD,
                                arch,
                                vram_size: 0,
                                compute_units: simd_count,
                                max_vgprs: 256,
                                max_sgprs: 106,
                                lds_size_per_cu: lds_size.max(65536),
                                wave_size,
                                clock_mhz: 2500,
                                memory_bandwidth_gbps: 960.0,
                                compute_tflops: 61.0,
                                supports_fp16: true,
                                supports_bf16: true,
                                supports_fp8: false,
                                supports_fp4: false,
                                supports_wmma: true,
                                supports_tensor_core: false,
                            });
                        }
                    }
                }
            }
            devices
        }
        #[cfg(not(feature = "rocm"))]
        { Vec::new() }
    }

    fn open(&self, device_id: u32) -> Result<Box<dyn GpuDevice>, String> {
        #[cfg(feature = "rocm")]
        {
            let dev = self.inner.as_ref()
                .ok_or("KFD device not available")?;
            // 枚举时用 node_id, 但 KfdDevice 的 gpu_id 不同
            // 只要设备存在就打开 (单 GPU 系统)
            Ok(Box::new(AmdDeviceBridge {
                inner: dev.clone(),
                info: self.enumerate().into_iter().next()
                    .ok_or("No AMD GPU found")?,
            }))
        }
        #[cfg(not(feature = "rocm"))]
        { Err("ROCm feature not enabled".into()) }
    }

    fn is_available(&self) -> bool { Self::is_available_fn() }
    fn name(&self) -> &str { "AMD KFD" }
}

// ═══════════════════════════════════════════════════════
// AmdDeviceBridge — 包装 KfdDevice 实现 GpuDevice trait
// ═══════════════════════════════════════════════════════

#[cfg(feature = "rocm")]
struct AmdDeviceBridge {
    inner: Arc<crate::kfd::KfdDevice>,
    info: DeviceInfo,
}

#[cfg(feature = "rocm")]
unsafe impl Send for AmdDeviceBridge {}
#[cfg(feature = "rocm")]
unsafe impl Sync for AmdDeviceBridge {}

#[cfg(feature = "rocm")]
impl GpuDevice for AmdDeviceBridge {
    fn info(&self) -> &DeviceInfo { &self.info }

    fn alloc(&self, size: usize, mem_type: MemType) -> Result<GpuMemory, String> {
        let buf = match mem_type {
            MemType::Vram | MemType::Scratch => self.inner.alloc_vram(size)?,
            MemType::Host | MemType::Unified => self.inner.alloc_uncached(size)?,
        };
        let host_ptr = buf.host_ptr as u64;
        let device_addr = buf.va_addr;
        let buf_size = buf.size;
        // 保持 GpuBuffer 存活 (泄漏, 由 free() 管理)
        Box::leak(Box::new(buf));
        Ok(GpuMemory {
            device_addr,
            host_ptr: Some(host_ptr),
            size: buf_size,
            mem_type,
            handle: 0,
        })
    }

    fn free(&self, _mem: GpuMemory) -> Result<(), String> { Ok(()) }

    fn map_to_cpu(&self, mem: &GpuMemory) -> Result<*mut u8, String> {
        Ok(mem.host_ptr.unwrap_or(0) as *mut u8)
    }

    fn unmap_from_cpu(&self, _mem: &GpuMemory) -> Result<(), String> { Ok(()) }

    fn copy_from_host(&self, dst: &GpuMemory, src: &[u8]) -> Result<(), String> {
        let ptr = dst.host_ptr.ok_or("Buffer not CPU-mapped")? as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: &GpuMemory) -> Result<(), String> {
        let ptr = src.host_ptr.ok_or("Buffer not CPU-mapped")? as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()); }
        Ok(())
    }

    fn copy_device(&self, _dst: &GpuMemory, _src: &GpuMemory, _size: usize) -> Result<(), String> {
        Err("Device-to-device copy not yet implemented".into())
    }

    fn create_compute_queue(&self, _config: QueueConfig) -> Result<Box<dyn ComputeQueue>, String> {
        let queue = self.inner.create_queue()?;
        Ok(Box::new(AqlQueueBridge {
            inner: queue,
            device: self.inner.clone(),
        }))
    }

    fn create_copy_queue(&self) -> Result<Box<dyn CopyQueue>, String> {
        Err("Copy queue not yet implemented".into())
    }

    fn create_signal(&self, _initial_value: u64) -> Result<Box<dyn Signal>, String> {
        let buf = self.inner.alloc_signal()?;
        let host_ptr = buf.host_ptr as *mut u8;
        let buf_addr = buf.va_addr;
        // 注意: 需要保持 GpuBuffer 存活, 否则 host_ptr 会变悬空
        // 这里用 Box::leak 让 buf 泄漏 (信号量通常长期存在)
        Box::leak(Box::new(buf));
        Ok(Box::new(AmdSignal {
            buf_addr,
            host_ptr,
        }))
    }

    fn wait_idle(&self) -> Result<(), String> { Ok(()) }

    fn load_kernel(&self, elf_bytes: &[u8], name: &str) -> Result<Box<dyn Kernel>, String> {
        let config = crate::kfd::KernelLoadConfig {
            lds_size: 0,
            workgroup_size: [256, 1, 1],
        };
        let kernel = crate::kfd::GpuKernel::load(
            &self.inner, elf_bytes, &config
        )?;
        Ok(Box::new(AmdKernelBridge { inner: kernel }))
    }

    fn load_kernel_with_wg(&self, elf_bytes: &[u8], _name: &str, wg_size: u32) -> Result<Box<dyn Kernel>, String> {
        let config = crate::kfd::KernelLoadConfig {
            lds_size: 0,
            workgroup_size: [wg_size, 1, 1],
        };
        let kernel = crate::kfd::GpuKernel::load(
            &self.inner, elf_bytes, &config
        )?;
        Ok(Box::new(AmdKernelBridge { inner: kernel }))
    }
}

// ═══════════════════════════════════════════════════════
// AmdKernelBridge — 包装 GpuKernel 实现 Kernel trait
// ═══════════════════════════════════════════════════════

#[cfg(feature = "rocm")]
struct AmdKernelBridge {
    inner: crate::kfd::GpuKernel,
}

#[cfg(feature = "rocm")]
unsafe impl Send for AmdKernelBridge {}
#[cfg(feature = "rocm")]
unsafe impl Sync for AmdKernelBridge {}

#[cfg(feature = "rocm")]
impl Kernel for AmdKernelBridge {
    fn name(&self) -> &str { "t0_kernel" }
    fn vgpr_count(&self) -> u32 { 0 }
    fn sgpr_count(&self) -> u32 { 0 }
    fn lds_size(&self) -> u32 { self.inner.lds_size }
    fn kernarg_size(&self) -> usize { self.inner.kernarg_size as usize }
    fn gpu_addr(&self) -> u64 { self.inner.descriptor_va }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// ═══════════════════════════════════════════════════════
// AmdSignal — 包装 GpuBuffer 实现 Signal trait
// ═══════════════════════════════════════════════════════

#[cfg(feature = "rocm")]
struct AmdSignal {
    buf_addr: u64,
    host_ptr: *mut u8,
}

#[cfg(feature = "rocm")]
unsafe impl Send for AmdSignal {}
#[cfg(feature = "rocm")]
unsafe impl Sync for AmdSignal {}

#[cfg(feature = "rocm")]
impl Signal for AmdSignal {
    fn value(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.host_ptr.add(8) as *const u64) }
    }

    fn set(&self, value: u64) {
        unsafe {
            std::ptr::write_volatile(self.host_ptr.add(8) as *mut u64, value);
        }
    }

    fn wait(&self, expected: u64, timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        loop {
            if self.value() == expected { return Ok(()); }
            if start.elapsed() > timeout {
                return Err(format!("Signal wait timeout ({}ms)", timeout.as_millis()));
            }
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    fn gpu_addr(&self) -> u64 { self.buf_addr }
}

// ═══════════════════════════════════════════════════════
// AqlQueueBridge — 包装 AqlQueue 实现 ComputeQueue trait
// ═══════════════════════════════════════════════════════

#[cfg(feature = "rocm")]
struct AqlQueueBridge {
    inner: crate::kfd::AqlQueue,
    device: Arc<crate::kfd::KfdDevice>,
}

#[cfg(feature = "rocm")]
unsafe impl Send for AqlQueueBridge {}
#[cfg(feature = "rocm")]
unsafe impl Sync for AqlQueueBridge {}

#[cfg(feature = "rocm")]
impl ComputeQueue for AqlQueueBridge {
    fn submit(
        &mut self,
        kernel: &dyn Kernel,
        grid: Grid,
        block: Block,
        kernargs: &[u8],
        signal: Option<&dyn Signal>,
    ) -> Result<(), String> {
        // 分配 kernarg buffer 并写入参数
        let kernarg_buf = self.device.alloc_uncached(kernargs.len().max(256))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                kernargs.as_ptr(),
                kernarg_buf.host_ptr,
                kernargs.len(),
            );
        }

        // 从 AmdKernelBridge 获取实际的 GpuKernel
        let kernel_ref: &crate::kfd::GpuKernel = if let Some(bridge) = kernel.as_any().downcast_ref::<AmdKernelBridge>() {
            &bridge.inner
        } else {
            return Err("Kernel is not an AmdKernelBridge".into());
        };

        // 提交到 AQL 队列
        self.inner.dispatch(kernel_ref, [grid.0, grid.1, grid.2], &kernarg_buf)?;

        // 等待 GPU 完成 (防止 kernarg_buf 被 drop 后 GPU 还在读)
        self.inner.wait_idle()?;

        Ok(())
    }

    fn barrier(&mut self, _signals: &[&dyn Signal]) -> Result<(), String> { Ok(()) }
    fn flush(&mut self) -> Result<(), String> { Ok(()) }

    fn wait_idle(&mut self) -> Result<(), String> {
        self.inner.wait_idle()
    }

    fn pending_count(&self) -> usize { 0 }
}

// ═══════════════════════════════════════════════════════
// 辅助函数
// ═══════════════════════════════════════════════════════

#[cfg(feature = "rocm")]
fn detect_arch(gfx_target_version: u32) -> Arch {
    match gfx_target_version {
        110000..=110099 => Arch::Gfx1100,
        120000..=120099 => Arch::Gfx1200,
        120100..=120199 => Arch::Gfx1201,
        94200..=94299 => Arch::Gfx942,
        95000..=95099 => Arch::Gfx950,
        _ => Arch::Gfx1200,
    }
}
