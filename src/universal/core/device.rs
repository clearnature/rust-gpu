use super::arch::{Arch, DType, Vendor};
use std::time::Duration;

// ═══════════════════════════════════════════════════════
// GPU 设备信息
// ═══════════════════════════════════════════════════════

/// GPU 设备信息 (枚举时填充)
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub vendor: Vendor,
    pub arch: Arch,
    pub vram_size: u64,
    pub compute_units: u32,
    pub max_vgprs: u32,
    pub max_sgprs: u32,
    pub lds_size_per_cu: u32,
    pub wave_size: u32,
    pub clock_mhz: u32,
    pub memory_bandwidth_gbps: f64,
    pub compute_tflops: f64,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub supports_fp4: bool,
    pub supports_wmma: bool,
    pub supports_tensor_core: bool,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            vendor: Vendor::Unknown,
            arch: Arch::Unknown,
            vram_size: 0,
            compute_units: 0,
            max_vgprs: 256,
            max_sgprs: 106,
            lds_size_per_cu: 65536,
            wave_size: 32,
            clock_mhz: 0,
            memory_bandwidth_gbps: 0.0,
            compute_tflops: 0.0,
            supports_fp16: false,
            supports_bf16: false,
            supports_fp8: false,
            supports_fp4: false,
            supports_wmma: false,
            supports_tensor_core: false,
        }
    }
}

// ═══════════════════════════════════════════════════════
// 内存
// ═══════════════════════════════════════════════════════

/// 内存类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemType {
    Vram,
    Host,
    Unified,
    Scratch,
}

/// GPU 内存句柄
pub struct GpuMemory {
    pub device_addr: u64,
    pub host_ptr: Option<u64>,
    pub size: usize,
    pub mem_type: MemType,
    pub handle: u64,
}

// ═══════════════════════════════════════════════════════
// Grid / Block
// ═══════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct Grid(pub u32, pub u32, pub u32);

#[derive(Clone, Copy, Debug)]
pub struct Block(pub u16, pub u16, pub u16);

// ═══════════════════════════════════════════════════════
// Kernel 句柄
// ═══════════════════════════════════════════════════════

pub trait Kernel: Send + Sync + std::any::Any {
    fn name(&self) -> &str;
    fn vgpr_count(&self) -> u32;
    fn sgpr_count(&self) -> u32;
    fn lds_size(&self) -> u32;
    fn kernarg_size(&self) -> usize;
    fn gpu_addr(&self) -> u64;
    fn as_any(&self) -> &dyn std::any::Any;
}

// ═══════════════════════════════════════════════════════
// 信号量
// ═══════════════════════════════════════════════════════

pub trait Signal: Send + Sync {
    fn value(&self) -> u64;
    fn set(&self, value: u64);
    fn wait(&self, expected: u64, timeout: Duration) -> Result<(), String>;
    fn gpu_addr(&self) -> u64;
}

// ═══════════════════════════════════════════════════════
// 计算队列
// ═══════════════════════════════════════════════════════

/// 队列配置
#[derive(Clone, Debug)]
pub struct QueueConfig {
    pub priority: QueuePriority,
    pub queue_type: QueueType,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            priority: QueuePriority::Normal,
            queue_type: QueueType::Compute,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum QueuePriority { Low, Normal, High }

#[derive(Clone, Copy, Debug)]
pub enum QueueType { Compute, Copy }

/// 计算队列 — kernel dispatch 的核心接口
pub trait ComputeQueue: Send + Sync {
    fn submit(
        &mut self,
        kernel: &dyn Kernel,
        grid: Grid,
        block: Block,
        kernargs: &[u8],
        signal: Option<&dyn Signal>,
    ) -> Result<(), String>;

    fn barrier(&mut self, signals: &[&dyn Signal]) -> Result<(), String>;
    fn flush(&mut self) -> Result<(), String>;
    fn wait_idle(&mut self) -> Result<(), String>;
    fn pending_count(&self) -> usize;
}

/// DMA 拷贝队列
pub trait CopyQueue: Send + Sync {
    fn copy(
        &mut self,
        dst: &GpuMemory,
        src: &GpuMemory,
        size: usize,
        signal: Option<&dyn Signal>,
    ) -> Result<(), String>;
    fn flush(&mut self) -> Result<(), String>;
}

// ═══════════════════════════════════════════════════════
// GPU 设备 — 所有 GPU 操作的入口
// ═══════════════════════════════════════════════════════

pub trait GpuDevice: Send + Sync {
    fn info(&self) -> &DeviceInfo;

    // ── 内存管理 ──
    fn alloc(&self, size: usize, mem_type: MemType) -> Result<GpuMemory, String>;
    fn free(&self, mem: GpuMemory) -> Result<(), String>;
    fn map_to_cpu(&self, mem: &GpuMemory) -> Result<*mut u8, String>;
    fn unmap_from_cpu(&self, mem: &GpuMemory) -> Result<(), String>;

    // ── 数据传输 ──
    fn copy_from_host(&self, dst: &GpuMemory, src: &[u8]) -> Result<(), String>;
    fn copy_to_host(&self, dst: &mut [u8], src: &GpuMemory) -> Result<(), String>;
    fn copy_device(&self, dst: &GpuMemory, src: &GpuMemory, size: usize) -> Result<(), String>;

    // ── 队列 ──
    fn create_compute_queue(&self, config: QueueConfig) -> Result<Box<dyn ComputeQueue>, String>;
    fn create_copy_queue(&self) -> Result<Box<dyn CopyQueue>, String>;

    // ── 同步 ──
    fn create_signal(&self, initial_value: u64) -> Result<Box<dyn Signal>, String>;
    fn wait_idle(&self) -> Result<(), String>;

    // ── Kernel 加载 ──
    fn load_kernel(&self, elf_bytes: &[u8], name: &str) -> Result<Box<dyn Kernel>, String>;
    fn load_kernel_with_wg(&self, elf_bytes: &[u8], name: &str, wg_size: u32) -> Result<Box<dyn Kernel>, String> {
        // 默认实现: 忽略 wg_size
        self.load_kernel(elf_bytes, name)
    }
}

// ═══════════════════════════════════════════════════════
// 驱动工厂 — 运行时发现并加载可用的 GPU 驱动
// ═══════════════════════════════════════════════════════

pub trait DriverFactory: Send + Sync {
    fn enumerate(&self) -> Vec<DeviceInfo>;
    fn open(&self, device_id: u32) -> Result<Box<dyn GpuDevice>, String>;
    fn is_available(&self) -> bool;
    fn name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════
// 设备管理器 — 自动发现所有 GPU
// ═══════════════════════════════════════════════════════

pub struct DeviceManager {
    factories: Vec<Box<dyn DriverFactory>>,
    devices: Vec<DeviceInfo>,
}

impl DeviceManager {
    pub fn discover() -> Self {
        let mut factories: Vec<Box<dyn DriverFactory>> = Vec::new();

        // 按优先级尝试每个驱动
        #[cfg(feature = "rocm")]
        {
            use crate::universal::driver::amd::AmdDriver;
            if AmdDriver::is_available_fn() {
                factories.push(Box::new(AmdDriver::new()));
            }
        }

        // NVIDIA 驱动 (总是尝试, 不需要特殊 feature)
        {
            use crate::universal::driver::nvidia::NvDriver;
            if NvDriver::is_available_fn() {
                factories.push(Box::new(NvDriver::new()));
            }
        }

        // 华为昇腾驱动
        {
            use crate::universal::driver::ascend::AscendDriver;
            if AscendDriver::is_available_fn() {
                factories.push(Box::new(AscendDriver::new()));
            }
        }

        let devices = factories.iter()
            .flat_map(|f| f.enumerate())
            .collect();

        Self { factories, devices }
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn open(&self, id: u32) -> Result<Box<dyn GpuDevice>, String> {
        for f in &self.factories {
            for dev_info in f.enumerate() {
                if dev_info.id == id {
                    return f.open(id);
                }
            }
        }
        Err(format!("Device {} not found", id))
    }

    pub fn devices_by_vendor(&self, vendor: Vendor) -> Vec<&DeviceInfo> {
        self.devices.iter().filter(|d| d.vendor == vendor).collect()
    }
}
