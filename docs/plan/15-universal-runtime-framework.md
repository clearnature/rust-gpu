# 通用 GPU 运行时 & 统一管理框架

> 完整架构设计, 可直接用于 Rust 实现
> 综合 t0-gpu + tinygrad + sass-assembler + ROCm 的所有分析成果

## 一、总体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        应用层 (Application Layer)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  ignis   │ │ PyTorch  │ │  vLLM    │ │ 自定义   │ │ CLI tools    │ │
│  │  NN框架  │ │ binding  │ │ serving  │ │ kernel   │ │ (ht-as 等)   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
└───────┼────────────┼────────────┼────────────┼───────────────┼─────────┘
        │            │            │            │               │
┌───────▼────────────▼────────────▼────────────▼───────────────▼─────────┐
│                     统一 API 层 (Unified API)                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Device::open() / Device::alloc() / Device::launch() / ...     │   │
│  │  Tensor / Module / Optimizer / LossScaler / DataLoader         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    调度框架 (Scheduling Framework)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Graph    │ │ Tile     │ │ Template │ │ Instr    │ │ Hardware   │  │
│  │ Scheduler│ │ Optimizr │ │ Selector │ │ Schedule │ │ Wave Sched │  │
│  │ (Level5) │ │ (Level4) │ │ (Level3) │ │ (Level2) │ │ (Level1)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    编译框架 (Compiler Framework)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ DSL/     │ │ SSA IR   │ │ Opt      │ │ RegAlloc │ │ ISA        │  │
│  │ UOp IR   │ │ + TileIR │ │ Passes   │ │ (SSA)    │ │ Encoder    │  │
│  │ Frontend │ │          │ │ (15 pass)│ │          │ │ (per-vendor│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    数学库 (Math Libraries)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ BLAS     │ │ Sparse   │ │ RNG      │ │ Scan/    │ │ FFT        │  │
│  │ GEMM/GEMV│ │ SpMV/MM  │ │ Philox   │ │ Sort     │ │ Radix-2    │  │
│  │ Batched  │ │ COO/CSR  │ │ Distrs   │ │ Reduce   │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    运行时核心 (Runtime Core)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Device HAL        Memory Manager       Queue Manager           │  │
│  │  ┌────────────┐    ┌────────────┐       ┌────────────┐          │  │
│  │  │ trait       │    │ trait       │       │ trait       │         │  │
│  │  │ GpuDriver   │    │ MemManager  │       │ QueueMgr    │        │  │
│  │  └──────┬─────┘    └──────┬─────┘       └──────┬─────┘         │  │
│  │         │                 │                     │               │  │
│  │  ┌──────▼─────┐    ┌──────▼─────┐       ┌──────▼─────┐         │  │
│  │  │ AmdDriver  │    │ BumpAlloc  │       │ AqlQueue   │         │  │
│  │  │ NvDriver   │    │ BuddyAlloc │       │ GpfifoQueue│         │  │
│  │  │ AscendDrv  │    │ PoolAlloc  │       │ Pm4Queue   │         │  │
│  │  └────────────┘    └────────────┘       └────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Shared GPU Scheduler                                            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │  │
│  │  │ TimeSlice  │ │ VRAM Quota │ │ CU/SM Part │ │ Priority   │   │  │
│  │  │ Scheduler  │ │ Manager    │ │ itioning   │ │ Queue      │   │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────────┐
│                    厂商驱动层 (Vendor Driver Layer)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │  AMD         │ │  NVIDIA      │ │  华为昇腾     │ │  通用        │ │
│  │  /dev/kfd    │ │  /dev/nvidia │ │  /dev/davinci│ │  plugin      │ │
│  │  AQL packet  │ │  GPFIFO+QMD  │ │  任务队列     │ │  interface   │ │
│  │  GFX ISA     │ │  SASS/PTX    │ │  Ascend ISA  │ │  自定义      │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                     ┌──────▼──────┐
                     │  GPU 硬件    │
                     └─────────────┘
```

## 二、核心 Trait 定义

### 2.1 设备管理

```rust
// ═══════════════════════════════════════════════════════
// 设备发现与管理
// ═══════════════════════════════════════════════════════

/// GPU 设备信息
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,                    // "RX 9060 XT" / "RTX 4090" / "Ascend 910B"
    pub vendor: Vendor,                  // AMD / NVIDIA / Huawei / ...
    pub arch: Arch,                      // GFX1200 / SM89 / AscendC64 / ...
    pub vram_size: u64,                  // 字节
    pub compute_units: u32,              // CU 数 (AMD) / SM 数 (NVIDIA) / AI Core 数
    pub max_waves_per_cu: u32,           // 每 CU 最大 wave/warp 数
    pub max_vgprs: u32,                  // 每线程最大向量寄存器
    pub max_sgprs: u32,                  // 每线程最大标量寄存器
    pub lds_size_per_cu: u32,            // 每 CU LDS/shared memory
    pub wave_size: u32,                  // 32 (AMD Wave32) / 32 (NVIDIA Warp) / 64 (Wave64)
    pub clock_mhz: u32,                  // GPU 时钟频率
    pub memory_bandwidth_gbps: f64,      // 显存带宽
    pub compute_tflops: f64,             // 峰值算力
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub supports_fp4: bool,
    pub supports_wmma: bool,             // 矩阵加速指令
    pub supports_tensor_core: bool,      // NVIDIA Tensor Core
    pub pcie_gen: u32,                   // PCIe 代数
    pub pcie_width: u32,                 // PCIe 通道数
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor { AMD, NVIDIA, Huawei, MooreThreads, Biren, Enflame, Intel, Unknown }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    // AMD
    Gfx1100, Gfx1200, Gfx942, Gfx950,
    // NVIDIA
    Sm80, Sm86, Sm89, Sm90, Sm100,
    // 华为
    AscendC64, AscendC68,
    // 通用
    Unknown,
}

/// 驱动工厂 — 运行时发现并加载可用的 GPU 驱动
pub trait DriverFactory: Send + Sync {
    /// 枚举所有可用的 GPU 设备
    fn enumerate(&self) -> Vec<DeviceInfo>;

    /// 打开指定设备
    fn open(&self, device_id: u32) -> Result<Box<dyn GpuDevice>>;

    /// 检查驱动是否可用 (驱动文件存在、权限足够)
    fn is_available(&self) -> bool;

    /// 驱动名称
    fn name(&self) -> &str;
}

/// 运行时设备管理器
pub struct DeviceManager {
    factories: Vec<Box<dyn DriverFactory>>,
    devices: Vec<DeviceInfo>,
}

impl DeviceManager {
    /// 自动发现所有 GPU
    pub fn discover() -> Self {
        let mut factories: Vec<Box<dyn DriverFactory>> = Vec::new();

        // 按优先级尝试每个驱动
        if AmdDriver::is_available()   { factories.push(Box::new(AmdDriver::new())); }
        if NvidiaDriver::is_available() { factories.push(Box::new(NvidiaDriver::new())); }
        if AscendDriver::is_available() { factories.push(Box::new(AscendDriver::new())); }
        // 插件式扩展...

        let devices = factories.iter()
            .flat_map(|f| f.enumerate())
            .collect();

        Self { factories, devices }
    }

    /// 获取所有设备信息
    pub fn devices(&self) -> &[DeviceInfo] { &self.devices }

    /// 按 ID 打开设备
    pub fn open(&self, id: u32) -> Result<Box<dyn GpuDevice>> {
        for f in &self.factories {
            if let Ok(dev) = f.open(id) { return Ok(dev); }
        }
        Err(format!("Device {} not found", id))
    }

    /// 按 vendor 过滤设备
    pub fn devices_by_vendor(&self, vendor: Vendor) -> Vec<&DeviceInfo> {
        self.devices.iter().filter(|d| d.vendor == vendor).collect()
    }
}
```

### 2.2 设备操作

```rust
// ═══════════════════════════════════════════════════════
// GPU 设备核心接口
// ═══════════════════════════════════════════════════════

/// GPU 设备 — 所有 GPU 操作的入口
pub trait GpuDevice: Send + Sync {
    /// 设备信息
    fn info(&self) -> &DeviceInfo;

    // ── 内存管理 ──

    /// 分配 GPU 内存
    fn alloc(&self, size: usize, mem_type: MemType) -> Result<GpuMemory>;

    /// 释放 GPU 内存
    fn free(&self, mem: GpuMemory) -> Result<()>;

    /// CPU 映射 (返回可读写的 CPU 指针)
    fn map_to_cpu(&self, mem: &GpuMemory) -> Result<*mut u8>;

    /// 取消 CPU 映射
    fn unmap_from_cpu(&self, mem: &GpuMemory) -> Result<()>;

    // ── 数据传输 ──

    /// Host → Device
    fn copy_from_host(&self, dst: &GpuMemory, src: &[u8]) -> Result<()>;

    /// Device → Host
    fn copy_to_host(&self, dst: &mut [u8], src: &GpuMemory) -> Result<()>;

    /// Device → Device (同设备)
    fn copy_device(&self, dst: &GpuMemory, src: &GpuMemory, size: usize) -> Result<()>;

    /// Device → Device (跨设备 P2P)
    fn copy_p2p(&self, dst: &GpuMemory, dst_dev: &dyn GpuDevice,
                src: &GpuMemory, size: usize) -> Result<()>;

    // ── 队列管理 ──

    /// 创建计算队列
    fn create_compute_queue(&self, config: QueueConfig) -> Result<Box<dyn ComputeQueue>>;

    /// 创建拷贝队列 (DMA)
    fn create_copy_queue(&self) -> Result<Box<dyn CopyQueue>>;

    // ── 同步 ──

    /// 创建信号量
    fn create_signal(&self, initial_value: u64) -> Result<Box<dyn Signal>>;

    /// 等待设备空闲
    fn wait_idle(&self) -> Result<()>;

    // ── 性能计数器 ──

    /// 读取性能计数器
    fn read_counters(&self) -> Result<GpuCounters>;
}

#[derive(Clone, Copy, Debug)]
pub enum MemType {
    Vram,           // GPU 本地显存 (最快)
    Host,           // CPU 系统内存 (pinned, GPU 可通过 PCIe 访问)
    Unified,        // 统一内存 (如果硬件支持)
    Scratch,        // 每线程私有 (寄存器溢出)
}

pub struct GpuMemory {
    pub device_addr: u64,       // GPU 虚拟地址
    pub host_ptr: Option<u64>,  // CPU 可见地址 (如果已映射)
    pub size: usize,
    pub mem_type: MemType,
    pub handle: u64,            // 驱动特定句柄
}
```

### 2.3 队列与调度

```rust
// ═══════════════════════════════════════════════════════
// 计算队列 — kernel dispatch 的核心接口
// ═══════════════════════════════════════════════════════

/// 计算队列 (AQL / GPFIFO / PM4)
pub trait ComputeQueue: Send + Sync {
    /// 提交 kernel dispatch (异步, 不等待完成)
    fn submit(&mut self, kernel: &dyn Kernel, grid: Grid, block: Block,
              kernargs: &[u8], signal: Option<&dyn Signal>) -> Result<()>;

    /// 提交 barrier (等待前序 dispatch 完成)
    fn barrier(&mut self, signals: &[&dyn Signal]) -> Result<()>;

    /// 刷新队列 (确保所有提交被 GPU 看到)
    fn flush(&mut self) -> Result<()>;

    /// 等待队列空闲
    fn wait_idle(&mut self) -> Result<()>;

    /// 队列状态
    fn pending_count(&self) -> usize;
}

/// DMA 拷贝队列
pub trait CopyQueue: Send + Sync {
    fn copy(&mut self, dst: &GpuMemory, src: &GpuMemory, size: usize,
            signal: Option<&dyn Signal>) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

/// GPU 信号量 (跨设备同步)
pub trait Signal: Send + Sync {
    fn value(&self) -> u64;
    fn set(&self, value: u64);
    fn wait(&self, expected: u64, timeout: Duration) -> Result<()>;
    fn gpu_addr(&self) -> u64;  // GPU 可见地址
}

pub struct Grid(pub u32, pub u32, pub u32);
pub struct Block(pub u16, pub u16, pub u16);

/// Kernel 句柄 (已编译的 GPU 程序)
pub trait Kernel: Send + Sync {
    fn name(&self) -> &str;
    fn vgpr_count(&self) -> u32;
    fn sgpr_count(&self) -> u32;
    fn lds_size(&self) -> u32;
    fn kernarg_size(&self) -> usize;
    fn gpu_addr(&self) -> u64;  // 代码在 GPU 内存中的地址
}
```

### 2.4 内存管理器

```rust
// ═══════════════════════════════════════════════════════
// 统一内存管理
// ═══════════════════════════════════════════════════════

/// 内存管理器 trait
pub trait MemoryManager: Send + Sync {
    /// 分配内存 (带对齐)
    fn alloc(&mut self, size: usize, align: usize, mem_type: MemType) -> Result<GpuMemory>;

    /// 释放内存
    fn free(&mut self, mem: GpuMemory);

    /// 统计
    fn used_bytes(&self) -> u64;
    fn total_bytes(&self) -> u64;
    fn fragmentation_ratio(&self) -> f64;
}

/// Bump 分配器 (线性, 极快, 适合单次 dispatch)
pub struct BumpAllocator {
    base: u64,
    offset: u64,
    size: u64,
}

/// 伙伴分配器 (可释放, 适合通用场景)
pub struct BuddyAllocator {
    free_lists: Vec<Vec<u64>>,  // 按 2^n 大小分桶
    total: u64,
    used: u64,
}

/// 池分配器 (2^n 桶缓存, 来自 t0-gpu buffer_pool)
pub struct PoolAllocator {
    buckets: HashMap<usize, Vec<GpuMemory>>,  // 2^n → 可用 buffer 列表
    device: Arc<dyn GpuDevice>,
    hits: u64,
    misses: u64,
}

impl PoolAllocator {
    /// 最小桶大小 = 4096 (KFD 页大小)
    const MIN_BUCKET: usize = 4096;

    pub fn alloc(&mut self, size: usize) -> Result<GpuMemory> {
        let bucket = size.max(Self::MIN_BUCKET).next_power_of_two();
        if let Some(buf) = self.buckets.get_mut(&bucket).and_then(|v| v.pop()) {
            self.hits += 1;
            return Ok(buf);
        }
        self.misses += 1;
        self.device.alloc(bucket, MemType::Vram)
    }

    pub fn free(&mut self, mem: GpuMemory) {
        let bucket = mem.size.next_power_of_two();
        self.buckets.entry(bucket).or_default().push(mem);
    }
}

/// 跨设备统一内存管理器
pub struct UnifiedMemoryManager {
    devices: Vec<Arc<dyn GpuDevice>>,
    allocators: Vec<Box<dyn MemoryManager>>,
    p2p_enabled: HashMap<(u32, u32), bool>,  // (dev_a, dev_b) → P2P 可用
}

impl UnifiedMemoryManager {
    /// 跨设备内存传输 (自动选择最优路径)
    pub fn transfer(&self, dst_dev: u32, dst: &GpuMemory,
                    src_dev: u32, src: &GpuMemory, size: usize) -> Result<()> {
        if dst_dev == src_dev {
            // 同设备: 直接 DMA
            self.devices[dst_dev as usize].copy_device(dst, src, size)
        } else if self.p2p_enabled.get(&(src_dev, dst_dev)).copied().unwrap_or(false) {
            // P2P: PCIe/NVLink/xGMI 直传
            self.devices[src_dev as usize].copy_p2p(dst, &*self.devices[dst_dev as usize], src, size)
        } else {
            // Bounce: 经 CPU 中转
            let mut host_buf = vec![0u8; size];
            self.devices[src_dev as usize].copy_to_host(&mut host_buf, src)?;
            self.devices[dst_dev as usize].copy_from_host(dst, &host_buf)
        }
    }
}
```

### 2.5 编译器框架

```rust
// ═══════════════════════════════════════════════════════
// 编译器 trait — 从 IR 到机器码
// ═══════════════════════════════════════════════════════

/// 编译器后端 trait
pub trait CompilerBackend: Send + Sync {
    /// 编译 IR → 机器码 ELF
    fn compile(&self, ir: &KernelIr, target: Arch) -> Result<CompiledKernel>;

    /// 编译延迟
    fn compile_time_estimate(&self) -> Duration;

    /// 是否支持目标架构
    fn supports(&self, target: Arch) -> bool;

    /// 后端名称
    fn name(&self) -> &str;
}

/// ISA 编码器 trait (底层, 每厂商实现)
pub trait IsaEncoder: Send + Sync {
    /// 编码单条指令
    fn encode_insn(&self, insn: &Insn) -> Result<Vec<u8>>;

    /// 编码整个函数
    fn encode_function(&self, func: &SsaFunc) -> Result<Vec<u8>>;

    /// 目标架构
    fn target(&self) -> Arch;

    /// 寄存器信息
    fn reg_info(&self) -> &RegInfo;
}

/// 已编译的 kernel
pub struct CompiledKernel {
    pub elf_bytes: Vec<u8>,         // ELF 二进制
    pub name: String,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub lds_size: u32,
    pub scratch_size: u32,
    pub kernarg_size: usize,
    pub workgroup_size: (u16, u16, u16),
    pub target: Arch,
}

/// 编译器管理器 — 自动选择最优后端
pub struct CompilerManager {
    backends: Vec<Box<dyn CompilerBackend>>,
    cache: HashMap<KernelKey, CompiledKernel>,  // 编译缓存
}

impl CompilerManager {
    /// 编译 kernel (自动选择后端)
    pub fn compile(&mut self, ir: &KernelIr, target: Arch) -> Result<&CompiledKernel> {
        let key = KernelKey::from_ir(ir, target);
        if self.cache.contains_key(&key) {
            return Ok(&self.cache[&key]);
        }

        // 优先用手写编码器 (快, 100μs)
        // 其次用 LLVM (慢, 10-50ms, 但覆盖广)
        for backend in &self.backends {
            if backend.supports(target) {
                let compiled = backend.compile(ir, target)?;
                self.cache.insert(key.clone(), compiled);
                return Ok(&self.cache[&key]);
            }
        }

        Err(format!("No compiler backend for {:?}", target))
    }
}

/// 多后端编译器架构
///   后端 1: t0 FastJit (GFX1100/1200, ~100μs)
///   后端 2: sass-assembler (Pascal/Volta/Ampere, ~100μs)
///   后端 3: LLVM (全平台, ~10-50ms)
///   后端 4: PTX (NVIDIA, ~1ms)
```

### 2.6 调度框架

```rust
// ═══════════════════════════════════════════════════════
// 五层调度框架
// ═══════════════════════════════════════════════════════

// ── Level 5: 图级调度 ──

/// 计算图节点
pub struct GraphNode {
    pub id: NodeId,
    pub kernel: KernelIr,
    pub inputs: Vec<BufferId>,
    pub outputs: Vec<BufferId>,
    pub deps: Vec<NodeId>,  // RAW/WAR 依赖
}

/// 图级调度器 (Kahn toposort + 融合)
pub trait GraphScheduler: Send + Sync {
    /// 拓扑排序 kernel 图
    fn schedule(&self, graph: &[GraphNode]) -> Vec<NodeId>;

    /// 尝试融合相邻 kernel
    fn try_fuse(&self, a: &GraphNode, b: &GraphNode) -> Option<GraphNode>;
}

// ── Level 4: 高层优化调度 ──

/// Tile 配置
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub waves: u32,
    pub split_k: u32,
    pub wgp_mode: bool,
    pub swap_grid: bool,
    pub lds_pad: usize,
    pub use_wmma: bool,
}

/// 高层优化调度器
pub trait TileOptimizer: Send + Sync {
    /// 为给定问题选择最优 tile 配置
    fn optimize(&self, m: u32, n: u32, k: u32, target: &DeviceInfo) -> TileConfig;

    /// 搜索策略
    fn strategy(&self) -> OptimizationStrategy;
}

pub enum OptimizationStrategy {
    /// 分析模型 (Roofline + K-loop, t0-gpu 风格, ~1ms)
    Analytical,
    /// BEAM 搜索 (实测, tinygrad 风格, ~1-10s)
    BeamSearch { beam_width: usize },
    /// 混合 (分析模型初筛 + 实测验证)
    Hybrid { candidates: usize },
}

// ── Level 3: 模板调度 ──

/// Kernel 模板
pub trait KernelTemplate: Send + Sync {
    /// 从 tile 配置生成 kernel IR
    fn generate(&self, config: &TileConfig, target: &DeviceInfo) -> KernelIr;

    /// 模板名称
    fn name(&self) -> &str;
}

// ── Level 2: 指令调度 ──

/// 指令调度器
pub trait InstructionScheduler: Send + Sync {
    /// 调度指令序列
    fn schedule(&self, func: &mut SsaFunc, target: &DeviceInfo);

    /// 调度阶段
    fn phase(&self) -> SchedPhase;
}

pub enum SchedPhase {
    PreRegalloc,    // SSA 级 (延迟隐藏 + 压力感知)
    PostRegalloc,   // 物理寄存器级 (精确依赖)
    SoftwarePipeline, // 循环级 (迭代重叠)
    Pingpong,       // wave 优先级 (s_setprio)
}

/// 调度管线 — 按顺序运行所有调度器
pub struct SchedulingPipeline {
    schedulers: Vec<Box<dyn InstructionScheduler>>,
}

impl SchedulingPipeline {
    /// 默认管线 (t0-gpu 的 4 阶段)
    pub fn default_pipeline() -> Self {
        Self {
            schedulers: vec![
                Box::new(DagScheduler::new()),           // Phase A: SSA 级重排
                Box::new(SoftwarePipeliner::new()),      // Phase B: 软件流水
                Box::new(PostRegallocScheduler::new()),  // Phase C: 物理寄存器 peephole
                Box::new(PingpongScheduler::new()),       // Phase D: wave 优先级
            ],
        }
    }

    /// 运行完整调度管线
    pub fn run(&self, func: &mut SsaFunc, target: &DeviceInfo) {
        for sched in &self.schedulers {
            sched.schedule(func, target);
        }
    }
}

// ── Level 1: 硬件调度 (通过指令影响) ──

/// 硬件调度影响器 — 通过特殊指令影响 GPU 硬件 wave scheduler
pub trait HardwareSchedInfluencer: Send + Sync {
    /// 插入 s_setprio 指令 (AMD)
    fn set_wave_priority(&self, func: &mut SsaFunc, priority: u8);

    /// 插入 waitcnt 指令
    fn insert_waitcnt(&self, func: &mut SsaFunc, vmcnt: u8, lgkmcnt: u8);

    /// 插入 barrier
    fn insert_barrier(&self, func: &mut SsaFunc);
}
```

### 2.7 共享 GPU 调度器

```rust
// ═══════════════════════════════════════════════════════
// 共享 GPU — 多进程/多任务共享
// ═══════════════════════════════════════════════════════

/// 进程/任务 ID
pub type TaskId = u64;

/// GPU 分区 (一个进程/任务的资源配额)
pub struct GpuPartition {
    pub task_id: TaskId,
    pub device_id: u32,
    pub vram_quota: u64,              // 最大 VRAM 使用量
    pub cu_mask: Option<Vec<bool>>,   // 哪些 CU/SM 可用 (None = 全部)
    pub priority: Priority,           // 调度优先级
    pub time_slice_ms: f64,           // 时间片长度 (毫秒)
    pub queue: Box<dyn ComputeQueue>, // 专属计算队列
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,  // 推理服务
}

/// 共享 GPU 调度器
pub struct SharedGpuScheduler {
    device: Arc<dyn GpuDevice>,
    partitions: HashMap<TaskId, GpuPartition>,
    policy: SchedulingPolicy,
    vram_manager: VramQuotaManager,
    cu_partitioner: CuPartitioner,
}

impl SharedGpuScheduler {
    /// 注册任务
    pub fn register(&mut self, task_id: TaskId, config: TaskConfig) -> Result<()> {
        let vram = self.vram_manager.allocate(task_id, config.vram_quota)?;
        let cu_mask = self.cu_partitioner.allocate(task_id, config.cu_fraction)?;
        let queue = self.device.create_compute_queue(QueueConfig::default())?;

        self.partitions.insert(task_id, GpuPartition {
            task_id,
            device_id: self.device.info().id,
            vram_quota: vram,
            cu_mask,
            priority: config.priority,
            time_slice_ms: config.time_slice_ms,
            queue,
        });
        Ok(())
    }

    /// 提交 kernel 到指定任务的队列
    pub fn submit(&mut self, task_id: TaskId, kernel: &dyn Kernel,
                  grid: Grid, block: Block, kernargs: &[u8]) -> Result<()> {
        let partition = self.partitions.get_mut(&task_id)
            .ok_or("Task not registered")?;

        // VRAM 配额检查
        self.vram_manager.check_usage(task_id)?;

        // 提交到任务专属队列
        partition.queue.submit(kernel, grid, block, kernargs, None)
    }

    /// 调度决策 (由定时器或事件驱动)
    pub fn tick(&mut self) {
        match self.policy {
            SchedulingPolicy::FairRoundRobin => self.fair_round_robin(),
            SchedulingPolicy::PriorityPreempt => self.priority_preempt(),
            SchedulingPolicy::Fifo => {},  // 无抢占, 先到先服务
        }
    }
}

pub enum SchedulingPolicy {
    FairRoundRobin,     // 公平轮转
    PriorityPreempt,    // 优先级抢占
    Fifo,               // 先到先服务
}

pub struct TaskConfig {
    pub vram_quota: u64,
    pub cu_fraction: f64,       // 0.0-1.0, 占总 CU 的比例
    pub priority: Priority,
    pub time_slice_ms: f64,
}

/// VRAM 配额管理器
pub struct VramQuotaManager {
    device_total: u64,
    allocations: HashMap<TaskId, u64>,
    usage: HashMap<TaskId, u64>,  // 实时使用量
}

impl VramQuotaManager {
    pub fn allocate(&mut self, task: TaskId, quota: u64) -> Result<u64> {
        let total_allocated: u64 = self.allocations.values().sum();
        if total_allocated + quota > self.device_total {
            return Err("VRAM quota exceeded");
        }
        self.allocations.insert(task, quota);
        Ok(quota)
    }

    pub fn check_usage(&self, task: TaskId) -> Result<()> {
        let usage = self.usage.get(&task).copied().unwrap_or(0);
        let quota = self.allocations.get(&task).copied().unwrap_or(0);
        if usage > quota {
            return Err(format!("Task {} exceeded VRAM quota: {} > {}", task, usage, quota));
        }
        Ok(())
    }
}

/// CU/SM 分区管理器
pub struct CuPartitioner {
    total_cus: u32,
    allocations: HashMap<TaskId, Vec<bool>>,  // CU mask
}

impl CuPartitioner {
    pub fn allocate(&mut self, task: TaskId, fraction: f64) -> Result<Vec<bool>> {
        let cus_to_allocate = (self.total_cus as f64 * fraction) as u32;

        // 找到未分配的 CU
        let mut mask = vec![false; self.total_cus as usize];
        let mut allocated = 0u32;
        let all_masks: Vec<bool> = (0..self.total_cus as usize)
            .map(|i| self.allocations.values().any(|m| m[i]))
            .collect();

        for i in 0..self.total_cus as usize {
            if !all_masks[i] && allocated < cus_to_allocate {
                mask[i] = true;
                allocated += 1;
            }
        }

        if allocated < cus_to_allocate {
            return Err("Not enough CU available");
        }

        self.allocations.insert(task, mask.clone());
        Ok(mask)
    }
}
```

### 2.8 数学库接口

```rust
// ═══════════════════════════════════════════════════════
// 统一数学库接口 — 跨厂商共享
// ═══════════════════════════════════════════════════════

/// BLAS 操作
pub trait BlasLib: Send + Sync {
    /// GEMM: C = alpha * A @ B + beta * C
    fn gemm(&self, queue: &dyn ComputeQueue,
            m: u32, n: u32, k: u32,
            alpha: f32, a: &GpuMemory, lda: u32,
            b: &GpuMemory, ldb: u32,
            beta: f32, c: &GpuMemory, ldc: u32,
            dtype: DType) -> Result<()>;

    /// GEMV: y = alpha * A @ x + beta * y
    fn gemv(&self, queue: &dyn ComputeQueue,
            m: u32, n: u32,
            alpha: f32, a: &GpuMemory, lda: u32,
            x: &GpuMemory, incx: u32,
            beta: f32, y: &GpuMemory, incy: u32) -> Result<()>;

    /// Batched GEMM
    fn gemm_batched(&self, queue: &dyn ComputeQueue,
                    batch: u32, m: u32, n: u32, k: u32,
                    a: &GpuMemory, b: &GpuMemory, c: &GpuMemory,
                    dtype: DType) -> Result<()>;
}

/// 并行原语
pub trait PrimLib: Send + Sync {
    /// 前缀和 (exclusive scan)
    fn scan_exclusive(&self, queue: &dyn ComputeQueue,
                      output: &GpuMemory, input: &GpuMemory,
                      n: u32, dtype: DType) -> Result<()>;

    /// 归约 (sum/max/min)
    fn reduce(&self, queue: &dyn ComputeQueue,
              output: &GpuMemory, input: &GpuMemory,
              n: u32, op: ReduceOp, dtype: DType) -> Result<()>;

    /// 基数排序
    fn radix_sort(&self, queue: &dyn ComputeQueue,
                  keys: &GpuMemory, values: Option<&GpuMemory>,
                  n: u32) -> Result<()>;

    /// 直方图
    fn histogram(&self, queue: &dyn ComputeQueue,
                 output: &GpuMemory, input: &GpuMemory,
                 n: u32, bins: u32) -> Result<()>;
}

pub enum ReduceOp { Sum, Max, Min, Prod }

/// 随机数生成
pub trait RngLib: Send + Sync {
    /// 生成均匀分布 [0, 1)
    fn uniform(&self, queue: &dyn ComputeQueue,
               output: &GpuMemory, n: u32, seed: u64) -> Result<()>;

    /// 生成正态分布 N(0, 1)
    fn normal(&self, queue: &dyn ComputeQueue,
              output: &GpuMemory, n: u32, seed: u64) -> Result<()>;

    /// 生成伯努利分布 (dropout 用)
    fn bernoulli(&self, queue: &dyn ComputeQueue,
                 output: &GpuMemory, n: u32, p: f32, seed: u64) -> Result<()>;
}

/// FFT
pub trait FftLib: Send + Sync {
    fn fft_1d(&self, queue: &dyn ComputeQueue,
              output: &GpuMemory, input: &GpuMemory,
              n: u32, direction: FftDirection) -> Result<()>;
}

pub enum FftDirection { Forward, Inverse }

/// 数学库管理器 — 自动选择最优实现
pub struct MathLibManager {
    blas: Box<dyn BlasLib>,
    prim: Box<dyn PrimLib>,
    rng: Box<dyn RngLib>,
    fft: Box<dyn FftLib>,
}

impl MathLibManager {
    /// 自动选择: 优先用 t0-gpu JIT (快), fallback 到 vendor 库
    pub fn auto_select(device: &DeviceInfo) -> Self {
        match device.vendor {
            Vendor::AMD => Self {
                blas: Box::new(T0BlasLib::new(device)),     // t0-gpu JIT GEMM
                prim: Box::new(T0PrimLib::new(device)),     // t0-gpu scan/sort
                rng: Box::new(T0RngLib::new(device)),       // t0-gpu Philox
                fft: Box::new(T0FftLib::new(device)),       // t0-gpu radix-2
            },
            Vendor::NVIDIA => Self {
                blas: Box::new(CublasLib::new(device)),     // cuBLAS wrapper
                prim: Box::new(CubLib::new(device)),        // CUB wrapper
                rng: Box::new(CurandLib::new(device)),      // cuRAND wrapper
                fft: Box::new(CufftLib::new(device)),       // cuFFT wrapper
            },
            _ => Self {
                blas: Box::new(T0BlasLib::new(device)),     // 通用 fallback
                prim: Box::new(T0PrimLib::new(device)),
                rng: Box::new(T0RngLib::new(device)),
                fft: Box::new(T0FftLib::new(device)),
            },
        }
    }

    pub fn blas(&self) -> &dyn BlasLib { &*self.blas }
    pub fn prim(&self) -> &dyn PrimLib { &*self.prim }
    pub fn rng(&self) -> &dyn RngLib { &*self.rng }
    pub fn fft(&self) -> &dyn FftLib { &*self.fft }
}
```

### 2.9 监控与性能

```rust
// ═══════════════════════════════════════════════════════
// 监控与性能分析
// ═══════════════════════════════════════════════════════

/// GPU 性能计数器
pub struct GpuCounters {
    pub gpu_utilization: f64,       // 0.0-1.0
    pub memory_utilization: f64,    // 0.0-1.0
    pub vram_used: u64,             // 字节
    pub vram_total: u64,
    pub temperature: f64,           // 摄氏度
    pub power_watts: f64,
    pub clock_mhz: u32,
    pub memory_clock_mhz: u32,
    pub pcie_throughput_gbps: f64,
}

/// Kernel 性能分析
pub struct KernelProfile {
    pub name: String,
    pub grid: Grid,
    pub block: Block,
    pub duration_us: f64,           // 执行时间 (微秒)
    pub vgpr_used: u32,
    pub sgpr_used: u32,
    pub lds_used: u32,
    pub occupancy: f64,             // 0.0-1.0
    pub achieved_tflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub compute_bound: bool,        // true = compute, false = memory
}

/// 性能分析器
pub trait Profiler: Send + Sync {
    /// 开始记录
    fn start(&mut self);

    /// 停止记录
    fn stop(&mut self) -> Vec<KernelProfile>;

    /// 实时计数器
    fn counters(&self) -> GpuCounters;
}

/// 监控守护进程
pub struct GpuMonitor {
    devices: Vec<Arc<dyn GpuDevice>>,
    interval: Duration,
    history: Vec<(Instant, Vec<GpuCounters>)>,
}

impl GpuMonitor {
    pub fn poll(&mut self) {
        let now = Instant::now();
        let counters: Vec<GpuCounters> = self.devices.iter()
            .map(|d| d.read_counters().unwrap_or_default())
            .collect();
        self.history.push((now, counters));

        // 保留最近 1 小时
        let cutoff = now - Duration::from_secs(3600);
        self.history.retain(|(t, _)| *t > cutoff);
    }
}
```

## 三、模块结构

```
t0-universal-runtime/
├── Cargo.toml
├── src/
│   ├── lib.rs                         — crate 入口
│   │
│   ├── core/                          — 核心 trait + 类型
│   │   ├── mod.rs
│   │   ├── device.rs                  — GpuDevice, DeviceInfo, DeviceManager
│   │   ├── memory.rs                  — GpuMemory, MemType, MemoryManager
│   │   ├── queue.rs                   — ComputeQueue, CopyQueue, Signal
│   │   ├── kernel.rs                  — Kernel, Grid, Block
│   │   ├── arch.rs                    — Vendor, Arch, Target
│   │   └── error.rs                   — 统一错误类型
│   │
│   ├── driver/                        — 厂商驱动实现
│   │   ├── mod.rs
│   │   ├── amd/
│   │   │   ├── mod.rs
│   │   │   ├── kfd.rs                 — 从 t0-gpu kfd/mod.rs 移植
│   │   │   ├── aql.rs                 — AQL packet 构造
│   │   │   └── amdgpu.rs              — DRM amdgpu 接口
│   │   ├── nvidia/
│   │   │   ├── mod.rs
│   │   │   ├── nvrm.rs                — RM ioctl 封装
│   │   │   ├── uvm.rs                 — UVM 内存管理
│   │   │   ├── gpfifo.rs              — GPFIFO ring + doorbell
│   │   │   └── qmd.rs                 — QMD 构造
│   │   └── ascend/
│   │       ├── mod.rs
│   │       └── davinci.rs             — 昇腾驱动
│   │
│   ├── compiler/                      — 编译器后端
│   │   ├── mod.rs
│   │   ├── ir.rs                      — 统一 IR (KernelIr, SsaFunc, Insn)
│   │   ├── ssa.rs                     — SSA IR + 优化 passes
│   │   ├── tile_ir.rs                 — Tile IR (从 t0-gpu 移植)
│   │   ├── regalloc.rs                — 寄存器分配
│   │   ├── backend/
│   │   │   ├── mod.rs
│   │   │   ├── t0_fastjit.rs          — t0-gpu 快速 JIT (GFX1100/1200)
│   │   │   ├── sass_asm.rs            — sass-assembler (Pascal/Volta/Ampere)
│   │   │   ├── llvm.rs                — LLVM 后端 (全平台)
│   │   │   └── ptx.rs                 — PTX 文本生成 (NVIDIA)
│   │   └── code_object/
│   │       ├── mod.rs
│   │       ├── hsa_elf.rs             — AMD HSA ELF (从 rdna3_code_object 移植)
│   │       └── cubin.rs               — NVIDIA CUBIN ELF
│   │
│   ├── scheduler/                     — 调度框架
│   │   ├── mod.rs
│   │   ├── graph/
│   │   │   ├── mod.rs
│   │   │   └── kahn.rs                — Kahn toposort (tinygrad 风格)
│   │   ├── tile/
│   │   │   ├── mod.rs
│   │   │   ├── cost_model.rs          — Roofline + K-loop (t0-gpu 风格)
│   │   │   ├── beam_search.rs         — BEAM 搜索 (tinygrad 风格)
│   │   │   └── hybrid.rs              — 混合策略
│   │   ├── template/
│   │   │   ├── mod.rs
│   │   │   └── kernel_templates.rs    — GEMM/Attention/Elementwise 模板
│   │   ├── instruction/
│   │   │   ├── mod.rs
│   │   │   ├── dag_sched.rs           — SSA 级两阶段调度 (t0-gpu 风格)
│   │   │   ├── post_regalloc.rs       — 物理寄存器 peephole
│   │   │   ├── sw_pipeline.rs         — 软件流水
│   │   │   ├── pingpong.rs            — wave 优先级 (s_setprio)
│   │   │   └── latency.rs             — 指令延迟模型
│   │   └── shared/
│   │       ├── mod.rs
│   │       ├── time_slice.rs          — 时间片调度
│   │       ├── vram_quota.rs          — VRAM 配额
│   │       ├── cu_partition.rs        — CU/SM 分区
│   │       └── priority.rs            — 优先级队列
│   │
│   ├── math/                          — 数学库
│   │   ├── mod.rs
│   │   ├── blas/
│   │   │   ├── mod.rs
│   │   │   ├── gemm.rs                — JIT GEMM (从 t0-gpu gemm_gen 移植)
│   │   │   ├── gemv.rs                — GEMV (新实现)
│   │   │   └── batched.rs             — Batched GEMM (新实现)
│   │   ├── prim/
│   │   │   ├── mod.rs
│   │   │   ├── scan.rs                — Prefix scan (新实现)
│   │   │   ├── sort.rs                — Radix sort (新实现)
│   │   │   └── reduce.rs              — Reduce (从 t0-gpu 提取)
│   │   ├── rng/
│   │   │   ├── mod.rs
│   │   │   ├── philox.rs              — Philox PRNG (新实现)
│   │   │   └── distributions.rs       — 分布采样 (新实现)
│   │   └── fft/
│   │       ├── mod.rs
│   │       └── radix2.rs              — Radix-2 FFT (新实现)
│   │
│   ├── runtime/                       — 运行时支撑
│   │   ├── mod.rs
│   │   ├── mem_pool.rs                — 池分配器 (从 ignis buffer_pool 移植)
│   │   ├── mem_buddy.rs               — 伙伴分配器
│   │   ├── mem_bump.rs                — Bump 分配器
│   │   ├── profiler.rs                — 性能分析
│   │   └── monitor.rs                 — GPU 监控
│   │
│   └── nn/                            — 神经网络框架 (从 ignis 移植)
│       ├── mod.rs
│       ├── tensor.rs
│       ├── tape.rs                    — 自动微分
│       ├── ops/                       — 运算层
│       ├── nn/                        — 网络层
│       └── training/                  — 训练支撑 (LossScaler, LrScheduler)
│
├── tests/                             — 集成测试
├── benches/                           — 基准测试
└── examples/                          — 示例
```

## 四、初始化流程

```rust
// ═══════════════════════════════════════════════════════
// 使用示例
// ═══════════════════════════════════════════════════════

fn main() -> Result<()> {
    // 1. 设备发现 (自动检测 AMD/NVIDIA/国产 GPU)
    let mgr = DeviceManager::discover();
    println!("发现 {} 个 GPU", mgr.devices().len());
    for d in mgr.devices() {
        println!("  [{}] {} {} ({:.1} TFLOPS, {} GB VRAM)",
            d.id, d.vendor, d.name, d.compute_tflops,
            d.vram_size / 1024 / 1024 / 1024);
    }

    // 2. 打开设备
    let device = mgr.open(0)?;

    // 3. 创建数学库
    let math = MathLibManager::auto_select(device.info());

    // 4. 创建编译器 (自动选择后端)
    let mut compiler = CompilerManager::new();
    compiler.register(Box::new(T0FastJitBackend::new()));  // GFX1100/1200
    compiler.register(Box::new(SassAsmBackend::new()));     // Pascal/Volta/Ampere
    compiler.register(Box::new(LlvmBackend::new()));        // 全平台 fallback

    // 5. 创建调度管线
    let sched = SchedulingPipeline::default_pipeline();

    // 6. 创建共享调度器 (可选, 多任务场景)
    let mut shared = SharedGpuScheduler::new(device.clone(), SchedulingPolicy::PriorityPreempt);
    shared.register(1, TaskConfig {
        vram_quota: 4 * 1024 * 1024 * 1024,  // 4 GB
        cu_fraction: 0.5,
        priority: Priority::High,
        time_slice_ms: 10.0,
    })?;

    // 7. 使用
    let queue = device.create_compute_queue(QueueConfig::default())?;

    // GEMM
    math.blas().gemm(&*queue, 4096, 4096, 512,
        1.0, &a_buf, 4096, &b_buf, 512,
        0.0, &c_buf, 4096, DType::BF16)?;

    // 或者用 T0 编译器自定义 kernel
    let ir = KernelIr::from_tile_config(TileConfig {
        tile_m: 128, tile_n: 128, tile_k: 32,
        use_wmma: true, ..Default::default()
    });
    let compiled = compiler.compile(&ir, device.info().arch)?;
    let kernel = device.load_kernel(&compiled)?;
    queue.submit(&*kernel, Grid(32, 32, 1), Block(256, 1, 1), &kernargs, None)?;

    // 8. 同步
    queue.wait_idle()?;

    Ok(())
}
```

## 五、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言 | Rust | t0-gpu 已验证, 零成本抽象, 内存安全 |
| ISA 编码 | 双后端 (手写 + LLVM) | 手写快 (100μs), LLVM 覆盖广 |
| 调度 | 5 层分层 | 不同粒度解决不同问题 |
| 数学库 | 独立 crate | 跨厂商共享, 与驱动解耦 |
| 共享 GPU | 用户态 | 零内核修改, 可移植 |
| 内存模型 | 先分离后统一 | UVM 跨厂商太复杂 |
| 编译缓存 | 磁盘 + 内存 | 首次编译慢, 后续快 |
| 插件化 | trait object | 厂商后端可动态加载 |

## 六、实施优先级

```
P0 (立刻):  core/ trait 定义 + AmdDriver 适配 + t0 FastJit 后端
P1 (近期):  math/ 数学原语 (scan/sort/rand) + NvDriver
P2 (中期):  scheduler/ 5 层调度 + shared/ 共享 GPU
P3 (远期):  ascend/ + llvm 后端 + 完整 nn/ 框架
```
