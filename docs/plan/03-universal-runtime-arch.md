# 通用 GPU 运行时架构设计

> 目标: 从 t0-gpu (AMD 单厂商) 扩展为支持 AMD/NVIDIA/国产 GPU 的通用运行时

## 各厂商驱动接口对比

```
┌──────────────┬──────────────┬──────────────┬────────────────────────┐
│   AMD        │   NVIDIA     │   华为昇腾    │   摩尔线程 / 壁仞 / 燧原│
│   KFD        │   nvidia.ko  │   davinci.ko │   vendor-specific      │
│   /dev/kfd   │   /dev/nvidia│   /dev/davinci│   /dev/mtgpu 等       │
│              │   -uvm       │              │                        │
│  ioctl:      │  ioctl:      │  ioctl:      │  ioctl:                │
│  CREATE_QUEUE│  ALLOC_MEM   │  ALLOC_MEM   │  各自定义               │
│  ALLOC_MEMORY│  CREATE_CTX  │  CREATE_STREAM│                       │
│  MAP_MEMORY  │  LAUNCH_KERNEL│ LAUNCH_KERNEL│                       │
├──────────────┼──────────────┼──────────────┼────────────────────────┤
│  AQL packet  │  GPFIFO      │  任务图       │  各自定义               │
│  64 bytes    │  channel     │  TSCH 通道    │                        │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

## 五层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: 应用层 (ignis / t0 compiler / PyTorch binding)        │
│  不关心厂商, 只调用统一 API                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 统一 trait
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 4: 统一调度层 (Unified Dispatch Layer)                    │
│  trait GpuDevice / GpuCompiler / GpuDriver                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 3: 厂商 HAL (Vendor Hardware Abstraction Layer)           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ AmdBackend  │ │ NvidiaBackend│ │AscendBackend│ │ Generic  │ │
│  │ KFD ioctl   │ │ nvidia.ko   │ │ davinci.ko  │ │ trait obj│ │
│  │ AQL queue   │ │ GPFIFO chan │ │ 任务队列     │ │ 自定义    │ │
│  │ GFX ISA     │ │ SASS/PTX    │ │ Ascend ISA  │ │ 自定义    │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 2: 共享 GPU 调度器 (Shared GPU Scheduler)                 │
│  时间片轮转 / 优先级队列 / 资源隔离 / 故障隔离                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Layer 1: 内核驱动 (各厂商的 .ko, 不修改)                        │
└─────────────────────────────────────────────────────────────────┘
```

## 核心 Trait 设计

### Driver HAL

```rust
trait GpuDriver: Send + Sync {
    type Device: GpuDevice;
    fn enumerate(&self) -> Vec<DeviceInfo>;
    fn open(&self, id: u32) -> Result<Self::Device>;
}

trait GpuDevice: Send + Sync {
    fn alloc_vram(&self, size: usize, flags: MemFlags) -> Result<GpuMemory>;
    fn free_vram(&self, mem: GpuMemory) -> Result<()>;
    fn map_to_cpu(&self, mem: &GpuMemory) -> Result<*mut u8>;
    fn create_queue(&self, config: QueueConfig) -> Result<GpuQueue>;
    fn submit(&self, queue: &GpuQueue, packet: &[u8]) -> Result<()>;
    fn wait(&self, signal: Signal, timeout: Duration) -> Result<()>;
    fn info(&self) -> DeviceInfo;
}
```

### ISA HAL

```rust
trait IsaEncoder {
    fn encode(&self, ir: &SsaIr) -> Vec<u8>;  // 机器码
    fn target(&self) -> Target;
    fn registers(&self) -> RegInfo;
    fn supports(&self, feature: Feature) -> bool;
}
```

### 编译器 HAL

```rust
trait CompilerBackend {
    fn compile(&self, ir: &SsaIr, target: Target) -> Result<CodeObject>;
    fn compile_time(&self) -> Duration;
    fn supports(&self, target: Target) -> bool;
}
```

## 各厂商实现现状

| 组件 | AMD | NVIDIA | 华为昇腾 |
|------|-----|--------|---------|
| **Driver HAL** | ✅ t0-gpu KFD | ❌ 需写 | ❌ 需写 |
| **ISA 编码器** | ✅ t0-gpu rdna3_asm.rs | ⚠️ sass-assembler Pascal/Volta/Ampere | ❌ 需写 |
| **Code Object** | ✅ rdna3_code_object.rs (HSA ELF) | ⚠️ 需 .cubin ELF writer | ❌ 需写 |
| **数学库** | ✅ GEMM/Attention 等 | 可共享 | 可共享 |
| **调度器** | 可共享 | 可共享 | 可共享 |

## ISA 编码器双后端策略

```
后端 1: 手写编码器 (快速 JIT, 性能最优)
  T0 IR → rdna3_asm.rs → GFX ISA (100μs)
  T0 IR → sass-assembler → SASS (100μs)
  用于: GEMM autotune、小 kernel、需要低延迟编译的场景

后端 2: LLVM 后端 (跨厂商, 编译慢但覆盖广)
  T0 IR → LLVM IR → llc → 各厂商 ISA (10-50ms)
  用于: 大 kernel、新硬件、没有手写编码器的目标

后端 3: PTX 中间表示 (NVIDIA 特有)
  T0 IR → PTX 文本 → NVIDIA 驱动运行时编译
  用于: 快速支持新 NVIDIA 硬件, 比手写 SASS 简单
```

## 共享 GPU 调度器

```rust
struct SharedScheduler {
    backends: Vec<Box<dyn GpuDriver>>,
    partitions: HashMap<ProcessId, GpuPartition>,
    policy: SchedulingPolicy,
}

struct GpuPartition {
    device_id: u32,
    vram_quota: usize,           // 最大 VRAM 使用量
    cu_mask: Option<Vec<bool>>,  // 哪些 CU/SM 可用
    priority: Priority,          // 调度优先级
    time_slice: Duration,        // 时间片长度
}
```

## 实现路线图

```
Phase 1: 抽象层 (2-3 周)
├─ 从现有 kfd/mod.rs 提取 trait GpuDriver / GpuDevice
├─ 现有 KFD 代码封装为 AmdBackend
├─ 接口不变, ignis/t0 通过 trait 调用
└─ 验证: 现有所有测试仍然通过

Phase 2: NVIDIA 后端 (4-6 周)
├─ 研究 nvidia.ko ioctl 接口 (参考 NVRM driver)
├─ 实现 NvidiaBackend: 内存、GPFIFO、dispatch
├─ ISA 编码器: SASS (参考 sass-assembler)
├─ Code object: ELF (CUDA fatbin 或 CUBIN)
└─ 验证: 简单 kernel 在 NVIDIA GPU 上跑通

Phase 3: 共享调度器 (3-4 周)
├─ 时间片调度 (timer-based preemption)
├─ VRAM 配额管理 (cgroup-like)
├─ CU/SM 分区 (AMD: CU_MASK, NVIDIA: MIG-like)
└─ 验证: 多进程同时跑不同 GPU 任务

Phase 4: 国产 GPU (按需)
├─ 华为昇腾: davinci.ko ioctl + Ascend ISA
├─ 摩尔线程: MUSA driver + MUSA ISA
├─ 壁仞: BIRENSUPA driver + BIR ISA
└─ 每个后端 4-8 周

Phase 5: 跨厂商调度 (高级)
├─ 混合调度: 一张 AMD + 一张 NVIDIA + 一张昇腾
├─ 统一内存模型 (不同 VRAM 空间的映射)
├─ 跨设备通信 (PCIe P2P / NVLink / xGMI)
└─ 每增加一种组合 +2-3 周
```

## 关键架构决策

| 决策 | 选项 A | 选项 B | 建议 |
|------|--------|--------|------|
| ISA 编码 | 每厂商手写编码器 | LLVM IR 转译 | 双后端并存 |
| 共享调度 | 内核态 (改驱动) | 用户态 Rust scheduler | 用户态 |
| 多厂商发现 | 编译时 feature flag | 运行时动态加载 | 运行时 |
| 内存模型 | 统一虚拟地址 (UVA) | 分离地址空间 | 先分离后加 UVA |
