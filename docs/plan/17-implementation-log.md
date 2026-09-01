# 实现日志

> 日期: 2026-08-24
> 阶段: Phase 0 — 框架骨架

## 已完成

### 1. 创建 universal 模块结构 (1086 行 Rust)

```
src/universal/
├── mod.rs              (16 行)  — 公共导出
├── core/
│   ├── mod.rs          (8 行)
│   ├── arch.rs         (95 行)  — Vendor/Arch/DType 定义
│   └── device.rs       (259 行) — 核心 trait (GpuDevice/ComputeQueue/Signal/Kernel/DriverFactory/DeviceManager)
├── driver/
│   ├── mod.rs          (1 行)
│   └── amd/
│       ├── mod.rs      (2 行)
│       └── kfd.rs      (420 行) — AmdDriver/AmdDevice (KFD ioctl)
├── compiler/
│   └── mod.rs          (64 行)  — CompilerBackend/IsaEncoder/KernelIr trait
├── scheduler/
│   └── mod.rs          (67 行)  — TileOptimizer/InstructionScheduler trait
├── math/
│   └── mod.rs          (88 行)  — BlasLib/PrimLib/RngLib trait
└── runtime/
    └── mod.rs          (66 行)  — MemoryManager/PoolAllocator
```

### 2. 已定义的 trait (11 个)

| Trait | 文件 | 说明 |
|-------|------|------|
| `GpuDevice` | core/device.rs | 设备操作 (alloc/free/copy/queue/kernel) |
| `ComputeQueue` | core/device.rs | 计算队列 (submit/barrier/flush/wait) |
| `CopyQueue` | core/device.rs | DMA 拷贝队列 |
| `Signal` | core/device.rs | GPU 信号量 |
| `Kernel` | core/device.rs | 已编译 kernel 句柄 |
| `DriverFactory` | core/device.rs | 驱动工厂 (enumerate/open) |
| `CompilerBackend` | compiler/mod.rs | 编译器后端 |
| `IsaEncoder` | compiler/mod.rs | ISA 编码器 |
| `TileOptimizer` | scheduler/mod.rs | Tile 优化器 |
| `BlasLib` | math/mod.rs | BLAS 操作 |
| `PrimLib` | math/mod.rs | 并行原语 (scan/sort/reduce) |
| `RngLib` | math/mod.rs | 随机数生成 |
| `MemoryManager` | runtime/mod.rs | 内存管理器 |

### 3. AmdDriver 实现

- KFD ioctl 封装 (通过 libc)
- `/dev/kfd` 设备打开
- GPU 拓扑枚举 (sysfs)
- VRAM 分配 (ALLOC_MEMORY + MAP_MEMORY_TO_GPU)
- CPU 映射 (mmap)
- 数据传输 (copy_from_host / copy_to_host)

### 4. 编译状态

```
cargo check --lib → 0 errors, 52 warnings (全部是已有代码的警告)
```

## 待完成 (Phase 1)

- [ ] 从 t0-gpu kfd/mod.rs 移植 AQL 队列创建 (create_compute_queue)
- [ ] 从 t0-gpu kfd/mod.rs 移植 ELF 加载 (load_kernel)
- [ ] 从 t0-gpu kfd/mod.rs 移植信号量 (create_signal)
- [ ] 在 RX 9060 XT 上验证基本操作
- [ ] NVIDIA 后端 (参考 tinygrad ops_nv.py)
- [ ] 数学库实现 (BlasLib 等)
