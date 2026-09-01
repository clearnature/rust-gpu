# ROCm 运行时栈分析

> 源码位置: /data/ROCm/rocm-systems/

## 四层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: HIP Runtime  (libhip_runtime.so)                  │
│  ─────────────────────────────────────────                  │
│  职责: CUDA 兼容 API 层                                      │
│  • 提供 hipMalloc/hipMemcpy/hipLaunchKernel 等 CUDA 等价 API │
│  • 翻译 CUDA 语法 → HSA 语义 (hipcc 编译期)                  │
│  • 管理 stream、event、module (加载 .hsaco 文件)              │
│  • 源码: hip/include/hip/hip_runtime_api.h (10,543 行)       │
│  • 实现: rocm-systems/projects/clr/hipamd/src/               │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用 hsa_* API
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 3: ROCr (HSA Runtime, libhsa-runtime64.so)           │
│  ─────────────────────────────────────────────              │
│  职责: GPU 资源管理 & 调度引擎                                │
│  • 枚举 GPU agents (hsa_agent_t)                             │
│  • 管理 memory pool (VRAM/GTT/System)                        │
│  • 创建 AQL 队列, 投递 64-byte dispatch packet               │
│  • 信号量 (hsa_signal_t) — GPU/CPU 同步原语                  │
│  • 源码: rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/ │
│  • 通过 dlopen("libhsakmt.so") + dlsym 加载 libhsakmt       │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用 hsakmt_* API
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 2: libhsakmt (libhsakmt.so)                          │
│  ─────────────────────────────────────────                  │
│  职责: KFD ioctl 的 C 封装 + 内存管理器                      │
│  • hsakmt_alloc_memory() → ioctl(AMDKFD_IOC_ALLOC_MEMORY)   │
│  • hsakmt_create_queue() → ioctl(AMDKFD_IOC_CREATE_QUEUE)   │
│  • fmm.c (4719 行) — Frame buffer Memory Manager            │
│  • 源码: rocm-systems/projects/rocr-runtime/libhsakmt/       │
└──────────────────────────┬──────────────────────────────────┘
                           │ ioctl(/dev/kfd)
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 1: KFD Kernel Driver (amdgpu.ko)                     │
│  ─────────────────────────────────────────                  │
│  职责: 硬件资源管理 (内核态)                                  │
│  • GPU VA 空间分配 (GPU 页表管理)                             │
│  • VRAM 物理内存分配 (GTT / VRAM / scratch)                  │
│  • AQL 硬件队列创建 (映射到 GPU CWSR 上下文)                 │
│  • 内存映射 (CPU mmap VRAM → BAR1, 或 GTT → PCIe DMA)       │
│  • ioctl 定义: libhsakmt/include/hsakmt/linux/kfd_ioctl.h    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     ┌─────▼─────┐
                     │  AMD GPU  │
                     └───────────┘
```

## 一条 hipMalloc 的完整路径 (5 层调用链)

```
hipMalloc(8MB)
  ↓  hip_memory.cpp:778
ihipMalloc() → amd::SvmBuffer::malloc()
  ↓  rocclr/device/rocm/rocdevice.cpp:2229
Device::svmAlloc() → hsa_amd_memory_pool_allocate()
  ↓  dlsym("hsa_amd_memory_pool_allocate")  ← ROCr 动态加载 HSA 符号
hsa_memory_allocate() → KfdDriver::AllocateMemory()
  ↓  amd_kfd_driver.cpp:536
HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, flags))
  ↓  dlopen("libhsakmt.so") + dlsym  ← ThunkLoader 模式
hsaKmtAllocMemory() → hsakmt_fmm_allocate_device()
  ↓  fmm.c:1216  (4719 行的 Frame buffer Memory Manager)
hsakmt_ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args)
  ↓  ioctl(/dev/kfd)
内核: TTM 分配器分配 VRAM 物理页 → 创建 GPU VA 映射 → 返回 handle
  ↓
映射回 KfdDriver → hsaKmtMapMemoryToGPUNodes()
  ↓  AMDKFD_IOC_MAP_MEMORY_TO_GPU
内核: 建立 GPU 页表条目 → 内存对 GPU 可见
```

## 一条 hipLaunchKernel 的完整路径

```
hipLaunchKernel(func, grid, block, args)
  ↓  hip_platform.cpp:689
ihipLaunchKernel() → 解析 host function → hipFunction_t
  ↓  hip_module.cpp:443
ihipModuleLaunchKernel() → 创建 NDRangeKernelCommand
  ↓  command->enqueue()
CLR stream worker 写 AQL packet 到 ring buffer:
  ├─ header = HSA_PACKET_TYPE_KERNEL_DISPATCH (2)
  ├─ kernel_object = GPU 代码地址
  ├─ kernarg_address = 参数指针
  └─ workgroup_size / grid_size
  ↓  amd_aql_queue.cpp:474
*(hardware_doorbell_ptr) = value  ← MMIO 直写 doorbell
  ↓
GPU Command Processor 读 AQL → 分发到 Compute Units
```

## 层间通信机制

| 边界 | 机制 |
|------|------|
| HIP → CLR/ROCr | C++ 虚函数调用; CLR 通过 dlsym 动态加载 HSA 符号 |
| ROCr → libhsakmt | dlopen("libhsakmt.so") + dlsym, 通过 HSAKMT_CALL() 宏 |
| libhsakmt → KFD | ioctl(fd, request, arg) on /dev/kfd, 包装在 hsakmt_ioctl() |
| GPU 通知 | Doorbell MMIO 写: 直接 *(hardware_doorbell_ptr) = value |

## 为什么分四层 (源码级原因)

### 原因 1: ROCr 通过 dlopen 加载 libhsakmt — 运行时可替换

```cpp
// thunk_loader.h:57
#define HSAKMT_CALL(function_name) \
    core::Runtime::runtime_singleton_->thunkLoader()->pfn_##function_name
```

ROCr 不直接链接 libhsakmt, 而是运行时 dlopen. 可以加载 libdtif.so (DXGI 后端) 或 librocdxg.so (DXG 后端) 替代.

### 原因 2: libhsakmt 不只是 ioctl 包装 — 它有 4719 行的内存管理器

fmm.c (Frame buffer Memory Manager) 是 libhsakmt 的真正核心:
- 管理 GPU 虚拟地址空间分配 (类似用户态的 GPU 页表)
- 处理内存映射的缓存一致性
- 管理 doorbell page 的 mmap
- 处理 CWSR (Context Save/Restore) 内存

### 原因 3: CLR 层存在是为了同时支持 OpenCL 和 HIP

```
CLR (Compute Language Runtime)
├── hipamd/    ← HIP 的实现
├── rocclr/    ← 共享的 device 抽象
│   └── device/rocm/ ← ROCr 后端
│   └── device/pal/  ← PAL 后端 (图形)
└── opencl/    ← OpenCL 的实现
```

### 原因 4: 各层的存在都是为了解决特定问题

| 层 | 解决的问题 |
|----|-----------|
| HIP | CUDA 生态兼容 |
| ROCr | HSA 标准化抽象 (agent/queue/signal/memory pool) |
| libhsakmt | 内核 ioctl 编号变化的稳定性防火墙 |
| KFD | 硬件唯一入口, 安全/资源管理 |

## t0-gpu 的简化

t0-gpu 把四层合并为一层 Rust 代码:

```
ROCm: 5 层内存分配, 3 次 dlsym, 2 次 ioctl, ~20μs
t0-gpu: 1 层, 0 次 dlsym, 1 次 ioctl, ~2μs
```

关键: t0-gpu 的 kfd/mod.rs (3235 行) 重新实现了 libhsakmt 的 fmm.c + queues.c + topology.c 的有效子集, 但更精简.
