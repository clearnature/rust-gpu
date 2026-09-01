# 内存模型对比分析

> 分析范围: AMD KFD vs NVIDIA vs 国产 GPU 的内存管理接口

## 各厂商内存管理接口

### AMD KFD (t0-gpu 已实现)

```rust
// 内存分配
ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &AllocMemoryArgs {
    va_addr: 0,           // 0 = KFD 自动选择 VA
    size: bytes,
    handle: 0,            // 输出: 内存句柄
    gpu_id: device_id,
    flags: VRAM | PUBLIC | COHERENT,
});

// CPU 映射 (BAR1 或 GTT)
mmap(kfd_fd, mmap_offset, size, PROT_READ|PROT_WRITE, MAP_SHARED, offset);

// GPU 映射 (建立 GPU 页表)
ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &MapMemoryArgs {
    handle: alloc_handle,
    device_ids_array_ptr: &gpu_ids,
    n_devices: num_gpus,
    // 输出: gpu_vm_addr (GPU 虚拟地址)
});

// 内存类型
// VRAM: GPU 本地显存 (最快)
// GTT: 系统内存, GPU 可通过 PCIe 访问 (较慢)
// Scratch: 每线程私有内存 (寄存器溢出用)
```

### NVIDIA (tinygrad 实现)

```python
# tinygrad/runtime/ops_nv.py
# 内存分配 — 通过 UVM (Unified Virtual Memory) 或直接 ioctl
nv_ioctl_alloc_memory(fd, {
    'size': size,
    'handle': 0,  # 输出
    'type': NV_MEMORY_DEVICE,  # GPU 内存
})

# 或通过 UVM
uvm_ioctl(fd, UVM_ALLOCATE, {
    'size': size,
    'type': UVM_MEMORY_DEVICE,
    'gpu_uuid': gpu_uuid,
})

# CPU 映射
mmap(fd, offset, size, PROT_READ|PROT_WRITE, MAP_SHARED)

# GPU 映射 — 通过 GPU VA space 管理
nv_ioctl_map_memory(fd, {
    'handle': alloc_handle,
    'gpu_vaspace': vaspace_handle,
    'gpu_offset': va_addr,
})
```

**NVIDIA 内存类型:**
- **Device Memory**: GPU 本地 VRAM
- **System Memory (pinned)**: CPU 内存, GPU 可通过 PCIe 访问
- **Unified Memory (UVM)**: 自动迁移的统一地址空间
- **Managed Memory**: 类似 UVM 但有不同迁移策略

### 华为昇腾 (参考)

```c
// 内存分配
aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
// 或通过驱动 ioctl
ioctl(davinci_fd, DAVINCI_IOC_ALLOC, {size, flags});

// 内存类型
// DDR: 设备内存 (类似 VRAM)
// HBM: 高带宽内存 (高端型号)
// SMMU: 系统内存映射
```

## 内存模型对比

| 特性 | AMD KFD | NVIDIA | 华为昇腾 |
|------|---------|--------|---------|
| 分配 API | ioctl (ALLOC_MEMORY_OF_GPU) | ioctl (ALLOC_MEMORY) / UVM | ioctl (DAVINCI_IOC_ALLOC) |
| CPU 映射 | mmap(kfd_fd) | mmap(nv_fd) / UVM 自动 | mmap(dev_fd) |
| GPU 映射 | ioctl(MAP_MEMORY_TO_GPU) | GPU VA space 管理 | SMMU 映射 |
| 统一地址空间 | 有限 (SVM) | 完整 (UVM) | 有限 |
| 多 GPU 共享 | 支持 (指定 device_ids) | UVM 自动 / NVLink P2P | 支持 |
| 内存类型 | VRAM/GTT/Scratch | Device/System/UVM/Managed | DDR/HBM/SMMU |
| 页大小 | 4KB (KFD 对齐) | 4KB / 64KB (大页) | 4KB |
| 虚拟地址管理 | KFD 内部管理 | 用户态管理 (VASpace) | 内核管理 |

## 统一内存模型设计

### 问题: 跨厂商内存如何统一?

```rust
// 方案 1: 分离地址空间 (简单, 推荐先做)
pub enum MemoryType {
    Device,    // GPU 本地 VRAM
    Host,      // CPU 系统内存 (pinned)
    Unified,   // 自动迁移 (如果硬件支持)
}

pub struct GpuMemory {
    host_ptr: Option<*mut u8>,     // CPU 可见地址
    device_addr: u64,              // GPU 虚拟地址
    size: usize,
    memory_type: MemoryType,
    owner_device: DeviceId,
}

// 方案 2: 统一虚拟地址 (复杂, 后期做)
pub struct UnifiedMemory {
    unified_ptr: u64,              // CPU 和 GPU 共享的地址
    size: usize,
    // 硬件自动处理迁移 (NVIDIA UVM) 或需要手动管理 (AMD SVM)
}
```

### 推荐策略

```
Phase 1: 分离地址空间
  - 每个设备有独立的 VRAM 堆
  - 显式 memcpy (Host→Device, Device→Host, Device→Device)
  - 跨设备传输通过 P2P DMA 或 bounce buffer

Phase 2: 部分统一
  - AMD: 利用 SVM (Shared Virtual Memory)
  - NVIDIA: 利用 UVM
  - 自动检测硬件能力, fallback 到显式传输

Phase 3: 完全统一 (高级)
  - 跨厂商统一地址空间
  - 硬件加速的页面迁移
  - 需要内核驱动支持
```

## 跨设备通信

| 机制 | 带宽 | 延迟 | 支持 |
|------|------|------|------|
| PCIe P2P (Peer-to-Peer) | 32 GB/s (PCIe 4.0 x16) | ~1μs | AMD + NVIDIA |
| NVLink | 900 GB/s (NVLink 4) | ~0.5μs | NVIDIA only |
| xGMI | 136 GB/s (xGMI 3) | ~0.8μs | AMD only |
| Bounce buffer (经 CPU) | ~25 GB/s (DDR5) | ~5μs | 所有 |

## t0-gpu 当前内存管理

```rust
// kfd/mod.rs — 已实现
struct KfdDevice {
    kfd_fd: RawFd,
    drm_fd: RawFd,
    gpu_id: u32,
    // ...
}

impl KfdDevice {
    fn alloc_vram(&self, size: usize) -> Result<GpuBuffer> {
        // 1. ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)
        // 2. ioctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU)
        // 3. mmap(kfd_fd, ...) for CPU access
    }
}
```

**已实现的功能:**
- VRAM 分配 + CPU mmap
- AQL 队列创建
- Doorbell 映射
- 多 GPU 支持 (gpu_id)

**缺少的功能:**
- 跨设备 P2P 内存访问
- GTT (系统内存) 映射
- Scratch (私有段) 管理
- 内存迁移 (如果做统一地址空间)
- 内存统计/监控
