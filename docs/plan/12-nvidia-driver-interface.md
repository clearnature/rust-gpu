# NVIDIA 内核驱动接口分析

> 数据来源: tinygrad/runtime/ops_nv.py + autogen/nv_570.py (24867 行绑定)

## 设备文件

| 设备文件 | 用途 |
|---------|------|
| `/dev/nvidiactl` | 控制通道 — 所有 RM (Resource Manager) ioctl 经过此 |
| `/dev/nvidia-uvm` | 统一虚拟内存 — 内存映射、页表、P2P 访问 |
| `/dev/nvidia<N>` | 每 GPU 设备文件 — 用于 CPU 内存映射 |

## 核心 Ioctl 命令

### 控制通道 (on /dev/nvidiactl)

| Ioctl | 值 | 结构体 | 用途 |
|-------|-----|--------|------|
| NV_ESC_CARD_INFO | 200 | nv_ioctl_card_info_t[64] | 枚举 GPU, 获取 minor 号 |
| NV_ESC_REGISTER_FD | 201 | nv_ioctl_register_fd_t | 注册 per-GPU fd |
| NV_ESC_RM_ALLOC | 0x2B | NVOS21_PARAMETERS | 分配 RM 对象 (设备/通道/VA空间等) |
| NV_ESC_RM_CONTROL | 0x2A | NVOS54_PARAMETERS | RM 控制命令 (通用分发) |
| NV_ESC_RM_FREE | 0x29 | NVOS00_PARAMETERS | 释放 RM 对象 |
| NV_ESC_RM_MAP_MEMORY | 0x4E | NVOS33_PARAMETERS | 映射 GPU 内存到 CPU 地址空间 |
| NV_ESC_RM_MAP_MEMORY_DMA | 0x57 | NVOS46_PARAMETERS | 映射内存到 GPU DMA 虚拟地址 |
| NV_ESC_RM_ALLOC_MEMORY | 0x27 | NVOS02_PARAMETERS | 分配 OS 后端内存 (host) |

### UVM (on /dev/nvidia-uvm)

| Ioctl | 值 | 用途 |
|-------|-----|------|
| UVM_INITIALIZE | 39 | 初始化 UVM 子系统 |
| UVM_REGISTER_GPU | 37 | 注册 GPU (通过 UUID) |
| UVM_REGISTER_GPU_VASPACE | 25 | 注册 VA 空间 |
| UVM_REGISTER_CHANNEL | 27 | 注册 GPFIFO 通道 |
| UVM_ENABLE_PEER_ACCESS | 29 | 启用 P2P 访问 |
| UVM_CREATE_EXTERNAL_RANGE | 73 | 创建外部管理内存的 VA 范围 |
| UVM_MAP_EXTERNAL_ALLOCATION | 33 | 映射 RM 内存分配到 UVM |
| UVM_FREE | 34 | 释放 UVM 映射 |

## RM 对象层次

```
NV01_ROOT_CLIENT (0x41) — 根客户端, 每进程一个
 |
 +-- NV01_DEVICE_0 (0x80) — 物理设备
 |   |
 |   +-- NV20_SUBDEVICE_0 (0x2080) — 子设备 (GPU 查询)
 |   +-- NV01_MEMORY_VIRTUAL (0x70) — 虚拟内存分配
 |   +-- FERMI_VASPACE_A (0x90f1) — 带 fault 模型的 VA 空间
 |   +-- KEPLER_CHANNEL_GROUP_A (0xa06c) — 通道组
 |   |
 |   +-- FERMI_CONTEXT_SHARE_A (0x9067) — 上下文共享
 |   +-- GPFIFO 类 — AMPERE_CHANNEL_GPFIFO_A (0xc56f)
 |   |                  或 BLACKWELL_CHANNEL_GPFIFO_A (0xc96f)
 |   |
 |   +-- Compute 类 — AMPERE_COMPUTE_B (0xc7c0)
 |   |                  ADA_COMPUTE_A (0xc9c0)
 |   |                  BLACKWELL_COMPUTE_B (0xcec0)
 |   |
 |   +-- DMA copy 类 — AMPERE_DMA_COPY_B (0xc7b5)
 |                      BLACKWELL_DMA_COPY_B (0xcab5)
```

## 内存分配流程

### GPU VRAM

1. 计算页大小 (>=8MB 用 2MB 大页, 否则 4KB)
2. 通过 bump allocator 分配 GPU VA 地址
3. `rm_alloc(NV1_MEMORY_USER, NV_MEMORY_ALLOCATION_PARAMS)` — 分配物理内存
4. CPU 映射: `NV_ESC_RM_MAP_MEMORY` + mmap(per-GPU fd)
5. GPU VA 映射: `UVM_CREATE_EXTERNAL_RANGE` + `NV_ESC_RM_MAP_MEMORY_DMA` + `UVM_MAP_EXTERNAL_ALLOCATION`

### Host 系统内存

1. mmap 匿名共享内存
2. `NV_ESC_RM_ALLOC_MEMORY` with class `NV01_MEMORY_SYSTEM_OS_DESCRIPTOR`
3. UVM 映射 (同上)

## GPFIFO — NVIDIA 的 AQL 等价物

GPFIFO (Graphics PFIFO) 是 NVIDIA 的命令提交机制, 等价于 KFD 的 AQL 队列.

### 环形缓冲区条目格式 (64-bit)

```
bits [41:0] = cmdq_addr >> 2  (命令缓冲区地址, 右移 2 位)
bit  [41]   = 1               (同步标志)
bits [63:42] = length          (命令缓冲区长度, 32-bit words)
```

### 提交流程

1. 写 ring entry: `gpfifo.ring[put % entries] = (addr/4 << 2) | (len << 42) | (1 << 41)`
2. 更新 GPPut: `gpfifo.gpput[0] = (put + 1) % entries`
3. 内存屏障
4. **Doorbell 写入**: `dev.gpu_mmio[0x90 // 4] = gpfifo.token`

等价于 KFD 的: 写 AQL packet → 更新 write pointer → 写 doorbell

### Userd 控制区布局

| 偏移 | 字段 | 大小 | 用途 |
|------|------|------|------|
| 64 | Put | u32 | 软件写指针 (CPU 写) |
| 68 | Get | u32 | 硬件读指针 (GPU 更新) |
| 72 | Reference | u32 | Fence 参考值 |
| 136 | GPGet | u32 | GPU 端 get 指针 |
| 140 | GPPut | u32 | CPU 写此位置推进环 |

## QMD — NVIDIA 的 AQL Packet 等价物

QMD (Queue Meta Data) 是 NVIDIA 的 kernel dispatch 描述符:

- **QMD v3** (pre-Blackwell): 256 字节 (0x40 dwords)
- **QMD v5** (Blackwell): 384 字节 (0x60 dwords)

关键字段:
- `program_address_upper/lower` — SASS shader 的 GPU VA
- `register_count` — 每线程 GPR 数
- `shared_memory_size` — 每 CTA 的 shared memory
- `cta_raster_width/height/depth` — grid 维度
- `cta_thread_dimension0/1/2` — block 维度
- `constant_buffer_addr_0` — kernel 参数指针
- `release0/1_enable/address/payload` — 嵌入式信号量

### Dispatch 流程

1. 复制 QMD 到 GPU 可访问缓冲区
2. 设置 grid/block 维度, 常量缓冲区地址
3. 首次 dispatch: 写 QMD 地址到 PCAS:
   ```python
   nvm(1, NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
   nvm(1, NVC6C0_SEND_SIGNALING_PCAS2_B, 9)  # action=9 = schedule
   ```
4. 后续 dispatch: 通过 QMD 的 `dependent_qmd0_pointer` 链式连接

## 初始化序列

```
1. NV_ESC_CARD_INFO — 枚举 GPU
2. NV_ESC_REGISTER_FD — 注册 per-GPU fd
3. rm_alloc(NV01_ROOT_CLIENT) — 创建根客户端
4. UVM_INITIALIZE + UVM_MM_INITIALIZE
5. Per-GPU:
   5a. rm_alloc(NV01_DEVICE_0) — 分配设备
   5b. rm_alloc(NV20_SUBDEVICE_0) — 分配子设备
   5c. rm_alloc(NV01_MEMORY_VIRTUAL) — VA 范围
   5d. rm_alloc(FERMI_VASPACE_A) — fault VA 空间
   5e. UVM_REGISTER_GPU + UVM_REGISTER_GPU_VASPACE
   5f. setup_usermode() — 分配 usermode BAR
   5g. rm_alloc(KEPLER_CHANNEL_GROUP_A) — 通道组
   5h. rm_alloc(GPFIFO_CLASS) — GPFIFO 通道
   5i. rm_alloc(compute_class) — 计算引擎对象
   5j. rm_alloc(dma_class) — DMA 引擎对象
   5k. UVM_REGISTER_CHANNEL — 注册到 UVM
   5l. GET_WORK_SUBMIT_TOKEN — 获取 doorbell token
   5m. 设置 shared/local memory 窗口
   5n. GPFIFO_SCHEDULE — 启用调度
```

## 与 KFD 对比

| 操作 | AMD KFD | NVIDIA |
|------|---------|--------|
| 设备文件 | /dev/kfd | /dev/nvidiactl + /dev/nvidia-uvm + /dev/nvidia<N> |
| 内存分配 | 1 次 ioctl (ALLOC_MEMORY_OF_GPU) | 多次: RM alloc + UVM map + DMA map |
| GPU 映射 | 1 次 ioctl (MAP_MEMORY_TO_GPU) | UVM 自动管理 |
| 队列创建 | 1 次 ioctl (CREATE_QUEUE) | 多次: RM alloc channel group + GPFIFO + compute |
| 命令格式 | AQL packet (64 bytes, 固定格式) | GPFIFO ring entry + QMD (256-384 bytes) |
| Doorbell | mmap doorbell page, 直接写 | MMIO write to BAR offset 0x90 |
| 信号量 | hsa_signal_t (独立对象) | NVC56F_SEM (in-band) 或 QMD embedded |
| 复杂度 | 简单 (1-2 次 ioctl) | 复杂 (10+ 次 RM alloc/ctrl) |

## 对 Rust NvidiaDriver 实现的建议

```rust
// 核心结构
struct NvidiaDriver {
    ctl_fd: RawFd,      // /dev/nvidiactl
    uvm_fd: RawFd,      // /dev/nvidia-uvm
    dev_fd: RawFd,      // /dev/nvidia<N>
    root_client: Handle,
    device: Handle,
    subdevice: Handle,
    vaspace: Handle,
    channel_group: Handle,
    gpfifo_channel: Handle,
    compute_engine: Handle,
    dma_engine: Handle,
    doorbell_token: u32,
    gpfifo_ring: MMIOInterface,
    gpfifo_put: usize,
}
```

**关键参考文件:**
- tinygrad/runtime/ops_nv.py — 完整实现 (~1100 行)
- tinygrad/runtime/autogen/nv_570.py — 24867 行自动生成绑定
