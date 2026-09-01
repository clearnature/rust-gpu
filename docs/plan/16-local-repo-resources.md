# 本地仓库资源分析 — 对通用 GPU 运行时有帮助的项目

> 扫描范围: /home/yanli/work/ + /data/ + /data/training/cli/
> 扫描日期: 2026-08-24

## 直接可用 (可移植代码)

### 1. cubecl — Rust GPU 计算框架 ⭐⭐⭐⭐⭐

```
路径: /home/yanli/work/9060xt/cubecl/
语言: Rust
代码量: 159,185 行 (788 个 .rs 文件)
```

**这是最大的发现。** cubecl 是 Burn 深度学习框架的 GPU 后端, 已经实现了**多厂商 GPU 运行时**:

| crate | 功能 | 对通用运行时的价值 |
|-------|------|-----------------|
| `cubecl-ir` | 统一 IR (20 个模块) | **直接复用** — 已有完整的 SSA IR + 类型系统 |
| `cubecl-hip` | AMD HIP 后端 | **参考** — Rust 实现的 HIP 运行时 |
| `cubecl-cuda` | NVIDIA CUDA 后端 | **参考** — Rust 实现的 CUDA 运行时 |
| `cubecl-llvm` | LLVM 后端 | **直接复用** — Rust LLVM 绑定 |
| `cubecl-spirv` | SPIR-V 后端 | **参考** — Vulkan compute |
| `cubecl-metal` | Apple Metal 后端 | **参考** — 多厂商模式 |
| `cubecl-cpu` | CPU 后端 | **直接复用** — CPU fallback |
| `cubecl-opt` | 优化 pass | **参考** — 编译器优化 |
| `cubecl-runtime` | 运行时抽象 | **直接复用** — Device/Queue/Memory trait |
| `cubecl-core` | 核心 API | **参考** — 用户 API 设计 |
| `cubecl-wgpu` | WebGPU 后端 | 低优先级 |
| `cubecl-common` | 公共工具 | **直接复用** |

**关键发现**: cubecl 已经有了 `trait Runtime` / `trait Device` / `trait Queue` 的多厂商抽象! 它的架构和我们设计的 15-universal-runtime-framework.md 高度相似。**应该优先研究它的 trait 设计。**

### 2. libkfd — KFD 用户态库 ⭐⭐⭐⭐

```
路径: /home/yanli/work/9060xt/libkfd/
语言: C++
代码量: ~2000 行
```

**独立的 KFD 用户态库**, 和 t0-gpu 的 kfd/mod.rs 做同样的事, 但是 C++ 实现:

| 文件 | 功能 |
|------|------|
| `lib/device.cpp` | GPU 设备管理 |
| `lib/memory.cpp` | VRAM 分配 |
| `lib/queue.cpp` | AQL 队列 |
| `lib/signal.cpp` | 信号量 |
| `lib/event.cpp` | 事件管理 |
| `lib/topology.cpp` | GPU 拓扑发现 |
| `lib/ioctl.h` | KFD ioctl 定义 |
| `lib/trap_handler.c` | GPU trap 处理 |

**价值**: 可以作为 t0-gpu KFD 运行时的 C++ 参考实现, 或者直接用 C FFI 调用。

### 3. aiter — AMD AI 算子库 ⭐⭐⭐⭐

```
路径: /home/yanli/work/9060xt/aiter/
语言: Python + C++ (HIP)
代码量: 大型
```

**AMD 官方 AI 算子集合**, 包含:

| 模块 | 功能 | 价值 |
|------|------|------|
| `aiter/ops/` | 高性能算子 | GEMM/Attention/MoE 内核参考 |
| `aiter/jit/` | JIT 编译 | 运行时 kernel 编译参考 |
| `aiter/aot/` | AOT 编译 | 预编译 kernel |
| `aiter/configs/` | 调优配置 | 各 GPU 型号的最优配置 |
| `fused_moe.py` | MoE 融合 | 稀疏 MoE kernel |
| `paged_attn.py` | Paged Attention | **KV cache 实现参考** |
| `mla.py` | MLA | Multi-head Latent Attention |
| `tuned_gemm.py` | 调优 GEMM | **GEMM autotune 参考** |
| `rotary_embedding.py` | RoPE | RoPE kernel |
| `bert_padding.py` | BERT padding | 动态 padding |

**关键价值**: `paged_attn.py` 和 `tuned_gemm.py` 是 t0-gpu 缺少的 KV cache 和 GEMM autotune 的直接参考。

### 4. rdna3-bare-metal — RDNA3 裸金属运行时 ⭐⭐⭐

```
路径: /home/yanli/work/9060xt/rdna3-bare-metal/
语言: Rust
代码量: ~小型 (4 个 .rs 文件)
```

**t0-gpu 的前身/简化版**:
- `asm.rs` — ISA 汇编
- `code_object.rs` — HSA ELF 生成
- `runtime.rs` — KFD 运行时

**价值**: 代码更简洁, 可以作为 t0-gpu 的教学参考或快速原型。

### 5. sass-assembler — 多架构 SASS 汇编器 ⭐⭐⭐⭐

```
路径: /data/rtl-sdr/sass-assembler/
语言: C++
代码量: ~4000 行 (核心)
```

**已分析过** — Pascal/Volta/Ampere 真实 SASS 编码器 + ILP 模型 + 流形调度器。

### 6. swmmac — RDNA4 SWMMAC 测试 ⭐⭐⭐

```
路径: /data/rtl-sdr/swmmac/
语言: ASM + Python
```

**RDNA4 SWMMAC/WMMA 指令的实际编码参考**, 含:
- `asm/` — GFX1200 汇编实例
- `bench_v1/` — 性能基准
- `calibration/` — 硬件校准数据

### 7. cpu_probe — CPU 微架构探测 ⭐⭐⭐

```
路径: /data/rtl-sdr/cpu_probe/
语言: C/ASM
```

**CPU 延迟/吞吐量实测数据**, 已用于 ILP 模型的 Broadwell/Zen4 延迟参数。

## 间接相关 (参考价值)

### 8. llamacpp-rocm — llama.cpp ROCm 移植 ⭐⭐

```
路径: /home/yanli/work/9060xt/llamacpp-rocm/
```

ROCm 版 llama.cpp, 可以参考其:
- ROCm kernel 调用模式
- KV cache 实现
- 量化 kernel

### 9. rdna4-container-virt — RDNA4 容器虚拟化 ⭐⭐⭐

```
路径: /home/yanli/work/9060xt/rdna4-container-virt/
```

**5 个 KFD 内核补丁** 实现多租户 GPU 隔离:
- VRAM cap / evict / tenant policy / CU confine
- Kubernetes DaemonSet + Prometheus 指标
- dma-buf 零拷贝共享
→ **共享 GPU 功能的直接参考**

### 10. scholar-loop — 自主 ML 研究 ⭐⭐

```
路径: /data/training/cli/scholar-loop/
```

### 11. BiSheng-Autotuner — 毕昇编译器自动调优 ⭐⭐

```
路径: /data/work/compiler/BiSheng-Autotuner/
```

华为毕昇编译器自动调优, 对**昇腾后端**有参考价值。

### 12. ptx_gp106 — PTX/SASS 参考数据 ⭐⭐

```
路径: /data/rtl-sdr/ptx_gp106/
```

### 13. ROCm 源码树 ⭐⭐⭐

```
路径: /data/ROCm/rocm-systems/
```

## 子代理深度扫描发现的额外仓库

### 14. DeepGEMM — DeepSeek 统一 GEMM 库 ⭐⭐⭐⭐

```
路径: /data/trit/deepseek/DeepGEMM/
```

**1550 TFLOPS on H800**. FP8/FP4/BF16, 融合 MoE, JIT 编译.
→ **最先进的 GEMM kernel 设计参考**

### 15. FlashMLA — DeepSeek 优化 Attention ⭐⭐⭐

```
路径: /data/trit/deepseek/FlashMLA/
```

**660 TFLOPS on H800**. 稀疏 + 稠密, prefill + decode.

### 16. CUTLASS — NVIDIA GEMM 模板库 ⭐⭐⭐

```
路径: /data/trit/cutlass/
```

**NVIDIA 官方 GEMM 参考实现**. CUTLASS 4.7.0 + CuTe DSL. Volta→Blackwell.

### 17. triton — Triton GPU 编译器 ⭐⭐⭐

```
路径: /data/trit/triton/
```

**高层 GPU kernel DSL 编译器**, NVIDIA + AMD 后端.
→ DSL→GPU ISA 编译器的参考

### 18. huntian-llvm — 浑天 LLVM 插件 ⭐⭐⭐

```
路径: /data/trit/浑天/huntian-llvm/
```

**LLVM 编译器插件** for 浑天 4320D 架构. V-AVX3 指令定义, 量子时钟相位追踪.

### 19. ternary-core — 三值计算 ISA ⭐⭐⭐

```
路径: /data/trit/浑天/ternary-core/
```

**83 条主权指令** (vavx3_api.h), 无乘法器三值点积, 4320D 涡旋映射.

### 20. taiji-ternary-isa — 三值 ISA 规范 ⭐⭐⭐

```
路径: /data/trit/taiji/taiji-ternary-isa/
```

**完整三值计算 ISA 规范** (中文). 微架构、汇编器、链接器、内存布局、总线/外设接口、模拟器规范.

### 21. taiji-neural-core — 三值推理引擎 ⭐⭐

```
路径: /data/trit/taiji/taiji-neural-core/
```

三值量化推理引擎. 1.58-bit 权重的 GEMM (标量 + AVX2).

### 22. DeepEP — MoE 通信库 ⭐⭐

```
路径: /data/trit/deepseek/DeepEP/ (NVIDIA)
路径: /data/ROCm/DeepEP/ (AMD)
```

MoE all-to-all 通信, FP8, 零 SM RDMA, EP2048.

### 23. FlashInfer ROCm — AMD Attention 内核 ⭐⭐

```
路径: /data/ROCm/flashinfer/
```

AMD ROCm 的 FlashInfer 移植. Attention/KV-cache/RoPE/归一化/采样内核.

### 24. oneDNN — Intel DNN 原语库 ⭐⭐

```
路径: /data/trit/oneDNN/
```

Intel 跨平台 DNN 库. 实验性 NVIDIA/AMD GPU 支持.

### 25. TileKernels — DeepSeek 优化内核 ⭐⭐

``路径: /data/trit/deepseek/TileKernels/``

MoE 路由、量化 (FP8/FP4/E5M6)、转置、Engram 优化内核.

## 最终优先级排序

```
P0 (立刻研究, 直接可用):
  1. cubecl          — Rust 多厂商 GPU 运行时 (159K 行)
  2. libkfd          — C++ KFD 库 (完整 API, 可 FFI)
  3. aiter           — AMD AI 算子 (paged_attn + tuned_gemm)
  4. tinygrad        — Python 多厂商运行时 (HCQ 抽象)

P1 (近期参考):
  5. sass-assembler  — NVIDIA SASS 编码器
  6. DeepGEMM        — 最先进 GEMM (1550T on H800)
  7. CUTLASS         — NVIDIA GEMM 模板
  8. rdna4-container-virt — 共享 GPU KFD 补丁

P2 (架构参考):
  9. triton          — DSL→GPU ISA 编译器
  10. huntian-llvm   — 浑天 LLVM 插件
  11. ternary-core   — 三值计算 ISA

P3 (按需):
  12. FlashMLA       — DeepSeek Attention
  13. DeepEP         — MoE 通信
  14. oneDNN         — Intel DNN 库
```
