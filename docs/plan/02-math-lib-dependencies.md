# 数学库依赖链分析

> 分析范围: /data/ROCm/ 下所有数学库

## ROCm 数学库全景

### 线性代数 (BLAS/LAPACK/Solver)
| 库 | 说明 | 对标 |
|----|------|------|
| rocBLAS | BLAS 1/2/3 | cuBLAS |
| hipBLAS | HIP 封装的 BLAS | - |
| hipBLASLt | 轻量级 BLAS (矩阵乘法加速) | - |
| rocSOLVER | LAPACK 级求解器 | cuSOLVER |
| rocALUTION | 稀疏线性求解器 | - |
| Tensile | 高性能 GEMM 内核生成器 | - |

### 稀疏计算
| 库 | 说明 | 对标 |
|----|------|------|
| rocSPARSE | 稀疏矩阵运算 | cuSPARSE |
| hipSPARSE | HIP 封装的稀疏库 | - |
| hipSPARSELt | 稀疏矩阵加速 (推理优化) | - |

### FFT / 随机数
| 库 | 说明 | 对标 |
|----|------|------|
| rocFFT | 快速傅里叶变换 | cuFFT |
| rocRAND | GPU 随机数生成 | cuRAND |

### 张量 / 矩阵加速
| 库 | 说明 | 对标 |
|----|------|------|
| composable_kernel | 可组合内核库 | CUTLASS |
| rocWMMA | Warp 级矩阵乘累加 | WMMA |

### 并行原语
| 库 | 说明 | 对标 |
|----|------|------|
| rocPRIM | 并行原语 (scan/sort/reduce) | CUB |
| hipCUB | HIP 封装的 CUB | CUB |
| rocThrust | 并行算法库 | Thrust |

### 深度学习 / 推理
| 库 | 说明 | 对标 |
|----|------|------|
| AMDMIGraphX | ONNX 推理加速器 | TensorRT |
| flash-attention | Flash Attention (CK tile 版) | - |
| TransformerEngine | Transformer 加速 (FP8) | - |
| vllm | LLM serving 引擎 | - |

### 通信 / 互连
| 库 | 说明 | 对标 |
|----|------|------|
| rccl | 多 GPU 集合通信 | NCCL |

### 数学基础
| 库 | 说明 | 对标 |
|----|------|------|
| half | 半精度浮点库 (CPU 端 f16) | - |

## FP4/MXFP4 支持情况

| 库 | 状态 |
|----|------|
| composable_kernel | ✅ 完整支持 (GFX950 MX pipeline, pk_fp4_t 类型) |
| hipBLASLt | ✅ gfx1200 已有 MXFP4 GEMM 内核 (E2M1) |
| libhipcxx | ✅ 基础类型支持 (__nv_fp4_e2m1) |

FP4 格式:
- E2M1 (OCP MXFP4): 1+2+1, ~0.5 位精度, 范围 ±3
- E3M0: 1+3+0, 仅整数幂, 范围 ±8

## t0-gpu 已有的数学功能

| 功能 | 位置 | 状态 |
|------|------|------|
| GEMM (bf16) | ignis/ops/bf16_matmul.rs + t0/auto_gemm.rs | ✅ 超越 rocBLAS |
| GEMM autotune | ignis/ops/gemm_autotune.rs | ✅ JIT 调优 |
| 融合 GEMM+RMSNorm | ignis/ops/fused_rmsnorm_gemm.rs | ✅ |
| Flash Attention | ignis/ops/ocpa_attention.rs + t0/flash_attn.rs | ✅ |
| Softmax | ignis/ops/shape_ops.rs + t0/softmax_kernels.rs | ✅ |
| RMSNorm | ignis/ops/rmsnorm.rs + t0/rmsnorm_kernels.rs | ✅ |
| Cross Entropy | ignis/ops/cross_entropy.rs + t0/ce_loss_kernels.rs | ✅ |
| Embedding | ignis/ops/embedding.rs + t0/embedding_kernels.rs | ✅ |
| SiLU / PSI 激活 | ignis/ops/silu.rs + psi_activation.rs | ✅ |
| Elementwise | t0/elementwise_kernels.rs | ✅ |
| RoPE | t0/rope_kernels.rs | ✅ |
| AdamW | t0/adamw_kernels.rs | ✅ |
| Reduce (add/max) | t0/tile_ssa.rs | ✅ |

## t0-gpu 缺失的数学原语

| 缺失 | 对标 | 影响 | 优先级 |
|------|------|------|--------|
| GEMV (矩阵×向量) | rocBLAS | Transformer decode 阶段 (seq_len=1) 浪费算力 | P0 |
| Batched GEMM | rocBLAS | 多 head 并行、batch 推理 | P0 |
| Prefix Scan | rocPRIM | BatchNorm、TopK、稀疏索引 | P0 |
| Radix Sort | rocPRIM | TopK 采样、稀疏 attention 排序 | P0 |
| 随机数 (PRNG) | rocRAND | Dropout、权重初始化、训练数据 shuffle | P0 |
| Half/BF16 算术 (host 端) | half 库 | host 端 f16 类型支持 | P0 |
| FFT | rocFFT | 信号处理、频域 positional encoding | P1 |
| TRSM / Cholesky / QR | rocSOLVER | 科学计算、某些正则化 | P1 |
| SpMV / SpMM | rocSPARSE | 稀疏模型、MoE 路由 | P2 |

## 建议的独立数学库结构

```
t0-math/  ← 独立数学库 crate (与 ignis 分离)
├── blas/
│   ├── gemm.rs      ← 已有
│   ├── gemv.rs      ← 需要补
│   ├── batched_gemm.rs  ← 需要补
│   └── trsm.rs      ← 需要补
├── sparse/
│   ├── spmv.rs      ← 需要补
│   └── spmm.rs      ← 需要补
├── rng/
│   ├── philox.rs    ← 需要补 (ML 标准 PRNG)
│   └── distributions.rs ← 需要补 (normal, uniform, bernoulli)
├── scan/
│   ├── prefix_sum.rs ← 需要补
│   └── radix_sort.rs ← 需要补
├── fft/
│   └── radix2.rs    ← 需要补
└── reduce/
    ├── sum.rs       ← 从 tile_ssa 提取
    ├── max.rs       ← 从 tile_ssa 提取
    └── histogram.rs ← 需要补
```

## 最小可行路线

1. **half (P0)** — 最简单, 纯 CPU 类型定义, 1-2 天
2. **rocPRIM 的 scan + reduce** — softmax/layernorm 必须, 2-3 周
3. **rocRAND** — dropout/初始化必须, 1 周
4. **完善 GEMV + batched GEMM** — 补全现有 GEMM, 1-2 周
