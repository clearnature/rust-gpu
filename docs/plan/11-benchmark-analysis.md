# 量化性能基准分析

> 数据来源: t0-gpu README.md, benchmarks/, README 声明的基准测试

## t0-gpu 性能数据 (来自项目 README)

### 推理性能

| 测试 | t0-gpu | 对比 | 提升 |
|------|--------|------|------|
| GEMM (bf16, 大矩阵) | 超越 rocBLAS | rocBLAS | >100% |
| 重复调度 (缓冲) | 2.3μs | HIP ~30μs | **13x** |
| 重复调度 (无缓冲) | 5.5μs | HIP ~6.5μs | **18%** |
| Paged decoding (500) | **151 tok/s** | 54 tok/s | **2.8x** |
| Paged decoding (2000) | **143 tok/s** | 50 tok/s | **2.9x** |
| Paged decoding (10000) | **105 tok/s** | 46 tok/s | **2.3x** |

### 训练性能 (B=8 端到端训练)

| 测试 | t0-gpu | 对比 | 提升 |
|------|--------|------|------|
| 完整训练循环 | **14.2ms** | HIP 22.3ms | **36%** |
| 训练 + 2x tiled RMSNorm | 20.6ms | HIP 26.9ms | **23%** |

### 完整 LLM 性能

| 测试 | t0-gpu | 对比 | 提升 |
|------|--------|------|------|
| Prefill (128→32 tokens) | **38.0ms** | 43.7ms | **13%** |
| Decode (1→1 tokens) | **5.8ms** | 6.5ms | **11%** |
| Decode (1→32 tokens) | **39.7ms** | 52.4ms | **24%** |
| Decode (1→64 tokens) | **73.8ms** | 99.6ms | **26%** |
| Decode (1→128 tokens) | **138.7ms** | 187.7ms | **26%** |
| Decode (1→256 tokens) | **267.9ms** | 367.8ms | **27%** |

### Llama 3 8B 解码

| 测试 | t0-gpu | 说明 |
|------|--------|------|
| Prefill 128 tokens | 394ms | +26ms 内核编译 |
| Decode 1 token | 10.7ms | 93 tok/s |

## tinygrad 性能参考

### tinygrad 在 AMD GPU 上的性能 (来自项目文档)

tinygrad 的 BEAM search 优化通常能达到 vendor 库的 80-95% 性能. 具体数据因硬件而异.

### tinygrad 的编译开销

| 优化级别 | 编译时间 | 说明 |
|---------|---------|------|
| 手写优化 | ~1ms | hand_coded_optimizations |
| BEAM=1 | ~100ms | 浅层搜索 |
| BEAM=3 | ~500ms | 中等搜索 |
| BEAM=10 | ~5s | 深层搜索 |

BEAM 搜索结果有 disk cache, 首次编译慢, 后续从缓存加载.

## sass-assembler 性能

| 测试 | 结果 |
|------|------|
| 100k 条指令编码 | <5s (通过) |
| 10k 条指令 roundtrip | <1s (通过) |
| 单条指令编码 | ~50ns |

## 性能分析: 为什么 t0-gpu 能超越 vendor 库

### 1. 零中间层

```
ROCm:   hipLaunchKernel → HIP → CLR → ROCr → libhsakmt → KFD (5 层, ~20μs)
t0-gpu: dispatch → KFD (1 层, ~2μs)
```

节省 ~18μs per dispatch. 对 decode (每 token 1 次 dispatch) 影响巨大.

### 2. JIT GEMM 针对性优化

t0-gpu 的 GEMM JIT 针对**每个矩阵形状**生成最优 kernel:
- rocBLAS 使用预编译的通用 kernel (tile=128x128, 不一定是最优)
- t0-gpu 运行时分析矩阵形状, 选择最优 tile 配置

### 3. 融合 kernel

```
rocBLAS 方式:
  RMSNorm kernel (dispatch 1)
  → GEMM kernel (dispatch 2)
  → SiLU kernel (dispatch 3)
  → GEMM kernel (dispatch 4)
  = 4 次 dispatch, 4 次 PCIe 往返

t0-gpu 方式:
  fused_rmsnorm_gemm kernel (dispatch 1)
  → fused_silu_gemm kernel (dispatch 2)
  = 2 次 dispatch, 2 次 PCIe 往返
```

### 4. Buffer pool 减少分配开销

```rust
// BufferPool 使用 2^n 桶缓存
// 避免每次 kernel 执行都调用 KFD alloc/free ioctl
// 首次分配后, 后续复用缓存的 buffer
```

### 5. BF16 计算

t0-gpu 的 GEMM 使用 BF16 (16-bit), 而不是 F32 (32-bit):
- 内存带宽减半
- 计算吞吐翻倍 (BF16 SIMD)
- 与 F32 累加保持精度

## 性能瓶颈分析

### 当前瓶颈

| 瓶颈 | 影响 | 严重度 |
|------|------|--------|
| 逐 head OCPA 分发 | Attention 无法 batch 所有 head | 中 |
| 梯度裁剪 GPU→CPU 往返 | 训练每步 1 次 PCIe 往返 | 中 |
| 无 KV cache | Decode 时重复计算 K,V | 高 |
| 缺少 GEMV kernel | Decode (M=1) 用小 GEMM 浪费 | 中 |
| 缺少 scan/sort | TopK 无法在 GPU 完成 | 低 (推理时) |
| 缺少随机数 | Dropout 无法在 GPU 完成 | 低 (推理时) |

### 性能优化建议

| 优先级 | 优化 | 预期提升 |
|--------|------|---------|
| P0 | KV cache 实现 | decode 2-5x |
| P0 | GEMV kernel (M=1 专用) | decode 1.5-2x |
| P1 | OCPA batch heads | attention 1.5-2x |
| P1 | GPU-side reduce (替代梯度裁剪 CPU 往返) | 训练 5-10% |
| P2 | Software pipelining (隐藏内存延迟) | GEMM 10-20% |
| P2 | Memory coalescing 优化 | 所有 kernel 5-15% |
| P3 | BEAM search autotune | 最优配置自动发现 |

## 与 vendor 库的性能差距

| 操作 | t0-gpu vs rocBLAS | 差距原因 |
|------|-------------------|---------|
| 大矩阵 GEMM | **超越** | JIT 针对性优化 |
| 小矩阵 GEMM | **超越** | 零中间层 |
| GEMV | **落后** (缺少专用 kernel) | 用小 GEMM 代替 |
| Batched GEMM | **落后** (缺少) | 需要实现 |
| Flash Attention | **相当** | 标准算法 |
| Softmax | **超越** | 融合 kernel |
| RMSNorm | **超越** | 融合 GEMM |
