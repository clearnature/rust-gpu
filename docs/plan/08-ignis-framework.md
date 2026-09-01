# ignis 框架深度分析

> 审计目标: /home/yanli/work/9060xt/t0-gpu/src/ignis/
> 总代码量: 6590 行 Rust

## 架构概览

```
ignis/
├── 核心层 (1556 行)
│   ├── tensor.rs      (462) ← GPU-backed tensor + autodiff
│   ├── tape.rs        (446) ← 反向模式自动微分
│   ├── gpu_context.rs (585) ← GPU 运行时封装
│   └── mod.rs          (30) ← 模块导出
│
├── 运算层 (2807 行)
│   ├── ops/bf16_matmul.rs     (424) ← BF16 GEMM
│   ├── ops/add.rs             (489) ← 向量加法 (含多形状支持)
│   ├── ops/ocpa_attention.rs  (589) ← OCPA 前向+反向
│   ├── ops/shape_ops.rs       (350) ← softmax/transpose/reshape
│   ├── ops/rmsnorm.rs         (317) ← RMSNorm
│   ├── ops/fusion.rs          (234) ← kernel 融合优化
│   ├── ops/silu.rs            (184) ← SiLU 激活
│   ├── ops/embedding.rs       (186) ← Embedding lookup
│   ├── ops/cross_entropy.rs   (127) ← Cross entropy loss
│   ├── ops/fused_rmsnorm_gemm.rs (154) ← 融合 RMSNorm+GEMM
│   ├── ops/psi_activation.rs   (59) ← PSI 激活函数
│   └── ops/gemm_autotune.rs    (30) ← GEMM 自动调优
│
├── 神经网络层 (552 行)
│   ├── nn/transformer.rs  (183) ← Transformer layer (OCPA + FFN)
│   ├── nn/model.rs        (108) ← 模型基类 (Module trait)
│   ├── nn/linear.rs        (86) ← Linear layer
│   ├── nn/embedding.rs    (142) ← Embedding layer
│   └── nn/mod.rs           (33) ← 模块导出
│
└── 训练支撑层 (974 行)
    ├── tests.rs          (701) ← 测试
    ├── tokenizer.rs      (171) ← BPE tokenizer
    ├── data_loader.rs     (99) ← 数据加载
    ├── loss_scaler.rs    (115) ← 动态 loss scaling
    ├── grad_clip.rs       (89) ← 梯度裁剪
    ├── lr_scheduler.rs    (75) ← 学习率调度
    └── buffer_pool.rs     (78) ← GPU buffer 缓存池
```

## 核心设计

### Tensor (tensor.rs, 462 行)

```rust
pub struct Tensor {
    id: TensorId,                    // 唯一 ID (AtomicU64 单调递增)
    buf: Arc<GpuBuffer>,             // VRAM 数据 (引用计数共享)
    runtime: Arc<GpuRuntime>,        // GPU 运行时引用
    shape: Vec<usize>,               // 形状
    dtype: DType,                    // F32/BF16
    label: String,                   // 调试标签
    grad: RefCell<Option<Arc<GpuBuffer>>>,  // 梯度 (惰性分配)
    tape_node: Cell<Option<NodeId>>, // 计算图节点
    requires_grad: bool,             // 是否需要梯度
}
```

**设计特点:**
- Arc<GpuBuffer> 实现零拷贝共享 (类似 PyTorch 的 tensor storage)
- 最小 512 字节分配 (向量化 kernel 需要 dwordx4 对齐)
- 512 字节对齐 (与 KFD 页对齐一致)

### Tape (tape.rs, 446 行)

```rust
// 线程本地计算图 (无锁)
thread_local! {
    static TAPE_NODES: RefCell<Vec<TapeNode>>,
    static TAPE_RECORDING: RefCell<bool>,
    static GRAD_REGISTRY: RefCell<HashMap<TensorId, Arc<GpuBuffer>>>,
}

pub struct TapeNode {
    pub output_id: TensorId,
    pub input_ids: Vec<Option<TensorId>>,
    pub input_requires_grad: Vec<bool>,
    pub saved_tensors: Vec<Arc<GpuBuffer>>,  // 前向保存的激活值
    pub backward_fn: Option<BackwardFn>,      // 反向传播闭包
}
```

**设计特点:**
- 完全模仿 PyTorch autograd
- backward_fn 是 FnOnce 闭包, 消费式调用
- 梯度注册表按 TensorId 查找
- 支持 no_grad 模式 (推理时不记录)

### BufferPool (buffer_pool.rs, 78 行)

```rust
pub struct BufferPool {
    device: Arc<KfdDevice>,
    buckets: HashMap<usize, Vec<GpuBuffer>>,  // 2^n 大小桶
    hits: u64,
    misses: u64,
}
```

**设计特点:**
- 2^n 桶缓存 (最小 4096 字节 = KFD 页大小)
- 减少 KFD alloc/free ioctl 开销
- 零缓存时自动 fallback 到新分配

### LossScaler (loss_scaler.rs, 115 行)

```rust
pub struct LossScaler {
    scale: f32,                // 初始 65536.0
    growth_factor: f32,        // 2.0
    backoff_factor: f32,       // 0.5
    growth_interval: usize,    // 200 步后增长
    consecutive_clean: usize,
    max_scale: f32,            // 2^24
    min_scale: f32,            // 1.0
}
```

**设计特点:**
- 完整的动态 loss scaling (混合精度训练必须)
- 检测 NaN/Inf 后自动 backoff
- 连续 200 步无 NaN 后 scale 翻倍

### LrScheduler (lr_scheduler.rs, 75 行)

```rust
pub struct CosineWarmupScheduler {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}
```

**设计特点:**
- 标准 cosine warmup (线性 warmup + cosine decay)
- 用于 LLM 训练的标准调度器

## OCPA Attention (ocpa_attention.rs, 589 行)

**OCPA = Orthogonal Chunked Pure-Matrix Attention**

这是 t0-gpu 最独特的创新——一种分块注意力算法:

```
Forward 5 步流水线:
  1. State update:     S_c = S_{c-1} + K_c^T @ V_c  (每 chunk)
  2. Prefix sum:       S̃_c = S_0 + S_1 + ... + S_{c-1}
  3. Forward inter:    O_inter = Q_c @ S̃_c
  4. Forward intra:    O_intra = mask(Q_c @ K_c^T) @ V_c
  5. Denom norm:       O = (O_inter + O_intra) / denominator

Backward 4 步流水线:
  1. dState:           dU_c = Q_c^T @ dO_c
  2. Reverse prefix:   dS̃_c = dU_c + dU_{c+1} + ...
  3. Backward inter:   dQ = dO @ S̃^T, dK_inter, dV_inter
  4. Backward intra:   dQ_intra, dK_intra, dV_intra
```

**优势:**
- 状态矩阵 S 是 [d, d] 而非 [seq, seq] — 内存 O(d²) 而非 O(seq²)
- 前向+反向完整实现 (590 行)
- 所有 kernel 用 40-byte kernarg 统一接口

## Transformer Layer (nn/transformer.rs, 183 行)

```rust
pub struct TransformerLayer {
    pub wq, wk, wv, wo: Linear,        // QKV 投影 + 输出投影
    pub w_gate, w_up, w_down: Linear,   // FFN (SwiGLU)
    pub attn_norm_gamma: Tensor,         // RMSNorm
    pub ffn_norm_gamma: Tensor,          // RMSNorm
    // 11 个参数张量
}
```

**完整 Transformer 实现:**
- RMSNorm → QKV 投影 → OCPA → 输出投影 → 残差
- RMSNorm → Gate/Up → SiLU gate → Down → 残差
- decode (M=1) 时使用融合 RMSNorm+GEMV

## 与 tinygrad ignis 对比

| 维度 | t0-gpu ignis | tinygrad |
|------|-------------|----------|
| 语言 | Rust | Python |
| Tensor | Arc<GpuBuffer> (零拷贝) | RawBuffer (lazy) |
| Autograd | Tape (PyTorch 风格) | 惰性计算图 |
| Attention | OCPA (自研, O(d²)) | FlashAttention (标准) |
| 混合精度 | LossScaler + BF16 | BF16/F16 支持 |
| 梯度裁剪 | 全局 L2 norm | 未见专门实现 |
| LR 调度 | CosineWarmup | 外部实现 |
| Buffer 缓存 | 2^n 桶池 | 内部管理 |
| Transformer | 完整 (OCPA + SwiGLU + RMSNorm) | 通过 nn 模块 |
| 测试 | 701 行测试代码 | 大量测试 |

## 生产就绪度评估

| 组件 | 就绪度 | 说明 |
|------|--------|------|
| Tensor | ✅ 生产级 | 完整的 GPU tensor, 零拷贝共享 |
| Autograd | ✅ 生产级 | Tape + backward 闭包, 模仿 PyTorch |
| OCPA | ✅ 可用 | 前向+反向完整, 但分块大小需调优 |
| GEMM | ✅ 生产级 | 超越 rocBLAS |
| Transformer | ✅ 可用 | 完整 layer, 但缺 KV cache |
| LossScaler | ✅ 生产级 | 动态 scaling 完整 |
| Tokenizer | ⚠️ 基础 | BPE tokenizer, 无 SentencePiece |
| DataLoader | ⚠️ 基础 | 简单加载, 无多进程/预取 |
| BufferPool | ✅ 生产级 | 2^n 桶缓存 |
| 梯度裁剪 | ⚠️ 有性能问题 | GPU→CPU→GPU (read_f32 往返) |

## 已知问题

### 1. 梯度裁剪的 GPU→CPU 往返

```rust
// grad_clip.rs — 读回 CPU 做 norm 计算, 再写回 GPU
let grad_data = read_f32(&grad, n);        // GPU → CPU
let norm_sq: f64 = grad_data.iter()...;     // CPU 计算
write_f32(&grad, &grad_data);               // CPU → GPU
```

应该用 GPU reduce kernel 计算 norm, 避免 PCIe 往返.

### 2. OCPA 逐 head 分发

```rust
for head in 0..h {  // 串行分发每个 head
    // 每个 head 独立 dispatch
}
```

应该 batch 所有 head 到一个 kernel dispatch.

### 3. 缺少 KV Cache

Transformer forward_simple 没有 KV cache, decode 时每次都重新计算所有 K,V.
