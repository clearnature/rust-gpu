# t0-gpu 代码分析地图

> 由 GitNexus 知识图谱自动生成 — 索引提交: ca8de73 | 2026-08-25

## 1. 项目概览

| 指标 | 值 |
|------|-----|
| 文件总数 | 262 |
| 源码文件 | 155 (151 Rust + 4 Python) |
| 节点总数 | 7,270 |
| 边总数 | 30,435 |
| 功能聚类 | 233 |
| 执行流程 | 273 |
| 函数 | 3,003 |
| 结构体 | 222 |
| 枚举 | 56 |
| Trait | 20 |
| 模块 | 191 |
| 常量 | 146 |
| Impl 块 | 112 |

## 2. 架构层次 (6 层)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: Examples & Benchmarks (examples/, benchmarks/)│
├─────────────────────────────────────────────────────────┤
│  Layer 4: Ignis Neural Framework (src/ignis/)           │
│   nn/ | ops/ | tensor | tape | tokenizer | lr_scheduler │
├─────────────────────────────────────────────────────────┤
│  Layer 3: T0 Compiler DSL (src/t0/ block_dsl, tile_ir)  │
├─────────────────────────────────────────────────────────┤
│  Layer 2: T0 SSA Backend (src/t0/ ssa_ir, regalloc,     │
│           asm_emitter, compile, opt_passes)              │
├─────────────────────────────────────────────────────────┤
│  Layer 1: ISA Encoding (src/rdna3_asm, rdna3_code_object│
│           wmma_db, t0/ir)                                │
├─────────────────────────────────────────────────────────┤
│  Layer 0: GPU Runtime (src/kfd/mod.rs — KFD driver,     │
│           AQL queues, PM4 commands, ELF loader)          │
└─────────────────────────────────────────────────────────┘
```

## 3. 核心模块地图

### 3.1 `src/kfd/` — GPU 运行时 (Layer 0)
**最大文件**: `src/kfd/mod.rs` — 包含全部 GPU 驱动抽象

| 结构体 | 行号 | 职责 |
|--------|------|------|
| `KfdDevice` | — | KFD 设备句柄, ioctl 封装 |
| `GpuBuffer` | 1068 | GPU 内存分配/映射/读写 |
| `AqlQueue` | 1279 | HSA AQL 队列管理 |
| `AqlPkt` | 1581 | AQL Packet 构造 |
| `Pm4Queue` | 2471 | PM4 命令队列 |
| `Pm4CmdBuilder` | 2302 | PM4 命令包构建器 |
| `GpuKernel` | 2697 | 内核加载/执行 |
| `ElfParser` | 2822 | ELF 二进制解析 |
| `DispatchPool` | 3004 | Dispatch 池管理 |
| `GpuMemset` | 3092 | GPU 内存填充 |

### 3.2 `src/t0/` — T0 编译器 (Layer 1-3, 45 个子模块)
项目最复杂的模块，45 个子模块，718 个函数。

**编译管线**:
```
Block DSL ──▶ Tile IR ──▶ Tile SSA ──▶ SSA IR ──▶ RegAlloc ──▶ ASM Emitter
  (block_dsl.rs)  (tile_ir.rs)  (tile_ssa.rs)  (ssa_ir.rs)  (regalloc.rs)  (asm_emitter.rs)
                      │                                        │
                      ▼                                        ▼
                block_dsl_to_ssa.rs                     ssa_regalloc.rs
```

**关键文件**:

| 文件 | 行数级 | 功能 |
|------|--------|------|
| `mod.rs` | 45 模块 | T0 模块入口，导出所有子模块 |
| `ir.rs` | — | IR 定义: Op, Operand, SOperand, Width, Alignment, WmmaFormat |
| `ssa_ir.rs` | — | SSA IR: MachFunc, ImplicitReg |
| `tile_ir.rs` | 5 模块 | Tile GEMM IR: EpilogueOp, TileTranspose, WarpOrientation |
| `tile_ssa.rs` | 1 模块 | Tile SSA: TileOp, BinOpKind, Terminator, ElemStep |
| `block_dsl.rs` | 3 模块 | Block DSL: BNode, BType, LdsDoubleBuffer |
| `asm_emitter.rs` | — | 汇编发射器: emit_op, finish, optimize_smem_loads |
| `compile.rs` | 1 模块 | 编译入口: T0Kernel::new, validate |
| `regalloc.rs` | — | 寄存器分配器 |
| `ssa_regalloc.rs` | 1 模块 | SSA 寄存器分配 |
| `opt_passes.rs` | 1 模块 | 优化 pass |
| `gemm_gen.rs` | 2 模块 | GEMM 生成器: GemmTranspose, EpilogueOp |
| `auto_gemm.rs` | 1 模块 | GEMM 自动调优: tune, benchmark, cache |
| `flash_attn.rs` | — | FlashAttention 内核 |
| `quant_kernels.rs` | — | INT4 量化内核 |
| `adamw_kernels.rs` | — | AdamW 优化器内核 |

**枚举定义** (56 个):
- `ir.rs`: Op, Operand, SOperand, Width, Alignment, WmmaFormat, Target, ArgKind
- `tile_ir.rs`: EpilogueOp, TileTranspose, WarpOrientation
- `tile_ssa.rs`: TileOp, BinOpKind, UnaryOpKind, CmpOpKind, ReduceKind, Terminator, ElemStep, ScalarDType, TensorLayout, TileType
- `block_dsl.rs`: BNode, BType
- `insn_latency.rs`: InsnClass
- `latency_model.rs`: Pipeline, WaitCounter

### 3.3 `src/ignis/` — 神经网络框架 (Layer 4)

| 子模块 | 文件 | 关键结构体/函数 |
|--------|------|-----------------|
| `tensor.rs` | — | `Tensor` — 核心张量类型 |
| `tape.rs` | — | `Tape`, `TapeNode`, `NoGrad` — 自动微分 |
| `gpu_context.rs` | — | `GpuRuntime`, `BufferPool` — GPU 运行时封装 |
| `nn/transformer.rs` | — | `TransformerLayer` |
| `nn/model.rs` | — | `LanguageModel` |
| `nn/linear.rs` | — | `Linear` |
| `nn/embedding.rs` | — | `Embedding` |
| `nn/mod.rs` | — | `Module` trait, `Parameter` |
| `ops/cross_entropy.rs` | — | `cross_entropy` |
| `ops/bf16_matmul.rs` | — | `matmul_with_wt_bf16` |
| `ops/ocpa_attention.rs` | — | `OcpaConfig`, `GpuBufferRef` |
| `ops/fusion.rs` | — | `FusedOp` |
| `ops/silu.rs` | — | SiLU 激活 |
| `ops/rmsnorm.rs` | — | RMSNorm |
| `ops/add.rs` | — | 加法 |
| `ops/shape_ops.rs` | — | 形状操作 |
| `ops/psi_activation.rs` | — | PSI 激活 |
| `tokenizer.rs` | — | `BpeTokenizer`, `VocabTokenizer` |
| `lr_scheduler.rs` | — | `CosineWarmupScheduler`, `ConstantLR`, `LrScheduler` trait |
| `loss_scaler.rs` | — | `LossScaler` |
| `buffer_pool.rs` | — | `BufferPool` |
| `data_loader.rs` | — | `DataLoader` |

### 3.4 `src/universal/` — 通用运行时 (18 模块)

| 子模块 | 功能 |
|--------|------|
| `core/device.rs` | `Kernel`, `Signal`, `ComputeQueue`, `CopyQueue`, `GpuDevice`, `DriverFactory` traits |
| `core/arch.rs` | `Vendor`, `Arch`, `DType` 枚举 |
| `driver/amd/` | AMD KFD 驱动适配 |
| `driver/nvidia/` | NVIDIA 驱动适配 (框架) |
| `driver/ascend/` | Ascend 驱动适配 (框架) |
| `compiler/mod.rs` | `CompilerBackend`, `IsaEncoder` traits |
| `compiler/llvm_backend.rs` | LLVM 后端 |
| `math/mod.rs` | `BlasLib`, `PrimLib`, `RngLib`, `FftLib`, `SparseLib` traits |
| `math/blas.rs` | BLAS 实现 |
| `math/fft.rs` | FFT 实现 |
| `math/swizzle.rs` | Swizzle 模式 |
| `scheduler/mod.rs` | `TileOptimizer`, `InstructionScheduler` traits |
| `scheduler/shared.rs` | `Priority`, `SchedulingPolicy` |
| `runtime/mod.rs` | `MemoryManager` trait |
| `runtime/multi_gpu.rs` | 多 GPU 支持 |
| `runtime/unified_mem.rs` | 统一内存管理 |

### 3.5 `src/rdna3_asm.rs` & `src/rdna3_code_object.rs` — ISA 编码

| 文件 | 功能 |
|------|------|
| `rdna3_asm.rs` | `Rdna3Assembler` — RDNA3/4 指令编码器 |
| `rdna3_code_object.rs` | ELF code object 构建 |
| `rdna3_disasm.rs` | 反汇编器 (`InsnFormat`) |
| `wmma_db.rs` | WMMA 指令数据库 (`WmmaType`, `WmmaTarget`) |

## 4. 边关系分析 (30,435 条)

| 边类型 | 数量 | 说明 |
|--------|------|------|
| CALLS | 10,949 | 函数调用关系 |
| ACCESSES | 10,269 | 字段/属性访问 |
| DEFINES | 2,406 | 定义关系 (模块→符号) |
| MEMBER_OF | 2,382 | 成员归属 |
| CONTAINS | 1,813 | 文件夹→文件包含 |
| HAS_PROPERTY | 1,035 | 结构体→字段 |
| STEP_IN_PROCESS | 1,016 | 执行流步骤 |
| IMPORTS | 182 | use/import 语句 |
| HAS_METHOD | 159 | 类型→方法 |
| METHOD_IMPLEMENTS | 132 | Trait 方法实现 |
| USES | 63 | 使用关系 |
| IMPLEMENTS | 29 | Trait 实现 |

## 5. 功能聚类 (Leiden 社区检测, 233 个)

### 主要聚类 (≥15 符号)

| 聚类 ID | 标签 | 符号数 | 凝聚度 | 说明 |
|---------|------|--------|--------|------|
| comm_8 | T0 | 239 | 0.875 | T0 编译器核心 |
| comm_9 | T0 | 201 | 0.938 | T0 编译器扩展 |
| comm_18 | T0 | 198 | 0.693 | T0 编译器测试 |
| comm_10 | T0 | 80 | 0.737 | T0 SSA/regalloc |
| comm_20 | Ops | 40 | 0.643 | 神经网络算子 |
| comm_32 | Ignis | 40 | 0.830 | Ignis 框架核心 |
| comm_7 | Examples | 38 | 0.670 | 示例与基准 |
| comm_47 | Cluster_47 | 31 | 0.535 | 混合功能 |
| comm_82 | Cluster_82 | 30 | 0.711 | 辅助功能 |
| comm_128 | T0 | 29 | 0.990 | T0 内核生成 |
| comm_231 | Tests | 29 | 0.982 | 测试框架 |
| comm_105 | Cluster_105 | 24 | 0.885 | 编码相关 |
| comm_67 | Cluster_67 | 21 | 0.909 | ISA 验证 |
| comm_154 | T0 | 19 | 0.947 | T0 优化 |
| comm_222 | Universal | 17 | 0.984 | 通用运行时 |
| comm_11 | Kernels | 16 | 0.476 | 内核集合 |
| comm_31 | Ops | 16 | 0.504 | 算子集合 |
| comm_56 | Tests | 16 | 0.714 | 测试 |
| comm_96 | Tests | 16 | 0.305 | 测试 |
| comm_112 | T0 | 16 | 0.857 | T0 辅助 |
| comm_64 | Tests | 15 | 0.848 | 测试 |
| comm_145 | T0 | 15 | 0.933 | T0 工具 |

**聚类总结**: T0 编译器占据 ~720 符号 (最大聚类群)，其次是 Ignis 框架 (80+)、Ops (56+)、Tests (90+)。

## 6. 执行流程 (273 个)

### 跨社区执行流 (最长, ≥5 步)

| 流程 | 步数 | 入口 → 终点 | 类型 |
|------|------|-------------|------|
| Cross_entropy → Tensor | 7 | cross_entropy → Tensor | cross_community |
| Cross_entropy → Ignore_sigpipe | 7 | cross_entropy → ignore_sigpipe | cross_community |
| Cross_entropy → Open_kfd_with_retry | 7 | cross_entropy → KfdDevice.open_kfd_with_retry | cross_community |
| Matmul_with_wt_bf16 → Validate | 6 | matmul_with_wt_bf16 → T0Kernel.validate | cross_community |
| Matmul_with_wt_bf16 → Finish | 6 | matmul_with_wt_bf16 → AsmEmitter.finish | cross_community |
| Compile_via_ssa → BasicBlock | 6 | compile_via_ssa → BasicBlock | cross_community |
| Compile_via_ssa → TileFunc | 6 | compile_via_ssa → TileFunc | cross_community |
| Matmul_with_wt_bf16 → VerifyResult | 5 | matmul_with_wt_bf16 → VerifyResult | cross_community |
| Matmul_with_wt_bf16 → New | 5 | matmul_with_wt_bf16 → T0Kernel.new | cross_community |
| Main → Result_type | 5 | train_mlp:main → BNode.result_type | intra_community |

**执行流分析**: 最关键的跨层调用链是 `cross_entropy` 和 `matmul_with_wt_bf16`，它们贯穿 Ignis → T0 → KFD 所有层次。

## 7. Trait 接口图谱 (20 个)

```
┌──────────── universal/core/ ────────────┐
│ Kernel, Signal, ComputeQueue, CopyQueue │
│ GpuDevice, DriverFactory                │
├──────────── universal/math/ ────────────┤
│ BlasLib, PrimLib, RngLib, FftLib        │
│ SparseLib                               │
├──────────── universal/runtime/ ─────────┤
│ MemoryManager                           │
├──────────── universal/compiler/ ────────┤
│ CompilerBackend, IsaEncoder             │
├──────────── universal/scheduler/ ───────┤
│ TileOptimizer, InstructionScheduler     │
├──────────── ignis/ ─────────────────────┤
│ Module (nn/mod.rs), LrScheduler         │
├──────────── t0/ ────────────────────────┤
│ CfgProvider (domtree.rs)                │
│ Schedule (schedule.rs)                  │
└─────────────────────────────────────────┘
```

## 8. 模块依赖拓扑 (按模块数量)

```
src/t0/mod.rs          (45 子模块)  ←── 最大模块
src/universal/mod.rs   (18 子模块)
src/ignis/ops/mod.rs   (13 子模块)
src/ignis/mod.rs       (12 子模块)
src/lib.rs             (9  子模块)
src/universal/math/    (8  子模块)
src/t0/tile_ir.rs      (5  子模块)
src/ignis/nn/mod.rs    (4  子模块)
src/t0/block_dsl.rs    (3  子模块)
```

## 9. 文件热度图 (按模块数排序)

**核心热区** (模块密度最高的文件):
1. `src/t0/mod.rs` — 45 个模块导出
2. `src/universal/mod.rs` — 18 个模块
3. `src/ignis/ops/mod.rs` — 13 个模块
4. `src/ignis/mod.rs` — 12 个模块
5. `src/lib.rs` — 9 个模块导出

## 10. 关键调用路径

### 训练路径: train_mlp → GPU 执行
```
main (train_mlp.rs)
 └─ LanguageModel.forward
     ├─ TransformerLayer.forward
     │   ├─ matmul_with_wt_bf16 (ops/bf16_matmul.rs)
     │   │   └─ T0Kernel.compile → BlockKernel.compile_via_ssa
     │   │       └─ SSA pipeline → AsmEmitter.finish
     │   ├─ cross_entropy (ops/cross_entropy.rs)
     │   └─ rmsnorm, silu, shape_ops
     └─ GpuRuntime.execute
         └─ KfdDevice → AqlQueue → GpuKernel.dispatch
```

### GEMM 编译路径
```
matmul_with_wt_bf16
 └─ T0Kernel.new (compile.rs)
     ├─ BlockKernel (block_dsl.rs)
     │   ├─ BNode::LoadGlobal, BNode::WMMA, BNode::StoreGlobal
     │   └─ LdsDoubleBuffer, barrier
     ├─ compile_via_ssa (block_dsl_to_ssa.rs)
     │   ├─ TileSSA lowering (tile_ssa_lower.rs)
     │   ├─ SSA optimization (opt_passes.rs)
     │   ├─ Register allocation (ssa_regalloc.rs)
     │   └─ ASM emission (asm_emitter.rs)
     └─ validate → ISA verification (isa_verifier.rs)
```

## 11. 技术债标记 (GitNexus 未解析项)

- `src/t0/ir.rs.bak` — 备份文件残留
- `src/t0/check_v_add_co.rs` — 检查用文件
- `src/t0/monitor.rs` — GPU 监控
- `src/t0/kernel_debugger.rs` — 内核调试器

---

*本报告由 GitNexus 知识图谱 Cypher 查询生成。如需重新索引，运行:*
```bash
cd /home/yanli/work/9060xt/t0-gpu
node /data/work/GitNexus/gitnexus/dist/cli/index.js analyze --index-only -v
```
