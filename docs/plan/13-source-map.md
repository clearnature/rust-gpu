# 源码地图 — 另一个 AI 接手指南

> 本文档列出所有关键源文件的精确路径、行数、功能和入口函数。
> 目标：让接手的 AI 在 5 分钟内定位任何功能。

## 一、t0-gpu 项目结构

```
/home/yanli/work/9060xt/t0-gpu/
├── src/
│   ├── lib.rs                    (51 行)  — crate 入口
│   ├── prelude.rs                (21 行)  — 公共导出
│   │
│   ├── kfd/
│   │   └── mod.rs                (3249 行) — KFD 运行时核心
│   │       ├── struct KfdDevice           — GPU 设备抽象
│   │       ├── fn alloc_vram()            — VRAM 分配
│   │       ├── fn create_queue()          — AQL 队列创建
│   │       ├── fn dispatch()              — kernel dispatch + doorbell
│   │       └── fn map_to_cpu()            — CPU mmap VRAM
│   │
│   ├── rdna3_asm.rs              (4505 行) — GFX1100/1200 ISA 编码器
│   │   ├── fn encode_vop2_fma()           — VOP2 FMA 编码
│   │   ├── fn encode_smem_load()          — SMEM load 编码
│   │   ├── fn encode_flat_store()         — FLAT store 编码
│   │   ├── fn encode_wmma()               — WMMA 矩阵指令
│   │   ├── fn encode_swmmac()             — SWMMAC 稀疏矩阵 (GFX1200)
│   │   └── fn is_gfx1200()               — 架构分支
│   │
│   ├── rdna3_code_object.rs      (1708 行) — HSA ELF code object 生成
│   │   ├── struct KernelConfig            — kernel 配置 (VGPR/SGPR/LDS/workgroup)
│   │   ├── fn generate_elf()              — 生成可加载的 ELF 二进制
│   │   └── struct KernelDescriptor        — 64 字节硬件描述符
│   │
│   ├── rdna3_disasm.rs           (1790 行) — GFX ISA 反汇编器
│   │
│   ├── wmma_db.rs                (786 行)  — WMMA 指令格式数据库
│   │
│   ├── t0/                       (45971 行) — T0 编译器
│   │   ├── mod.rs                (入口)
│   │   │
│   │   ├── dsl.rs                (65 行)   — DSL 类型定义 (DType, CompiledKernel)
│   │   │
│   │   ├── ir.rs                 (1432 行) — IR 定义 (80+ Op 变体)
│   │   │   ├── enum Op                    — VAddF32/VFmaF32/GlobalLoad/WMMA/...
│   │   │   ├── struct VReg/SReg           — 虚拟寄存器
│   │   │   └── fn Target::detect()        — GFX1100/1200 自动检测
│   │   │
│   │   ├── ssa_ir.rs             (3677 行) — Machine SSA IR
│   │   │   ├── fn lift_to_ssa()           — Vec<Op> → MachFunc (SSA 提升)
│   │   │   ├── fn lower_from_ssa()        — MachFunc → Vec<Op> (SSA 降级)
│   │   │   ├── struct MachFunc/BasicBlock — SSA 函数/基本块
│   │   │   └── 8+ 优化 pass (常量折叠/CSE/LICM/DCE/...)
│   │   │
│   │   ├── opt_passes.rs         (1536 行) — 优化 pass 管线
│   │   │   ├── Phase A: 常量折叠, 代数简化, 复制传播, CSE, FMA 合成, LICM
│   │   │   ├── Phase B: 循环展开, 强度削减
│   │   │   ├── Phase C: 迭代代数简化 + DCE
│   │   │   └── Phase D: Waitcnt 优化, 合并, 软件流水, 指令调度, Pingpong
│   │   │
│   │   ├── regalloc.rs           (361 行)  — 线性扫描寄存器分配 (遗留)
│   │   ├── ssa_regalloc.rs       (1439 行) — SSA-aware 寄存器分配
│   │   │   ├── fn allocate_ssa()          — SSA 区间线性扫描
│   │   │   ├── fn insert_spill_reloads()  — LDS 溢出插入
│   │   │   └── VGPR 上限 254 (CWSR 安全)
│   │   │
│   │   ├── compile.rs            (1532 行) — 完整编译流水线
│   │   │   ├── struct T0Kernel            — kernel builder (60+ 便捷方法)
│   │   │   ├── fn compile()               — validate→optimize→regalloc→emit
│   │   │   └── fn to_assembly_with_info() — 生成汇编 + LDS 信息
│   │   │
│   │   ├── asm_emitter.rs        — 汇编文本发射器
│   │   │
│   │   ├── tile_ir.rs            (7574 行) — Tile IR (核心创新)
│   │   │   ├── struct TileGemm            — Tile 级 GEMM 规格
│   │   │   ├── fn lower_gemm()            — TileGemm → T0Kernel (自动编译)
│   │   │   ├── 20+ 预设配置 (32x32 到 256x128)
│   │   │   └── enum EpilogueOp            — 8 种融合 epilogue
│   │   │
│   │   ├── tile_ssa.rs           (2775 行) — Tile SSA IR
│   │   │   ├── enum TileOp               — 35 种 tile 操作
│   │   │   ├── enum TensorLayout          — Blocked/Shared/MmaAccumulator
│   │   │   └── struct TileFunc            — SSA builder API
│   │   │
│   │   ├── tile_ssa_lower.rs     (2911 行) — Tile SSA → ISA 降级
│   │   │
│   │   ├── gemm_gen.rs           (1383 行) — GEMM kernel 生成器
│   │   │   ├── struct GemmConfig          — tile_m/n/k, split_k, wgp, transpose
│   │   │   ├── fn build_kernel()          — 生成完整 GEMM kernel
│   │   │   └── 13 预设 tile 配置
│   │   │
│   │   ├── auto_gemm.rs          (611 行)  — GEMM 运行时 autotune
│   │   │   ├── fn auto_gemm()             — 一键 tune+dispatch
│   │   │   └── 磁盘缓存: ~/.t0_autotune/
│   │   │
│   │   ├── cost_model.rs         (1368 行) — 性能代价模型
│   │   │   ├── struct GFX1100Limits       — 硬件参数 (256 VGPR, 96 CU, ...)
│   │   │   ├── fn predict_best()          — Roofline + K-loop → 最优配置
│   │   │   └── 400+ 候选穷举搜索
│   │   │
│   │   ├── latency_model.rs      (575 行)  — N14 校准延迟模型
│   │   │   ├── VALU=1, VMEM=47, LDS=7, WMMA=4, SWMMAC=2
│   │   │   └── GPU 时钟 3.15 GHz, Shader cycle ~3175 ps
│   │   │
│   │   ├── flash_attn.rs         (776 行)  — FlashAttention-1
│   │   │   ├── fn flash_attn_forward()    — 在线 softmax, 因果掩码, GQA
│   │   │   └── MAX_KV_LEN=4 (SGPR 预算限制)
│   │   │
│   │   ├── block_dsl.rs          — 块级 DSL (逐元素 kernel builder)
│   │   ├── block_dsl_to_ssa.rs   — DSL → SSA 转换
│   │   ├── schedule.rs           — 指令调度
│   │   ├── domtree.rs            — 支配树
│   │   ├── insn_latency.rs       — 指令延迟查询
│   │   ├── isa_probe.rs          — ISA 特性探测
│   │   ├── isa_verifier.rs       — ISA 验证器
│   │   ├── gpu_probe.rs          — GPU 硬件探测
│   │   ├── monitor.rs            — 性能监控
│   │   ├── math.rs               — 数学工具
│   │   ├── prefill_dispatch.rs   — Prefill 调度
│   │   ├── prefill_spec_cache.rs — Prefill 规格缓存
│   │   │
│   │   ├── elementwise_kernels.rs — 逐元素 kernel
│   │   ├── softmax_kernels.rs    — Softmax kernel
│   │   ├── rmsnorm_kernels.rs    — RMSNorm kernel
│   │   ├── rope_kernels.rs       — RoPE kernel
│   │   ├── ce_loss_kernels.rs    — Cross Entropy loss kernel
│   │   ├── embedding_kernels.rs  — Embedding kernel
│   │   ├── adamw_kernels.rs      — AdamW optimizer kernel
│   │   ├── causal_mask_kernels.rs — 因果掩码 kernel
│   │   ├── quant_kernels.rs      — 量化 kernel
│   │   ├── ffn_fused_kernels.rs  — FFN 融合 kernel
│   │   │
│   │   └── (测试文件)
│   │       ├── precision_vs_torch.rs (310 行) — PyTorch 精度对比
│   │       ├── test_e2e_pipeline.rs  — 端到端测试
│   │       ├── test_tile_gemm_suite.rs — Tile GEMM 测试套件
│   │       └── gpu_tests.rs          — GPU 硬件测试
│   │
│   └── ignis/                    (6590 行) — 神经网络框架
│       ├── mod.rs                (30 行)   — 模块导出
│       │
│       ├── tensor.rs             (462 行)  — GPU Tensor + autodiff
│       │   ├── struct Tensor              — Arc<GpuBuffer> + shape + grad
│       │   ├── fn from_f32()              — 创建 f32 tensor
│       │   └── fn set_requires_grad()     — 启用梯度追踪
│       │
│       ├── tape.rs               (446 行)  — 反向模式自动微分
│       │   ├── struct TapeNode            — 前向节点 + backward 闭包
│       │   ├── fn Tape::record()          — 记录前向操作
│       │   └── fn Tape::backward()        — 反向传播
│       │
│       ├── gpu_context.rs        (585 行)  — GPU 运行时封装
│       │   ├── struct GpuRuntime          — device + buffer pool + compile cache
│       │   └── fn alloc_f32/alloc_bf16()  — 快速分配
│       │
│       ├── buffer_pool.rs        (78 行)   — 2^n 桶缓存池
│       ├── loss_scaler.rs        (115 行)  — 动态 loss scaling
│       ├── grad_clip.rs          (89 行)   — 梯度裁剪 (⚠️ GPU→CPU 往返)
│       ├── lr_scheduler.rs       (75 行)   — CosineWarmup + ConstantLR
│       ├── data_loader.rs        (99 行)   — 数据加载
│       ├── tokenizer.rs          (171 行)  — BPE tokenizer
│       ├── tests.rs              (701 行)  — 框架测试
│       │
│       ├── ops/                  (2807 行) — 运算层
│       │   ├── bf16_matmul.rs    (424 行)  — BF16 GEMM
│       │   ├── add.rs            (489 行)  — 向量加法
│       │   ├── ocpa_attention.rs (589 行)  — OCPA 前向+反向
│       │   ├── shape_ops.rs      (350 行)  — softmax/transpose/reshape
│       │   ├── rmsnorm.rs        (317 行)  — RMSNorm
│       │   ├── fusion.rs         (234 行)  — kernel 融合
│       │   ├── silu.rs           (184 行)  — SiLU 激活
│       │   ├── embedding.rs      (186 行)  — Embedding
│       │   ├── cross_entropy.rs  (127 行)  — Cross entropy
│       │   ├── fused_rmsnorm_gemm.rs (154 行) — 融合 RMSNorm+GEMM
│       │   ├── psi_activation.rs (59 行)   — PSI 激活
│       │   └── gemm_autotune.rs  (30 行)   — GEMM autotune
│       │
│       └── nn/                   (552 行)  — 神经网络层
│           ├── transformer.rs    (183 行)  — TransformerLayer (OCPA + SwiGLU)
│           ├── model.rs          (108 行)  — Module trait
│           ├── linear.rs         (86 行)   — Linear layer
│           └── embedding.rs      (142 行)  — Embedding layer
│
├── benchmarks/                   — 基准测试数据
├── tests/                        — 集成测试
├── examples/                     — 示例代码
└── docs/plan/                    — 本文档目录
```

## 二、外部参考项目

### sass-assembler (浑天)
```
/data/rtl-sdr/sass-assembler/
├── src/sass/
│   ├── encoder/pascal_encoder.h  (171 行) — Pascal SASS 编码 (真实 bit 操作)
│   ├── backends/
│   │   ├── volta_encoder.h       (239 行) — Volta 128-bit 编码
│   │   ├── volta_opcode_table.h  (115 行) — Volta opcode 常量 (cuobjdump 验证)
│   │   ├── ampere_precise_encoder.h (91 行) — Ampere 128-bit 双字编码
│   │   ├── amd_rdna4.h           (683 行) — RDNA4 ISA 参考表 (⚠️ 无编码)
│   │   ├── volta_backend.h       (200 行) — Volta 后端
│   │   ├── jit_compiler.h        (82 行)  — JIT 编译器
│   │   ├── dsl_encoder.h         (93 行)  — DSL 编码器
│   │   └── x86_backend.h         (177 行) — x86 定义 (⚠️ 无编码)
│   ├── pascal_backend.h          (151 行) — Pascal 后端
│   ├── pascal_hal.cpp            (151 行) — Pascal HAL (v512 向量)
│   ├── ilp_model.h               (654 行) — ILP 硬件模型 (7 架构延迟表)
│   ├── block_encoder.cpp         (164 行) — 64-bit 块编码 + VAVX3 融合
│   ├── manifold_scheduler.cpp    (121 行) — 4320D 流形调度
│   └── assembler.cpp             (49 行)  — 主汇编器入口
├── build/
│   ├── ht-as                     — 汇编器 CLI
│   ├── ht-dis                    — 反汇编器 CLI
│   ├── libsass_core.a            — 核心静态库
│   └── test_*                    — 22 个测试可执行文件
└── tests/                        — 测试源码
```

### tinygrad
```
/home/yanli/work/9060xt/tinygrad/tinygrad/
├── runtime/
│   ├── ops_amd.py                — AMD 运行时 (KFD + AM 双路径)
│   ├── ops_nv.py                 — NVIDIA 运行时 (GPFIFO + QMD)
│   ├── ops_cuda.py               — CUDA 运行时
│   ├── ops_metal.py              — Metal 运行时
│   └── support/
│       ├── hcq.py                (631 行) — HCQ 抽象 (FileIO/MMIO/信号/队列)
│       └── nv/nvdev.py           — 裸金属 NV 设备 (MMIO + 页表)
├── codegen/
│   ├── late/
│   │   ├── linearizer.py         (96 行)  — 优先级 toposort
│   │   ├── regalloc.py           (137 行) — Linear scan + spill
│   │   ├── coalesce.py           — 内存合并
│   │   └── gater.py              — 门控优化
│   ├── opt/
│   │   ├── heuristic.py          (195 行) — 手写优化启发式
│   │   ├── search.py             (179 行) — BEAM search
│   │   ├── tc.py                 — Tensor core 匹配
│   │   └── postrange.py          — Scheduler (轴类型管理)
│   └── __init__.py               — full_rewrite_to_sink (25+ passes)
├── schedule/__init__.py           — 图级调度 (Kahn 算法)
├── uop/ops.py                    — UOp IR 定义
├── device.py                     — Compiled device 抽象
└── runtime/autogen/
    └── nv_570.py                 (24867 行) — NVIDIA 自动生成绑定
```

### ROCm
```
/data/ROCm/rocm-systems/projects/
├── rocr-runtime/runtime/hsa-runtime/
│   ├── inc/hsa.h                 (5762 行) — HSA API 头文件
│   ├── inc/amd_hsa_queue.h       — AQL 队列结构
│   ├── core/runtime/hsa.cpp      — HSA API 实现入口
│   ├── core/runtime/runtime.cpp  — Runtime 单例
│   ├── core/runtime/amd_gpu_agent.cpp — GPU agent
│   ├── core/runtime/amd_aql_queue.cpp — AQL 队列
│   ├── core/driver/kfd/amd_kfd_driver.cpp — KFD 驱动适配
│   └── core/runtime/thunk_loader.cpp — dlopen libhsakmt
├── rocr-runtime/libhsakmt/
│   ├── include/hsakmt/hsakmt.h   (1259 行) — libhsakmt API
│   ├── include/hsakmt/linux/kfd_ioctl.h (1843 行) — KFD ioctl 定义
│   ├── src/fmm.c                 (4719 行) — Frame buffer Memory Manager
│   ├── src/memory.c              (932 行)  — 内存分配
│   ├── src/queues.c              (1080 行) — 队列管理
│   └── src/topology.c            — GPU 拓扑发现
└── clr/
    ├── hipamd/src/hip_memory.cpp  — hipMalloc 实现
    ├── hipamd/src/hip_platform.cpp — hipLaunchKernel 实现
    └── rocclr/device/rocm/rocdevice.cpp — Device::svmAlloc
```

## 三、快速定位指南

| 我想找... | 去哪里看 |
|----------|---------|
| KFD ioctl 怎么调用 | `src/kfd/mod.rs` → fn alloc_vram(), fn create_queue() |
| GFX1200 指令怎么编码 | `src/rdna3_asm.rs` → fn encode_*() + is_gfx1200() 分支 |
| GEMM 怎么生成的 | `src/t0/gemm_gen.rs` → fn build_kernel() |
| GEMM 怎么自动调优 | `src/t0/auto_gemm.rs` → fn auto_gemm() |
| Tile IR 怎么工作 | `src/t0/tile_ir.rs` → struct TileGemm + fn lower_gemm() |
| SSA 优化有哪些 pass | `src/t0/opt_passes.rs` → fn optimize() (4 阶段 15 pass) |
| 寄存器怎么分配 | `src/t0/ssa_regalloc.rs` → fn allocate_ssa() |
| ELF 怎么生成 | `src/rdna3_code_object.rs` → fn generate_elf() |
| Autograd 怎么工作 | `src/ignis/tape.rs` → fn Tape::backward() |
| Transformer 怎么实现 | `src/ignis/nn/transformer.rs` → fn forward_simple() |
| OCPA Attention 怎么工作 | `src/ignis/ops/ocpa_attention.rs` → fn ocpa_forward() |
| NVIDIA 怎么 dispatch | tinygrad `ops_nv.py` → fn _submit_to_gpfifo() |
| NVIDIA 内存怎么分配 | tinygrad `ops_nv.py` → NVKIface.alloc() |
| KFD vs NVIDIA 差异 | `docs/plan/01-rocm-runtime-stack.md` + `12-nvidia-driver-interface.md` |
| 性能数据在哪里 | `docs/plan/11-benchmark-analysis.md` |
| 缺什么数学库 | `docs/plan/02-math-lib-dependencies.md` |
| 通用运行时怎么设计 | `docs/plan/03-universal-runtime-arch.md` |
| 浑天汇编器能用吗 | `docs/plan/04-sass-assembler-audit.md` |

## 四、关键常量和配置

| 常量 | 值 | 文件 |
|------|-----|------|
| VGPR 上限 (GFX1100) | 256 | src/t0/regalloc.rs |
| VGPR CWSR 安全上限 | 254 | src/t0/ssa_regalloc.rs |
| SGPR 上限 | 106 | src/t0/ir.rs |
| LDS 大小 (GFX1100) | 64KB/WG, 128KB/CU | src/t0/cost_model.rs |
| WMMA tile (GFX1100) | 16x16x16 | src/wmma_db.rs |
| SWMMAC tile (GFX1200) | 16x16x64 | src/wmma_db.rs |
| GPU 时钟 (GFX1200) | 3.15 GHz | src/t0/latency_model.rs |
| VMEM 延迟 | 47 cycle | src/t0/latency_model.rs |
| LDS 延迟 | 7 cycle | src/t0/latency_model.rs |
| WMMA 延迟 | 4 cycle | src/t0/latency_model.rs |
| SWMMAC 延迟 | 2 cycle | src/t0/latency_model.rs |
| Autotune 缓存目录 | ~/.t0_autotune/ | src/t0/auto_gemm.rs |
| Buffer pool 最小分配 | 4096 字节 (KFD 页) | src/ignis/buffer_pool.rs |
| Tensor 最小分配 | 512 字节 (dwordx4 对齐) | src/ignis/tensor.rs |
