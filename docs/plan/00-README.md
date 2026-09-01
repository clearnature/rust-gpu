# t0-gpu 通用 GPU 运行时规划

> 生成日期: 2026-08-24
> 来源: ROCm/TheRock 数学库依赖链分析 + t0-gpu/sass-assembler/tinygrad 三方对比

## 文档索引

| 文件 | 内容 |
|------|------|
| `01-rocm-runtime-stack.md` | ROCm 四层运行时栈分析 (HIP→ROCr→libhsakmt→KFD) |
| `02-math-lib-dependencies.md` | ROCm 数学库依赖链 + t0-gpu 缺失分析 |
| `03-universal-runtime-arch.md` | 通用 GPU 运行时架构设计 |
| `04-sass-assembler-audit.md` | 浑天 SASS 汇编器深度审计 |
| `05-ilp-scheduling-analysis.md` | ILP 调度模型深度分析 |
| `06-tinygrad-comparison.md` | tinygrad 对比分析 |
| `07-universal-runtime-plan.md` | 综合路线图 |
| `08-ignis-framework.md` | ignis 框架深度分析 |
| `09-t0-compiler-pipeline.md` | T0 编译器流水线分析 |
| `10-memory-model-comparison.md` | 内存模型对比 (AMD/NVIDIA/国产) |
| `11-benchmark-analysis.md` | 量化性能基准分析 |
| `12-nvidia-driver-interface.md` | NVIDIA 内核驱动接口分析 (tinygrad 参考) |
| `13-source-map.md` | 源码地图 — 另一个 AI 接手指南 (精确文件路径+行号) |
| `14-scheduler-architecture.md` | 完整调度器架构 (5 层调度 + 三方对比) |
| `15-universal-runtime-framework.md` | **通用 GPU 运行时完整框架设计 (可直接用于实现)** |
| `16-local-repo-resources.md` | 本地仓库资源分析 (13 个相关项目) |

## 核心结论

1. t0-gpu 已经用 KFD 直通替代了 ROCm 的 HIP+ROCr+libhsakmt 三层，延迟降低 13-27%
2. 缺失的是通用数学原语层 (scan/sort/rand/GEMV) 和跨厂商驱动抽象
3. sass-assembler 提供了 Pascal/Volta/Ampere 的真实 SASS 编码器，但 AMD RDNA4 只有定义
4. tinygrad 的 BEAM search 优化策略值得借鉴，但不做指令级调度
5. 理想的通用运行时 = t0-gpu 运行时 + tinygrad BEAM 搜索 + sass-assembler ILP 调度 + 多厂商 ISA 编码器
6. ignis 框架 (6590 行) 是完整的训练+推理框架，包含 Autograd、OCPA Attention、Transformer
7. T0 编译器 (45971 行) 含 Tile IR (13260 行)，是最大的创新——tile 级别的计算抽象
8. 内存模型建议先做分离地址空间，后期再加统一地址空间 (UVM/SVM)
9. 最大性能瓶颈是缺少 KV cache 和 GEMV kernel
10. NVIDIA 驱动接口比 KFD 复杂 10 倍 (多设备文件 + RM 对象层次 + UVM), 但 tinygrad 已有完整实现 (~1100 行 Python)
11. 调度器是 5 层架构: 图级→高层优化→模板→指令→硬件; t0-gpu 在指令调度层最强 (软件流水+Pingpong), tinygrad 在高层优化层最强 (BEAM search)
12. 本地有 55 个相关仓库; cubecl (159K 行 Rust, 6 个 GPU 后端) 是最大宝藏, 已实现通用运行时的 80% 架构
13. DeepGEMM (1550T on H800) / CUTLASS / FlashMLA / triton 等前沿项目都在本地可用
