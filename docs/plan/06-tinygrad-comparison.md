# tinygrad 对比分析

> 分析目标: /home/yanli/work/9060xt/tinygrad
> 审计日期: 2026-08-24

## tinygrad 架构概览

```
Tensor ops → UOp graph → Schedule (toposort kernels)
  → Optimize (range transforms, tensor cores, beam search)
  → Expand/Devectorize → Memory coalescing → Decomposition
  → Linearize (priority-based toposort)
  → Register allocation (ISA renderers only)
  → Render to source/assembly → Compile → Execute via HCQ
```

## 关键组件

### 1. 调度器 — 两层设计

**图级调度** (schedule/__init__.py):
- Kahn 算法 (BFS + in-degree) 拓扑排序
- RAW/WAR 依赖通过 AFTER 链编码
- 检测循环并报错

**Kernel 级调度** (codegen/opt/postrange.py):
- RANGE 轴类型: GLOBAL/LOCAL/UPCAST/UNROLL/REDUCE/GROUP/WARP/THREAD
- 不是 ILP 指令调度, 是 tiling 和变换调度

### 2. 编译流水线

25+ graph rewrite passes:
1. Multi-device resolution
2. Symbolic simplification
3. **BEAM search 优化** (搜索 tiling/unrolling/local/group/thread/TC 参数组合)
4. Expander (向量化→显式元素访问)
5. Reduce removal (累加器模式)
6. Local buffer allocation (LDS)
7. Devectorization (向量→标量)
8. **Memory coalescing** (相邻 load 合并成 float4/half2)
9. Decomposition (浮点除法→近似, 超越函数→多项式)
10. Implicit barriers (shared memory RAW/WAR hazard)
11. Control flow insertion

### 3. 线性化器 (codegen/late/linearizer.py)

优先级 toposort, 不是延迟感知调度:
```python
match u.op:
  case Ops.LOAD: priority = -1    # loads early
  case Ops.STORE: priority = 1    # stores late
  case Ops.RANGE: priority = 5    # ranges late
  case _: priority = 0
```

静态优先级, 不看延迟. LD_GLOBAL (350 cycle) 和 LDS (8 cycle) 同等对待.

### 4. BEAM 搜索优化 (codegen/opt/search.py)

```python
actions = [Opt(op=OptOps.UPCAST, axis=axis, arg=amt) for amt in [0,2,3,4,5,7] ...]
actions += [Opt(op=OptOps.UNROLL, axis=axis, arg=amt) for amt in [0,4,7] ...]
actions += [Opt(op=OptOps.LOCAL, axis=axis, arg=amt) for amt in [2,3,4,8,13,16,29] ...]
actions += [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, ...))]
# 暴力尝试所有组合, 实测每个 kernel 的执行时间, 选最快的
```

### 5. 寄存器分配 (codegen/late/regalloc.py)

Linear scan + spilling:
- 计算 live range
- 分配最优寄存器 (最远下次使用优先)
- 溢出到栈, 插入 load/store fill

### 6. AMD 运行时 (runtime/ops_amd.py)

**双重 GPU 接口:**

| 接口 | 类 | 方式 | 说明 |
|------|-----|------|------|
| KFD | KFDIface | /dev/kfd ioctl | 标准路径, 同 t0-gpu |
| AM | PCIIface | 直接 PCI MMIO | 绕过 KFD, 更快但需 root |

AM (AMDGPU Manager) 通过 /dev/mem 或 PCI sysfs 直接读写 GPU 的 GC/NBIO/PM4 寄存器.

**HCQ (Hardware Command Queue) 抽象:**
- AMDComputeQueue / AMDComputeAQLQueue — PM4 command buffer
- AMDCopyQueue — SDMA copy
- AMDProgram — kernel descriptor + 资源描述符
- AMDSignal — GPU timeline 信号

**支持的 GPU:** gfx942 (MI300), gfx950 (MI350), gfx1100 (RX 7900), gfx1200 (RX 9060 XT), gfx1201 (RX 9700)

### 7. ISA 渲染

三种路径:
| 路径 | 输出 | 说明 |
|------|------|------|
| HIPRenderer | C-like HIP 源码 → comgr/HIPCC → HSACO ELF | 主要路径 |
| AMDLLVMRenderer | LLVM IR → LLVM 后端 → HSACO ELF | 替代路径 |
| ISA Renderer | ISA 指令 UOp → 寄存器分配 → 二进制编码 | 目前仅 x86 |

AMD ISA 编码: renderer/amd/elf.py 的 assemble_linear() 直接编码 AMD ISA 指令到 ELF.

## tinygrad vs sass-assembler ILP 对比

| 维度 | tinygrad | sass-assembler ILP |
|------|---------|-------------------|
| 核心策略 | BEAM search 暴力搜索 | 延迟模型静态调度 |
| 搜索空间 | ~50 种优化动作组合 | 无搜索, 确定性重排 |
| 线性化 | toposort + 优先级 (LOAD 早, STORE 晚) | cycle-aware list scheduling |
| 延迟建模 | ❌ 无 — 靠实测量 | ✅ 7 个架构的延迟表 |
| 端口冲突 | ❌ 无 | ❌ 有但没用 |
| 寄存器分配 | ✅ linear scan + spill | ⚠️ 基础线性扫描 |
| Tensor Core | ✅ WMMA UOp 匹配 | ✅ 三阶 FP16 延迟建模 |
| 软件流水线 | ❌ | ❌ |
| 跨基本块 | ✅ (UOp 图级别) | ❌ (单基本块) |
| 输出格式 | 文本 (HIPRenderer/LLVM IR) → 外部编译器 | 直接机器码 (bit 操作) |
| 编译速度 | 慢 (BEAM search: 秒级) | 快 (确定性: μs 级) |
| 最优性 | 经验最优 (实测最快) | 理论最优 (延迟模型) |

## tinygrad vs t0-gpu 对比

| | t0-gpu | tinygrad |
|---|---|---|
| 语言 | Rust | Python |
| 高层优化 | ❌ 手写 kernel | ✅ 自动 (BEAM search) |
| 指令调度 | ❌ 手动 | ❌ 交给 LLVM/comgr |
| ISA 编码 | ✅ GFX1100/1200 直接编码 | ❌ 依赖 comgr/LLVM |
| GPU 运行时 | ✅ KFD 直通 | ✅ KFD + AM (PCI 直通) |
| 寄存器分配 | ✅ 专用分配器 | ✅ linear scan + spill |
| 数学库 | ✅ GEMM/Attention | ❌ 调用 vendor 库 |
| 跨厂商 | ❌ AMD only | ✅ 10+ 后端 |
| 编译速度 | ✅ 快 (μs) | ❌ 慢 (BEAM search) |
| 实际性能 | ✅ 超越 rocBLAS | ✅ 接近 vendor 库 |

## tinygrad 对通用运行时的启示

| tinygrad 的设计 | 对 t0-gpu 的价值 |
|----------------|-----------------|
| HCQ 抽象 | 可复用 — compute queue + copy queue + signal 抽象 |
| AM 直接 PCI | 值得考虑 — 比 KFD 更快 |
| 双路径选择 | 运行时自动选 KFD 或 AM |
| BEAM search | t0-gpu 缺少的高层优化 |
| 内存合并 | 自动把相邻 load 合并成 vector load |
| 25+ rewrite passes | Pattern matching 驱动的编译优化 |
| UOp IR | 比 SSA IR 更高层更通用, 支持 symbolic shape |

## 三角对比

```
                    高层优化能力
                        ▲
              tinygrad ●│  BEAM search, 25+ passes
                        │  tensor core 匹配
                        │
                        │         sass-assembler ●
                        │           流形调度, ILP 模型
                        │
           t0-gpu ●─────┼──────────────────────────► 底层控制能力
             JIT GEMM   │                    KFD 直通
             手写 kernel │                    ISA 编码
             零依赖     │                    零中间层
```

## 理想组合

t0-gpu 的 KFD 运行时 + tinygrad 的 BEAM 搜索优化 + sass-assembler 的 ILP 调度模型 + 所有厂商的 ISA 编码器.
