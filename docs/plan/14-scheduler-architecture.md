# 完整调度器架构分析

> 调度器是一个分层系统，每一层解决不同粒度的调度问题

## 调度层次总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 5: 图级调度 (Graph Scheduling)                           │
│  "哪些 kernel 先执行？"                                         │
│  ─────────────────────────────────────                          │
│  tinygrad: Kahn 算法 (BFS + in-degree), RAW/WAR AFTER 链      │
│  t0-gpu:   无显式图调度 (单 kernel 执行)                        │
│  ROCm:     CLR stream worker (命令队列顺序执行)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Level 4: 高层优化调度 (High-Level Optimization)                │
│  "这个 kernel 用什么 tile/向量化/分组策略？"                      │
│  ─────────────────────────────────────                          │
│  tinygrad: BEAM search (50+ 动作组合, 实测选最快)               │
│  t0-gpu:   CostModel (Roofline + K-loop 分析, 400+ 候选穷举)   │
│  sass:     无 (手写 tile 配置)                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Level 3: Kernel 模板调度 (Kernel Template Scheduling)          │
│  "用哪个预编译模板？"                                           │
│  ─────────────────────────────────────                          │
│  t0-gpu:   Schedule trait (GFX1100Schedule / AutoGemmSchedule)  │
│            → build_gemm_forward() → T0Kernel                    │
│  tinygrad: hand_coded_optimizations() (启发式)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Level 2: 指令调度 (Instruction Scheduling)                     │
│  "指令按什么顺序发射？"                                         │
│  ─────────────────────────────────────                          │
│  t0-gpu:   4 阶段优化 pass (Phase A/B/C/D)                     │
│            Phase D: 软件流水 + Pingpong + 指令重排               │
│  tinygrad: 优先级 toposort (LOAD 早, STORE 晚)                 │
│  sass:     ILPSchedulerV2 (RAW 检测 + issue_width)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Level 1: 硬件调度 (Hardware Scheduling)                        │
│  "GPU 硬件自己怎么调度 wave/warp？"                              │
│  ─────────────────────────────────────                          │
│  AMD: CU 上的 wave scheduler (硬件, 4-8 waves/SIMD)            │
│       GFX1200: MES v2 firmware 调度                             │
│  NVIDIA: GigaThread engine (硬件 warp scheduler)               │
│  t0-gpu 通过 s_setprio 影响硬件调度优先级                       │
└─────────────────────────────────────────────────────────────────┘
```

## Level 5: 图级调度

### tinygrad (唯一有显式图调度的)

```python
# schedule/__init__.py — create_schedule()
# Kahn 算法 (BFS + in-degree)
# 依赖类型:
#   RAW (Read After Write): kernel B 读 kernel A 写的 buffer
#   WAR (Write After Read): kernel B 写 kernel A 读的 buffer

def create_schedule(sink):
    # 1. 建立 AFTER 链 (buffer 状态机)
    # 2. 计算每个 kernel 的 in-degree
    # 3. BFS 拓扑排序
    # 4. 检测循环 → RuntimeError
    return linear_schedule
```

**调度粒度**: kernel 级 (一个 matmul 是一个 kernel, 一个 relu 是一个 kernel)
**调度目标**: 满足数据依赖的前提下, 最小化 kernel 数量 (通过融合)
**调度约束**: RAW/WAR 依赖, buffer 生命周期

### t0-gpu

**无显式图调度**。ignis 的操作是命令式的——每层调用 dispatch() 并等待完成。

```rust
// ignis/nn/transformer.rs — forward_simple()
let h = ops::rmsnorm::rmsnorm(x, ...);    // dispatch 1 + wait
let q = self.wq.forward(&h)?;              // dispatch 2 + wait
let attn_out = ops::bf16_matmul::matmul(...); // dispatch 3 + wait
// ...
```

**唯一的优化**: `DispatchQueue` (prefill_dispatch.rs) 批量提交多个 dispatch, 最后一次 flush 同步:
```
32 layers × 4 kernels = 128 dispatches
Per-dispatch wait: 128 × 10μs = 1.3ms
Queued flush: 1 × 50μs = 0.05ms  → 25× 减少同步开销
```

## Level 4: 高层优化调度

### t0-gpu CostModel (cost_model.rs, 1368 行)

```rust
// 穷举搜索 400+ 候选配置
fn auto_schedule_gemm(m: u32, n: u32, k: u32) -> Vec<GemmCandidate> {
    for tile_m in [32, 64, 128] {
        for tile_n in [64, 128] {
            for tile_k in [16, 32, 48, 64] {
                for waves in [2, 4, 8] {
                    for split_k in [1, 2, 4, 8] {
                        for wgp in [false, true] {
                            for swap_grid in [false, true] {
                                for lds_pad in [0, 4, 8] {
                                    // 评估代价
                                    let score = evaluate(m, n, k, config);
                                    candidates.push(score);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    candidates.sort_by_score();
}
```

**评估模型**:
```
Compute ceiling = K-loop cycles × effective_iters → TFLOPS
Memory ceiling  = Roofline(occupancy, bandwidth, L2 hit rate)
Effective TFLOPS = min(compute, memory, peak=123T)
```

**关键参数**:
- VGPR 估算: acc + frag_a + frag_b + gmem + temps
- Occupancy: VGPR 数 → waves/SIMD (<=64→4, <=85→3, <=128→2, <=256→1)
- L2 命中率: tile 面积相关 (0.1 ~ 0.7)

### tinygrad BEAM Search (codegen/opt/search.py, 179 行)

```python
# 50+ 种优化动作
actions = [Opt(UPCAST, axis, amt) for amt in [0,2,3,4,5,7] for axis in range(8)]
actions += [Opt(UNROLL, axis, amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(LOCAL, axis, amt) for amt in [2,3,4,8,13,16,29] for axis in range(6)]
actions += [Opt(GROUPTOP, axis, amt) for amt in [13,16,28,29,32,49,64,256] ...]
actions += [Opt(TC, axis, (-1, tc_opt, use_tc)) for axis in range(9)]
actions += [Opt(SWAP, a0, a1) for a0 in range(5) for a1 in range(a0+1, 5)]
actions += [Opt(THREAD, axis, amt) for amt in [2,3,4,5,8,12,16,24,32,64] ...]

# 暴力搜索
for candidate in all_combinations(actions):
    kernel = apply_opts(base_kernel, candidate)
    time = benchmark(kernel, real_data)
    if time < best_time: best = candidate

# 磁盘缓存 (diskcache)
```

**与 t0-gpu CostModel 的区别**:

| | t0-gpu CostModel | tinygrad BEAM |
|---|---|---|
| 搜索方式 | 分析模型 (Roofline) | 暴力实测 |
| 搜索空间 | 400+ (穷举) | 10^6+ (组合爆炸) |
| 搜索速度 | ~1ms (CPU 计算) | ~1-10s (GPU 实测) |
| 准确性 | 依赖模型精度 | 实测最准 |
| 通用性 | 只有 GEMM | 任意 kernel |
| 缓存 | ~/.t0_autotune/ | diskcache |

## Level 3: Kernel 模板调度

### t0-gpu Schedule Trait (schedule.rs)

```rust
pub trait Schedule {
    fn gemm_tile_mn(&self) -> (usize, usize);  // (32, 64) for GFX1100
    fn gemm_tile_k(&self) -> usize;             // 16
    fn use_wmma(&self) -> bool;                 // true
    fn wmma_format(&self) -> WmmaFormat;        // BF16_F32
    fn workgroup_size(&self) -> (u16, u16, u16); // (64, 1, 1)
    fn lds_budget(&self) -> u32;                // 65536
    // ...
}
```

**两个实现:**
- `GFX1100Schedule` — 手写参数 (tile=32x64, k=16, 2 waves)
- `AutoGemmSchedule` — CostModel 自动选择

**调度流程:**
```
Schedule trait + Template 函数 → T0Kernel → compile() → ELF
     ↑                              ↑
  硬件参数                        指令序列
```

## Level 2: 指令调度 (核心)

### t0-gpu 4 阶段优化 (opt_passes.rs, 1536 行)

```rust
fn optimize(ops: Vec<Op>, opt_level: u32) -> Vec<Op> {
    // Phase A: SSA 级优化
    constant_folding();           // fadd 0, x → x
    algebraic_simplification();   // fmul x, 2 → fadd x, x
    copy_propagation();           // mov a, b; use a → use b
    cse_with_dominator_tree();    // 公共子表达式消除 (跨基本块)
    instruction_combining();      // fmul + fadd → fma
    licm();                       // 循环不变量外提

    // Phase B: 循环优化
    loop_unrolling();             // 展开因子: body≤16→4x, ≤48→2x
    strength_reduction();         // 循环内乘法 → 累加加法

    // Phase C: 迭代优化 (最多 5 轮)
    for _ in 0..5 {
        algebraic_simplification();
        dead_code_elimination();  // 含循环活跃分析
    }

    // Phase D: 后端优化
    waitcnt_optimization();       // 合并/消除 waitcnt
    load_store_coalescing();      // 相邻 load → dwordx4
    software_pipelining();        // 迭代 N 计算与 N+1 load 重叠
    post_regalloc_scheduling();   // 寄存器分配后指令重排
    pingpong_schedule();          // s_setprio 交错 WMMA 和内存
}
```

### t0-gpu 软件流水线 (opt_passes.rs:1026)

```rust
fn software_pipeline(ops: Vec<Op>) -> (Vec<Op>, usize) {
    // 三阶段: Schedule → Lower → Pipeline
    // 重叠 iteration N 的 compute 和 iteration N+1 的 load
    //
    // 未流水:
    //   iter 0: [LOAD_A][LOAD_B][WMMA][STORE]
    //   iter 1: [LOAD_A][LOAD_B][WMMA][STORE]
    //
    // 流水后:
    //   iter 0: [LOAD_A0][LOAD_B0][WMMA0]
    //   iter 1: [LOAD_A1][LOAD_B1][WMMA1][STORE0]
    //   iter 2: [LOAD_A2][LOAD_B2][WMMA2][STORE1]
    //
    // 禁用条件: WMMA 循环 (已手写流水)
}
```

### t0-gpu Pingpong 调度 (ssa_ir.rs)

```rust
fn pingpong_schedule(ops: &mut Vec<Op>) -> usize {
    // 基于 Triton 的 BlockPingpong.cpp
    // 在 WMMA 和内存集群之间插入 s_setprio
    //
    // 效果: 让 WMMA wave 和 memory wave 交替获得高优先级
    // GPU 硬件 scheduler 会优先执行高优先级 wave
    //
    // s_setprio 1  → WMMA 集群 (计算密集)
    // s_setprio 0  → memory 集群 (访存密集)
    // 交替执行, 隐藏内存延迟
}
```

### tinygrad 线性化器 (codegen/late/linearizer.py, 96 行)

```python
def linearize(sink: UOp) -> list[UOp]:
    # 静态优先级 toposort
    priorities = {
        PARAM: -20,      # 参数最先
        BUFFER: -17/-18, # buffer 次之
        LOAD: -1,        # load 尽早 (隐藏延迟)
        STORE: 1,        # store 尽晚
        RANGE: 5,        # 循环头晚放
        END: -5,
        default: 0,
    }
    # 按优先级排序, 同时满足 toposort 约束
    # 使用 heap 实现
```

**不是 ILP 调度** — 不看延迟, 不看端口冲突, 只看操作类型。

### sass-assembler ILP Scheduler (ilp_model.h)

```cpp
class ILPSchedulerV2 {
    void schedule(vector<Instruction>& insts) {
        int cycle = 0;
        while (issued < total) {
            for (each instruction) {
                if (can_issue(inst, cycle)) {
                    issue(inst, cycle);
                    slot_issued++;
                }
            }
            cycle++;
        }
    }

    bool can_issue(inst, cycle) {
        // 只检查 RAW 数据冒险
        for (each source register) {
            int last_write = last_writer[reg];
            int latency = model->latency_for(classify(opcode));
            if (cycle - last_write < latency) return false;
        }
        return true;
    }
};
```

## Level 1: 硬件调度

### AMD GPU Wave Scheduler

```
CU (Compute Unit) 内部:
├── SIMD0: 4 waves (或 2/1, 取决于 VGPR 使用量)
│   ├── Wave 0: [VMEM load pending...] [VALU compute]
│   ├── Wave 1: [VALU compute] [WMMA pending...]
│   ├── Wave 2: [LDS load] [SALU compute]
│   └── Wave 3: [等待 VMEM...] [VMEM 结果就绪 → 继续]
├── SIMD1: 4 waves
│   └── ...
└── Wave Scheduler: 每 cycle 选一个就绪 wave 发射指令
    优先级: s_setprio 可影响
    就绪条件: waitcnt 满足, 寄存器无 RAW 冲突
```

**t0-gpu 如何影响硬件调度:**
- `s_setprio 1/0` — Pingpong 交替 WMMA 和内存的优先级
- `wait_vmcnt(0)` — 等待所有 VMEM 完成
- `wait_lgkmcnt(0)` — 等待 LDS/SMEM 完成
- `s_barrier` — 工作组同步

### NVIDIA Warp Scheduler

```
SM (Streaming Multiprocessor) 内部:
├── Warp Scheduler 0: 管理 ~16 warps
│   ├── 每 cycle 选就绪 warp
│   ├── 发射 1-2 条指令 (Volta+ 双发射)
│   └── 切换零开销 (所有 warp 状态常驻寄存器)
├── Warp Scheduler 1: 管理 ~16 warps
│   └── ...
└── 调度策略: GTO (Greedy Then Oldest) 或 RR (Round-Robin)
```

## 调度器对比总表

| 调度层次 | t0-gpu | tinygrad | sass-assembler | ROCm |
|---------|--------|----------|---------------|------|
| **图级** (kernel 顺序) | ❌ 命令式 | ✅ Kahn BFS | ❌ | CLR stream |
| **高层优化** (tile/向量) | ✅ CostModel 穷举 | ✅ BEAM search | ❌ | rocBLAS Tensile |
| **模板选择** | ✅ Schedule trait | ✅ hand_coded | ❌ | hipBLASLt |
| **指令重排** | ✅ Phase D | ❌ 交给 LLVM | ⚠️ 基础 RAW | LLVM/ptxas |
| **软件流水** | ✅ 3 阶段 | ❌ | ❌ | LLVM |
| **Pingpong** | ✅ s_setprio | ❌ | ❌ | Triton |
| **寄存器压力** | ✅ SSA regalloc + spill | ✅ linear scan + spill | ⚠️ 基础 | LLVM |
| **硬件调度影响** | ✅ s_setprio/waitcnt | ❌ | ❌ | ROCm |

## 完整调度路径示例

### t0-gpu 执行一次 GEMM 的完整调度路径

```
1. [Level 4] CostModel.evaluate(M=4096, N=4096, K=512)
   → 穷举 400+ 候选
   → 选: tile=128x128, k=32, waves=8, split_k=1, wgp=true
   → 预估: 78.5 TFLOPS, 瓶颈: compute

2. [Level 3] AutoGemmSchedule → build_gemm_forward()
   → 生成 T0Kernel (K-loop + WMMA + store)
   → 13 条 Op: global_load × N, wmma × N, global_store × N

3. [Level 2] optimize(ops, level=4)
   Phase A: CSE 消除重复 load, FMA 合成
   Phase B: 循环展开 4x
   Phase C: 迭代 DCE
   Phase D: 
     - waitcnt 优化: 合并多余 wait
     - 软件流水: 重叠 load 和 compute
     - Pingpong: 插入 s_setprio
     - 指令重排: 把 load 提前

4. [Level 2] ssa_regalloc::allocate()
   → VGPR 分配 (目标 ≤254)
   → 如果溢出: 插入 LDS spill/reload
   → WMMA 8-aligned 寄存器组

5. [Level 1] dispatch() → KFD AQL packet → doorbell
   → GPU CU wave scheduler 接管
   → 8 waves/SIMD 交替执行
   → s_setprio 影响 wave 优先级
   → VMEM/VALU 流水线自动重叠
```

### tinygrad 执行一次 GEMM 的完整调度路径

```
1. [Level 5] create_schedule()
   → 识别 GEMM 为一个独立 kernel
   → 检查 RAW/WAR 依赖
   → 拓扑排序确定执行顺序

2. [Level 4] BEAM search (search.py)
   → 尝试 100+ 种 (UPCAST, UNROLL, LOCAL, TC, SWAP) 组合
   → 每种实测 3 次, 取中位数
   → 选最快的配置
   → 缓存到 disk

3. [Level 3] hand_coded_optimizations()
   → 尝试 tensor core (WMMA)
   → 尝试 matvec 优化
   → 尝试 group reduce

4. [Level 2] linearize()
   → 优先级 toposort: LOAD(-1) 早, STORE(+1) 晚
   → 不做 ILP 调度

5. [Level 2] regalloc (linear scan)
   → 分配寄存器 + 溢出

6. 编译: HIPRenderer → HIP C++ → comgr → HSACO
   或: AMDLLVMRenderer → LLVM IR → llc → HSACO
   (ILP 调度由 LLVM 完成)

7. dispatch → KFD AQL → doorbell
```

## t0-gpu 指令调度器详细分解 (16 个调度器)

子代理深度扫描发现 t0-gpu 有 **6 个独立的指令级调度器**，比初步分析发现的更多：

### sched-1: schedule_mach_func (DAG 级指令重排)
- **文件**: `src/t0/ssa_ir.rs:522-767` (~245 行)
- **算法**: 两阶段 list scheduling on SSA MachInsts
- **Phase 1**: 延迟隐藏重排 — 检测 VMEM/LDS load 集群，把不依赖 load 的计算指令移到 load 和 waitcnt 之间
- **Phase 2**: 寄存器压力感知 — 当活跃 MVal > 96 时，优先发射消耗更多即将死亡值的指令
- **依赖处理**: 完整 RAW/WAW/WAR (通过 SSA MVal 集合) + VCC/SCC/EXEC 隐式寄存器
- **运行时机**: regalloc 之前 (SSA 虚拟寄存器)

### sched-2: post_regalloc_schedule (物理寄存器级 peephole)
- **文件**: `src/t0/opt_passes.rs:803-993` (~190 行)
- **算法**: load→fill→wait peephole 重排
- **关键**: 物理寄存器允许精确依赖分析 (SSA 虚拟寄存器无法捕获两个不同虚拟寄存器共享同一物理 VReg 的情况)
- **安全**: 跳过有 BufferLoad/BufferStore 的基本块 (手写调度的 GEMM K-loop)
- **运行时机**: regalloc 之后

### sched-3: software_pipeline (循环级软件流水)
- **文件**: `src/t0/opt_passes.rs:1026-1164` (~138 行)
- **算法**: 3 阶段变换 (prologue/main/epilogue)
- **效果**: 迭代 N 的 compute 与迭代 N+1 的 load 重叠
- **约束**: 禁用于 WMMA 循环 (已手写流水)、含 LDS/barrier 的循环

### sched-4: pingpong_schedule (wave 优先级调度)
- **文件**: `src/t0/ssa_ir.rs:3452-3506` (~54 行)
- **算法**: WMMA 集群前插入 s_setprio(1)，集群后插入 s_setprio(0)
- **来源**: Triton 的 BlockPingpong.cpp
- **运行时机**: regalloc 之后 (s_setprio 无 VReg 依赖)

### sched-5: waitcnt_optimization (等待计数优化)
- **合并/消除多余的 wait_vmcnt/wait_lgkmcnt

### sched-6: load_store_coalescing (内存访问合并)
- **相邻 load → dwordx4 向量化

## sass-assembler 流形调度器 (独特设计)

### ManifoldScheduler (manifold_scheduler.cpp, 121 行)

**算法**: 4320D 能量场 + Yamabe 流平滑 — 微分几何方法

```
1. 构建 4320D 能量场:
   每条指令 → phase = complexity_score × 0.618
   → slot = manifold_slot(seed, complexity, phase) (哈希到 4320 个槽)
   → energy[slot] += complexity_score

2. 计算 Laplacian 曲率:
   curvature = laplacian(energy)

3. Yamabe 流平滑:
   smoothed = yamabe_smooth(energy, yamabe_flow(curvature))

4. 提取优化序列:
   按 smoothed energy 降序排列指令

4.5 三层 FP16 电流平滑 (RDNA4 专有):
   检测 tier 转换 (VALU → Dense → Sparse)
   在转换点注入 SALU NOP:
   Sparse→Dense: 4 NOP (AGU 卸载)
   Dense→Sparse: 2 NOP + EXEC lock
   其他: 1 NOP
   → 防止 di/dt 瞬态电流触发 VRM 谐振和 PLL 抖动
```

**关键**: 第 4.5 步的 NOP 注入是**硬件电气级约束** — 不是性能优化，是防止 GPU 物理层出问题。

## 完整调度器清单 (16 个)

| # | 调度器 | 项目 | 文件 | 行数 | 算法 | 粒度 |
|---|--------|------|------|------|------|------|
| 1 | Schedule trait | t0-gpu | schedule.rs | 530 | 参数模板 | Tile |
| 2 | AutoGemmSchedule | t0-gpu | cost_model.rs | 1369 | 穷举+Roofline | Tile |
| 3 | GemmTuner | t0-gpu | auto_gemm.rs | 611 | 实测验证穷举 | Tile |
| 4 | DispatchQueue | t0-gpu | prefill_dispatch.rs | 233 | 批量提交+flush | Kernel |
| 5 | PrefillSpecCache | t0-gpu | prefill_spec_cache.rs | 218 | HashMap 缓存 | Config |
| 6 | schedule_mach_func | t0-gpu | ssa_ir.rs:522-767 | ~245 | **两阶段 DAG 调度** | 指令 |
| 7 | post_regalloc_schedule | t0-gpu | opt_passes.rs:803-993 | ~190 | **物理寄存器 peephole** | 指令 |
| 8 | software_pipeline | t0-gpu | opt_passes.rs:1026-1164 | ~138 | **3 阶段软件流水** | 循环 |
| 9 | pingpong_schedule | t0-gpu | ssa_ir.rs:3452-3506 | ~54 | **s_setprio 优先级** | 指令 |
| 10 | insn_latency + K-loop | t0-gpu | insn_latency.rs + cost_model.rs | 528+214 | ASAP 关键路径 | 指令 |
| 11 | create_schedule | tinygrad | schedule/__init__.py | 209 | Kahn toposort | 图 |
| 12 | beam_search | tinygrad | codegen/opt/search.py | 178 | BEAM 搜索 | Tile |
| 13 | Scheduler (postrange) | tinygrad | codegen/opt/postrange.py | 352 | 轴操作 | 循环轴 |
| 14 | linearize | tinygrad | codegen/late/linearizer.py | 95 | 优先级 toposort | UOp |
| 15 | ManifoldScheduler | sass | manifold_scheduler.cpp | 121 | **4320D 能量场+Yamabe 流** | 指令 |
| 16 | ILPSchedulerV2 | sass | ilp_model.h | 654 | 周期精确记分板 | 指令 |

## 依赖处理对比

| 调度器 | RAW | WAR | WAW |
|--------|-----|-----|-----|
| t0-gpu ssa_ir (MachSSA) | ✅ SSA 值集 | ✅ SSA 值集 | ✅ SSA def |
| t0-gpu post_regalloc | ✅ 物理 VReg/SReg | ✅ 物理 VReg/SReg | ✅ 物理 def |
| tinygrad create_schedule | ✅ AFTER 节点 | ✅ AFTER supersede | ✅ AFTER 链 |
| tinygrad linearize | ✅ toposort (隐式) | ❌ | ❌ |
| sass ILPSchedulerV2 | ✅ 记分板 (last_writer) | ❌ | ❌ |
| sass ManifoldScheduler | ❌ (能量场方法) | ❌ | ❌ |

## 独特发现

### 1. t0-gpu 是唯一做两遍指令调度的
- **遍 1** (sched-1): regalloc 前，SSA 虚拟寄存器，延迟隐藏 + 压力感知
- **遍 2** (sched-2): regalloc 后，物理寄存器，精确依赖分析

### 2. sass 流形调度器有硬件电气约束
- di/dt NOP 注入不是性能优化，是防止 GPU VRM 谐振
- 三层 FP16 (VALU/Dense/Sparse) 转换需要不同的 NOP 数量

### 3. t0-gpu 的 GEMM K-loop 是手写的，跳过所有调度
- `set_skip_optimize(true)` — GEMM kernel 跳过编译器优化
- 手写的 LDS double-buffer + WMMA 调度已经是最优的
- BufferLoad/BufferStore 块被 post_regalloc_schedule 跳过

1. **t0-gpu 的指令调度比 tinygrad 强** — 有软件流水、Pingpong、waitcnt 优化, tinygrad 把这些全交给 LLVM
2. **tinygrad 的高层优化比 t0-gpu 强** — BEAM search 覆盖 50+ 动作, t0-gpu 的 CostModel 只覆盖 GEMM tile
3. **t0-gpu 是唯一做 Pingpong 调度的** — s_setprio 交替 wave 优先级, 源自 Triton
4. **sass-assembler 的 ILP 调度是最弱的** — 只有 RAW 检测, 没有端口冲突/优先级/跨基本块
5. **三者都不做完整的 ILP 调度** — 这是 LLVM/ptxas 的领域
