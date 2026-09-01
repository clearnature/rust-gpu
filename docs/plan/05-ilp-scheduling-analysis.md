# ILP 调度模型深度分析

> 分析目标: /data/rtl-sdr/sass-assembler/src/sass/ilp_model.h (654 行)

## 什么是 ILP 调度

ILP = Instruction Level Parallelism (指令级并行). 目标是重新排列指令顺序, 让尽可能多的指令同时在硬件上执行.

GPU 是 in-order (顺序执行) 的, 没有 CPU 的乱序重排能力, 所以 ILP 调度完全靠编译器/汇编器完成.

## 这个 ILP 模型实际做了什么

### 核心逻辑 (~50 行)

```cpp
bool can_issue(inst, cycle) {
    // 只检查一种冒险: RAW (Read After Write) 数据依赖
    for (each source register) {
        if (cycle - last_writer[reg] < latency) return false;
    }
    return true;
}

void schedule(insts) {
    while (issued < total) {
        for (each instruction)
            if (can_issue(inst, cycle)) issue(inst, cycle);  // 线性扫描
        cycle++;
    }
}
```

**只做了两件事:**
1. RAW 数据冒险检测 — 源寄存器的上次写入是否已完成
2. issue_width 限制 — 每周期最多发射几条

**不是完整的 ILP 调度模型**, 是最基础的 latency-aware list scheduling.

## 硬件模型层 — 真正有价值的通用部分

### HardwareModel 结构体

```cpp
struct HardwareModel {
    int issue_width;      // 每周期发射几条
    int retire_width;     // 每周期退休几条
    bool out_of_order;    // GPU=false, CPU=true
    int rob_size;         // Reorder Buffer (仅 CPU)

    int fma_units, fp_ports, int_ports, ld_ports, st_ports;

    int l1_latency, l2_latency, l3_latency, dram_latency;
    int warps_per_sm, warp_size, sm_count;
    int scoreboard_regs, max_active_warps;
    bool zero_overhead_switch;
    bool has_tensor_cores;

    vector<LatencyEntry> latency_table;
};
```

### InsnClass 指令分类

```cpp
enum class InsnClass {
    FP32_FMA, FP32_ADD, INT_ADD, INT_MUL,
    LD_GLOBAL, LD_SHARED, ST_GLOBAL,
    BRANCH, BARRIER, ...
    FP16_VALU, FP16_DENSE_MATRIX, FP16_SPARSE_MATRIX,  // RDNA4 三阶
};
```

所有架构的指令都映射到同一套 InsnClass, 调度器只看 InsnClass 不看具体 opcode.

### 已覆盖的 7 个硬件架构

| 架构 | 类型 | 数据来源 | 特殊建模 |
|------|------|---------|---------|
| Pascal GP106 | GPU | ptx_gp106 实测 | 32-bank shared memory, 64 warps/SM |
| Volta GV100 | GPU | denvdis | +Tensor Cores |
| Ampere GA100 | GPU | denvdis | +FP16 native, 40MB L2 |
| Hopper GH100 | GPU | DeepGEMM | +WGMMA, TMA, FP8 |
| RDNA4 gfx1200 | GPU | swmmac 实测 | 三阶 FP16 算力路由 |
| Broadwell-EP | CPU | cpu_probe 实测 | OoO, ROB=192 |
| Zen4/Zen5 | CPU | AMD PPR | OoO, ROB=320/448, AVX-512 |
| Cortex-A78 | ARM | 公开数据 | OoO, ROB=160, NEON |

### RDNA4 三阶 FP16 建模

```cpp
// Level 1: 传统矢量 — 25.6 TFLOPS
{InsnClass::FP16_VALU,          4,  2, "v_pk_fma_f16"},

// Level 2: 稠密矩阵 — 103 TFLOPS  
{InsnClass::FP16_DENSE_MATRIX,  26, 16, "v_swmmac 16x16x32"},

// Level 3: 稀疏矩阵 — 205 TFLOPS
{InsnClass::FP16_SPARSE_MATRIX, 26, 32, "v_swmmac + 2:4 sparsity"},
```

## 通用性保证机制

**机制 1: 指令分类抽象 (InsnClass)**
同一套调度算法, 不同延迟参数.

**机制 2: 模型注册表 + 运行时选择**
```cpp
const HardwareModel& m = ilp::model_for("sm_61");
const HardwareModel& m = ilp::model_for("gfx1200");
const HardwareModel& m = ilp::model_for("zen4");
```

**机制 3: 延迟表驱动, 非硬编码**
加新硬件只需添加一个 xxx_model() 函数, 调度器代码零修改.

## 缺失分析 — 真正的 ILP 调度应该包含什么

### 缺失 1: 端口冲突检测 (声明了但没用)

```cpp
std::vector<int> port_busy;  // 声明了
// LatencyEntry 有 port_mask
// 但 can_issue() 从不检查端口!
```

### 缺失 2: 只处理 RAW, 不处理 WAR/WAW

```cpp
// 示例:
FADD R1, R2, R3     // cycle 0: 写 R1
FMUL R4, R1, R5     // cycle 0: 读 R1 (RAW, 正确等待)
FADD R1, R6, R7     // cycle 1: 写 R1 (WAR: 覆盖 R1) — 未处理
```

### 缺失 3: 没有调度优先级

线性扫描, 按指令在序列中的顺序尝试发射. 真正的 ILP 调度应该计算每条指令到关键路径末端的距离, 选最紧急的.

### 缺失 4: GPU 特有问题完全没处理

- Shared memory bank conflict
- Warp divergence 感知调度
- Occupancy 感知 (寄存器数 vs wave 数)
- SWMMAC EXEC lock (RDNA4 稀疏指令)

### 缺失 5: 高级调度策略

- Software pipelining (循环模调度)
- Trace scheduling (热路径优先)
- Region scheduling (跨基本块)
- If-conversion (分支→谓词)
- Instruction clustering
- Prefetch 插入

## 与 LLVM MachineScheduler / nvcc ptxas 对比

| 维度 | 这个模型 | LLVM MachineScheduler | nvcc ptxas |
|------|---------|----------------------|------------|
| RAW 检测 | ✅ | ✅ | ✅ |
| WAR/WAW | ❌ | ✅ | ✅ |
| 端口冲突 | ❌ (声明了) | ✅ (Resource Table) | ✅ |
| 调度优先级 | ❌ (线性扫描) | ✅ (SUnit depth/height) | ✅ |
| 跨基本块 | ❌ | ✅ (Region-based) | ✅ |
| Software pipelining | ❌ | ✅ | ✅ |
| 寄存器压力 | ❌ | ✅ (RegPressure tracker) | ✅ |
| Bank conflict | ❌ | N/A (GPU 特有) | ✅ |
| 真实性 | 原型级 | 生产级 | 生产级 |

## 结论

这个 ILP 模型实际上是:

1. **一个很好的硬件参数数据库** — 7 个架构的延迟表、流水线参数、内存层次, 这部分真实且有价值
2. **一个最基础的 latency-aware scheduler** — 只做 RAW 检测 + issue_width, 约 50 行核心代码
3. **不是完整的 ILP 调度框架** — 缺少端口冲突、WAR/WAW、调度优先级、跨基本块、寄存器压力等所有高级特性

**通用性保证的是硬件模型层** (HardwareModel 结构体 + InsnClass 分类), 不是调度算法本身. 调度算法是简化的, 适用于 GPU (in-order), 但对 CPU (OoO) 不够用.
