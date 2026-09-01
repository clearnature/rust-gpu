# RDNA4 ISA 结构化参考（T0 编译器相关）

> 来源: AMD "RDNA4" Instruction Set Architecture Reference Guide (7-April-2025, 707pp)
> 提取方式: pdftotext + 结构化整理
> 日期: 2026-08-23

---

## 1. 目录结构

| 章节 | 标题 | 页码 | T0 相关度 |
|------|------|------|----------|
| 1 | Introduction | 5 | 低 |
| 2 | Shader Concepts (Wave32/64, Work-groups) | 9 | 中 |
| 3 | Program State (SGPR, VGPR, LDS, EXEC) | 13 | **高** |
| 4 | Shader Instruction Set | 38 | 低 |
| 5 | Program Flow Control | 43 | 中 |
| 6 | Scalar ALU Operations | 58 | **高** |
| 7 | Vector ALU Operations | 66 | **高** |
| 8 | Scalar Memory Operations | 97 | 中 |
| 9 | Vector Memory Buffer Instructions | 102 | **高** |
| 10 | Vector Memory Image Instructions | 116 | 低 |
| 11 | Global, Scratch and Flat Address Space | 135 | 中 |
| 12 | Local Data Share Operations | 144 | **高** |
| 13 | Export | 156 | 低 |
| 14 | GDS/Append/Consume | 158 | 低 |
| 15 | Microcode Formats | 162 | **高** |
| 16 | Instruction Detail | 206 | **高** |

---

## 2. 关键寄存器与状态

### 2.1 SGPR (Scalar General Purpose Register)
- **数量**: 106 可用（TotalNumSgprs: 106）
- **分配**: 按需，s63 保留为 MUBUF soffset 零寄存器（GFX1200 特有）
- **对齐**: SBASE 必须偶数对齐（8字节地址）

### 2.2 VGPR (Vector General Purpose Register)
- **数量**: 256/SIMD（实测确认，LLVM 封顶 256，超出溢出到 LDS）
- **双波驻留**: ≤128 VGPR → 2 waves/SIMD（256/2=128，与 rtl-sdr docs 一致）
- **单波**: ≤256 VGPR → 1 wave/SIMD

### 2.3 EXEC (Execute Mask)
- **Wave32**: 32-bit（低 32 位有效，高 32 位忽略）
- **初始化**: 硬件在 wave 启动时设为全 1
- **VCC**: 32-bit 标量比较结果（Wave32: 低 32 位有效）

### 2.4 LDS (Local Data Share)
- **每 WGP**: 64 KB
- **模式**: CU 模式（每 CU 独立）vs WGP 模式（跨 CU 共享）
- **初始化**: 硬件不保证清零（未初始化内容未定义）

---

## 3. 指令集分类（T0 相关）

### 3.1 Scalar ALU (SALU)
| 格式 | 指令示例 | T0 使用 |
|------|---------|--------|
| SOP2 | S_ADD, S_AND, S_OR, S_LSHL, S_LSHR | ✅ 大量使用 |
| SOPK | S_CMP, S_MOVK | ✅ |
| SOP1 | S_MOV, S_NOT | ✅ |
| SOPC | S_CMP_EQ, S_CMP_GE, S_CMP_LT | ✅ 条件分支 |
| SOPP | S_BARRIER, S_WAITCNT, S_SETPRIO | ✅ |

### 3.2 Vector ALU (VALU)
| 格式 | 指令示例 | T0 使用 |
|------|---------|--------|
| VOP2 | V_ADD, V_MUL, V_FMA, V_AND, V_OR | ✅ |
| VOP1 | V_MOV, V_READFIRSTLANE, V_MBCNT | ✅ |
| VOPC | V_CMP_EQ, V_CMP_LT, V_CMP_GE | ✅ |
| VOP3P | **V_PK_FMA_F16, V_PK_ADD_F16, V_DOT2_F32_F16** | 部分使用 |
| VOP3 | V_FMA_F32, V_ADD_CO_U32 | ✅ |
| VOPD | Dual Issue VALU | 未使用 |

### 3.3 WMMA (Matrix Operations) — **T0 核心**
| 指令 | 输入 | 输出 | K | T0 支持 |
|------|------|------|---|--------|
| **V_WMMA_F32_16X16X16_BF16** | BF16×BF16 | F32 | 16 | ✅ 主用 |
| V_WMMA_F32_16X16X16_F16 | FP16×FP16 | F32 | 16 | ✅ |
| V_WMMA_BF16_16X16X16_BF16 | BF16×BF16 | BF16 | 16 | ✅ |
| V_WMMA_F16_16X16X16_F16 | FP16×FP16 | FP16 | 16 | ✅ |
| V_WMMA_I32_16X16X16_IU8 | INT8×INT8 | I32 | 16 | ✅ |
| V_WMMA_I32_16X16X16_IU4 | INT4×INT4 | I32 | 16 | ✅ |
| V_WMMA_I32_16X16X32_IU4 | INT4×INT4 | I32 | 32 | ❌ |
| V_WMMA_F32_16X16X16_FP8_FP8 | FP8×FP8 | F32 | 16 | ✅ |
| V_WMMA_F32_16X16X16_BF8_BF8 | BF8×BF8 | F32 | 16 | ✅ |
| V_WMMA_F32_16X16X16_FP8_BF8 | FP8×BF8 | F32 | 16 | ✅ |
| V_WMMA_F32_16X16X16_BF8_FP8 | BF8×FP8 | F32 | 16 | ✅ |

### 3.4 SWMMAC (Sparse Matrix) — **T0 未支持**
| 指令 | 输入 | 输出 | K | 稀疏 |
|------|------|------|---|------|
| V_SWMMAC_I32_16X16X64_IU4 | INT4 | I32 | 64 | 2:4 |
| V_SWMMAC_I32_16X16X32_IU8 | INT8 | I32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_BF16 | BF16 | F32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_F16 | FP16 | F32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_FP8_FP8 | FP8 | F32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_BF8_BF8 | BF8 | F32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_FP8_BF8 | FP8×BF8 | F32 | 32 | 2:4 |
| V_SWMMAC_F32_16X16X32_BF8_FP8 | BF8×FP8 | F32 | 32 | 2:4 |

### 3.5 Memory Operations
| 格式 | 指令示例 | T0 使用 |
|------|---------|--------|
| SMEM | S_LOAD, S_STORE, S_BUFFER_LOAD | ✅ kernarg 读取 |
| VBUFFER | BUFFER_LOAD, BUFFER_STORE | ✅ GMEM 访问 |
| VDS | DS_LOAD, DS_STORE, DS_LOAD_B128 | ✅ LDS 访问 |
| FLAT/SCRATCH/GLOBAL | GLOBAL_LOAD, GLOBAL_STORE | ✅ |

---

## 4. WMMA 关键约束（来自 ISA 文档 §7.12.1）

1. **EXEC 必须全 1**: WMMA 不支持部分 EXEC 掩码（HWXDL silent drop）
2. **ACC 累加器**: 8 个连续 VGPR（F32 输出）或 4 个（BF16/FP16 输出）
3. **A/B 片段**: GFX1200 上各 4 个 VGPR（1×ds_load_b128），GFX1100 各 8 个（2×ds_load_b128）
4. **C/D 片段**: 与 ACC 同寄存器（原地累加）
5. **数据类型混合**: FP8/BF8 变体支持混合输入（FP8×BF8、BF8×FP8）
6. **K 维度**: 16x16x16（标准）或 16x16x32（INT4 K=32 变体，T0 未支持）
7. **OPSEL/NEG**: FP8/BF8 变体不支持 OPSEL、ABS、NEG、OMOD、DPP、clamp
8. **内联常数**: 16-bit 数据源可用内联常数，BF16 用 FP32 常数的高 16 位

---

## 5. GFX1200 (RDNA4) 特有适配

| 适配项 | 说明 |
|--------|------|
| s63 零寄存器 | MUBUF soffset 必须 SGPR（不能立即数），保留 s63 为 0 |
| s_setexeclo_b32 | 不支持，用 s_mov_b32 exec_lo, -1 替代 |
| s_barrier | 不支持，用 s_barrier_signal/wait 替代 |
| s_wait_loadcnt | 代替旧的 s_waitcnt vmcnt |
| s_wait_kmcnt | 代替旧的 s_waitcnt lgkmcnt |
| VCC 寄存器 | Wave32 只用低 32 位（高 32 位忽略） |
| MUBUF soffset | 必须 SGPR（不能立即数 0），用 s63 |

---

## 6. VOP3P 编码格式

VOP3P 指令格式 (64-bit):
- [31:24] OPCODE
- [23:16] VDST (目标 VGPR)
- [15:8]  SRC0
- [7:0]   SRC1
- [63:56] SRC2
- [55:48] OPSEL (packed math: 选择高/低 16 位)
- [47:40] OPSEL_HI
- [39:32] NEG/CLAMP/OMOD

**packed math 规则**:
- OPSEL[0] = src0 选择高/低 16 位
- OPSEL[1] = src1 选择高/低 16 位
- NEG[1:0] = signed(1)/unsigned(0) for src0 和 src1
- CLAMP: 浮点 clamp [0,1.0]，整数 clamp [MIN,MAX]

---

## 7. WMMA 数据冒险要求（§7.12.1）

1. WMMA 指令之间需要 **S_DELAY_ALU** 或独立指令间隔
2. WMMA 结果不能立即用于下一条 WMMA 的输入（需要 delay）
3. compiler 插入 S_DELAY_ALU 避免依赖 stall
4. S_DELAY_ALU 可能零周期执行（与前指令并行）
