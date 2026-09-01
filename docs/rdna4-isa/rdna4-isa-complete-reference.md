# RDNA4 指令集架构 (ISA) 完整参考

> 来源: AMD "RDNA4" Instruction Set Architecture Reference Guide (7-April-2025, 707pp)
> 目标读者: 编译器编写者、驱动开发者、底层运行时开发者
> 提取方式: pdftotext + 结构化整理
> 日期: 2026-08-23

---

## 文档说明

这份《RDNA4 指令集架构 (ISA)》文档是 AMD 官方专门为**编译器编写者、驱动开发者和底层运行时开发者**准备的，它完整定义了 RDNA4 硬件的指令集、二进制编码格式以及执行语义。

如果你正在使用 Rust 编写一个 GPU 运行时（例如一个自定义的 GPU 驱动、模拟器、编译器后端），这份 PDF 提供的指令是**发射、验证和调度代码的核心依据**。

---

## 1. 程序流控制指令 (Chapter 5)

### 分支与跳转
- `S_BRANCH`, `S_CBRANCH_SCC0/1`, `S_CBRANCH_VCCZ/VCCNZ`
- `S_SETPC_B64`, `S_SWAPPC_B64`, `S_CALL_B64`, `S_RFE_B64`（用于 trap 返回）

### 屏障与同步
- `S_BARRIER_SIGNAL`, `S_BARRIER_SIGNAL_ISFIRST`, `S_BARRIER_WAIT`, `S_GET_BARRIER_STATE`
- 用于 Workgroup 内 Wave 同步

### 陷阱与异常
- `S_TRAP`, `S_ENDPGM`

### 指令 Clauses
- `S_CLAUSE`：强制硬件在一个周期组内执行同一类指令，对性能优化至关重要

---

## 2. 标量 ALU 指令 (SALU, Chapter 6 & 16)

操作作用于整个 Wave（32/64 线程共享的值）。格式有 SOP1, SOP2, SOPK, SOPC, SOPP。

### 算术/逻辑
- `S_ADD_CO_U32`, `S_SUB_CO_I32`, `S_MUL_I32`, `S_AND`, `S_OR`, `S_XOR`, `S_LSHL` 等

### 比较
- `S_CMP_EQ_U32`, `S_CMP_LT_F32` 等，结果写入 SCC (Scalar Condition Code)

### 浮点
- `S_ADD_F32`, `S_MUL_F16` 等

### 状态管理
- `S_GETREG_B32`, `S_SETREG_B32`, `S_SETREG_IMM32_B32`（用于读写硬件寄存器，如 MODE 寄存器）
- `S_ALLOC_VGPR`（运行时动态分配 VGPR）

---

## 3. 向量 ALU 指令 (VALU, Chapter 7 & 16)

作用于每个 Thread/Lane。格式有 VOP1, VOP2, VOPC, VOP3, VOP3P, VOPD（双发）。

### 通用算术/逻辑
- `V_ADD_F32`, `V_FMA_F32`, `V_MUL`, `V_AND`, `V_ASHR` 等

### 整数点积与 AI 加速（针对 9060 XT 的 AI 核心）
- `V_DOT2_F32_F16`, `V_DOT4_I32_IU8`, `V_WMMA_*`（Wave Matrix Multiply Accumulate，矩阵乘加指令）
- 支持 FP16、BF16、FP8 (E4M3/E5M2) 等格式
- 如果要做计算运行时，这些是核心

### 跨线程数据操作
- `V_PERMLANE16`, `DPP8`, `DPP16`（用于跨 Lane 的数据交换，如 Scan 扫描算法）

### 索引寻址
- `V_MOVRELD`, `V_MOVRELS`（配合 M0 寄存器实现动态 VGPR 索引）

---

## 4. 内存访问指令 (Chapter 8 - 13)

### 标量内存 (SMEM)
- `S_LOAD_B32/B64/B128`, `S_BUFFER_LOAD`, `S_DCACHE_INV`
- 用于加载常数和描述符

### 向量 Buffer (VBUFFER)
- `BUFFER_LOAD_*`, `BUFFER_STORE_*`, `BUFFER_ATOMIC_*`
- 用于访问结构化缓冲

### 向量 Image (VIMAGE / VSAMPLE)
- `IMAGE_LOAD/STORE`, `IMAGE_SAMPLE`, `IMAGE_GATHER4`, `IMAGE_BVH_INTERSECT_RAY`（光追指令）

### 全局/私有/平面寻址 (VFLAT, VGLOBAL, VSCRATCH)
- `GLOBAL_LOAD_*`, `GLOBAL_STORE_*`, `GLOBAL_ATOMIC_*`, `FLAT_*`, `SCRATCH_*`
- 用于批量移动的 `GLOBAL_LOAD_BLOCK`, `GLOBAL_LOAD_TR_*`（矩阵转置加载）

### 本地数据共享 (LDS)
- `DS_LOAD_*`, `DS_STORE_*`, `DS_ATOMIC_*`, `DS_PERMUTE`, `DS_SWIZZLE`
- 专为光追设计的 `DS_BVH_STACK_PUSH/POP` 栈操作指令

---

## 5. 同步与依赖管理指令 (Chapter 5.7, 16.5)

### 内存一致性
- `S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_EXPCNT`, `S_WAIT_DSCNT`, `S_WAIT_SAMPLECNT`, `S_WAIT_BVHCNT`, `S_WAIT_KMCNT`
- 通过计数器确保内存操作完成

### ALU 依赖
- `S_DELAY_ALU`（用于在流水线中插入延迟以避免数据冒险）

### 全局同步/缓存
- `GLOBAL_INV`, `GLOBAL_WB`, `GLOBAL_WBINV`, `S_ICACHE_INV`

---

## 6. 输出与导出指令 (Chapter 14, 16.19)

- `VEXPORT`: `EXPORT` 指令，用于将 VGPR 中的像素颜色、深度或顶点位置发送到固定功能硬件（Raster Backend）

---

## 💡 对 Rust GPU 运行时的开发建议

1. **二进制编码格式**：如果你需要**发射 (Emit)** 这些指令，请重点参考 **第 15 章 (Microcode Formats)**，它详细列出了 32位/64位指令中每一位的字段定义（例如 VOP3 的 src0 是第 40 到 32 位）。

2. **指令语义**：如果你需要 **模拟 (Emulate)**、**反汇编 (Disassemble)** 或做 **代码校验**，请参考 **第 16 章 (Instructions)** 中的伪代码逻辑。

3. **硬件限制**：第 3.5 节的初始 Wave 状态（寄存器初始化）和第 2.4 节的 shader 填充要求（S_CODE_END 补位）是驱动加载器必须处理的标准。

---

## 附录：T0 编译器 ISA 覆盖核对

| 指令类别 | T0 支持状态 |
|----------|------------|
| SOP2/SOPK/SOP1/SOPC/SOPP (SALU) | ✅ 全部使用 |
| VOP2/VOP1/VOPC/VOP3 (VALU) | ✅ 全部使用 |
| VOP3P (Packed Math) | ✅ 部分使用 |
| VOPD (Dual Issue) | ❌ 未使用 |
| WMMA (11种变体) | ✅ 10种，缺 INT4 K=32 |
| SWMMAC (12种变体) | ❌ 全未支持 |
| SMEM/VBUFFER/VDS/FLAT/GLOBAL | ✅ 全部使用 |
| S_BARRIER/S_WAIT_* | ✅ 全部使用 |
