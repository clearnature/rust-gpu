# 浑天 SASS 汇编器深度审计

> 审计目标: /data/rtl-sdr/sass-assembler
> 审计日期: 2026-08-24

## 概述

浑天 SASS 汇编器 (HunTian) 是一个多架构 GPU 汇编器项目. 测试结果 22/22 全部通过, 但各架构的实现深度差异巨大.

## 逐架构审计

### Pascal (sm_61) — ✅ 完整, 真实编码

| 文件 | 行数 | 状态 |
|------|------|------|
| pascal_backend.h | 151 | 后端调度, 委托给 encoder |
| encoder/pascal_encoder.h | 171 | **真实 bit-level 编码** |
| disassembler/pascal_disasm.h | 106 | 反汇编 |
| pascal_hal.cpp | 151 | HAL 抽象 (v512 向量运算) |

**真实性验证:**
- 22 opcode 家族, 108 条指令, 全部有编码函数
- EXIT 硬编码 `0xe30000000007000f` — 与 cuobjdump 实际 hex 完全一致
- FFMA: `op::FFMA<<56` + 寄存器偏移 0, 16, 24, 32 — 匹配 Pascal SASS 布局
- 测试断言: `ASSERT_EQ(code[0], 0xe30000000007000fULL)`

### Volta (sm_70) — ✅ 扎实, 真实编码

| 文件 | 行数 | 状态 |
|------|------|------|
| volta_backend.h | 200 | 后端调度 |
| volta_encoder.h | 239 | **真实 128-bit 编码** |
| volta_opcode_table.h | 115 | cuobjdump 验证的 opcode 常量 |
| volta_disasm.h | (反汇编) | 反汇编 |

**真实性验证:**
- 128-bit 编码模型 (2×64-bit words)
- EXIT=0x794d, RET=0x7950, BRA=0x7947 — 真实 Volta SASS opcode
- BRA 编码: `0xfffffff000007947ULL` with offset at bits [31:12]
- 65 条指令编码/反汇编, 35 个测试通过

### Ampere (sm_80/86/89) — ✅ 可用, 扩展 Volta

| 文件 | 行数 | 状态 |
|------|------|------|
| ampere_precise_encoder.h | 91 | **真实 128-bit 双字编码** |

**真实性验证:**
- STG=0x7986, LDG=0x7981, BAR=0xc0ff, MEMBAR=0xd000
- FP16: HADD2=0x0804, HMUL2=0x080e, HFMA2=0x0812
- FFMA 13-bit opcode split: bits [0:10] + bit 91 in word1
- Scoreboard: dst_wr_sb = bits [110:112], src_rel_sb = bits [113:115]
- 21 个测试通过, opcode 值精确匹配

### Ada (sm_89) / Hopper (sm_90) / Blackwell (sm_100) — ⚠️ 骨架

继承自 Volta/Ampere, 只添加:
- Hopper: WGMMA/TMA/mbarrier opcode (0x7f00-0x7f40) + 简单反汇编
- Ada/Blackwell: 更名 + opcode 表更新

13 个测试 (factory + 3 条基础指令: exit/stg/ldg)

### AMD RDNA4 (gfx1200) — ❌ 只有定义, 无编码

| 文件 | 行数 | 状态 |
|------|------|------|
| amd_rdna4.h | 682 | **ISA 参考表 + enum/struct 定义** |

**问题:** encode() 函数产出的是浑天内部 tagged word, 不是 RDNA4 机器码:
```cpp
SASSWord w = 0xE0ULL << 56;  // tag byte (非 AMD 编码)
w |= (uint64_t)(inst.opcode) << 48;  // 浑天 opcode (非 AMD opcode)
```

ISA 表中 200+ 条指令的元数据 (opcode 号, 延迟, 编码族) 是正确的, 但从未用于实际编码.
0 个编码测试.

### x86 — ❌ 只有定义

177 行, 40+ 指令映射表 + Broadwell 延迟数据, 无实际编码.

## 代码量审计

```
真实编码逻辑 (有 bit 操作的函数): ~837 行
  pascal_encoder.h          171 行
  volta_encoder.h           239 行
  ampere_precise_encoder.h   91 行
  volta_opcode_table.h      115 行
  fast_opcode_table.h       220 行

纯定义 (enum, struct):      ~860 行
  amd_rdna4.h                682 行
  x86_backend.h              177 行

基础设施:                   ~2375 行
  pascal_backend.h           151 行
  volta_backend.h            200 行
  pascal_hal.cpp             151 行
  block_encoder.cpp          164 行
  manifold_scheduler.cpp     121 行
  optimizer.h                133 行
  ilp_model.h                654 行
  reg_allocator.h            147 行
  disassembler.cpp           118 行
  assembler.cpp               49 行
  simd_kernels.h             186 行

测试: ~440 个 test case (22 个 test executable)
```

## 测试验证矩阵

| 测试 | 验证内容 |
|------|---------|
| test_pascal_backend (9) | Pascal opcode bytes 匹配 cuobjdump hex, encode-decode roundtrip |
| test_volta_encode (35) | Volta opcode 值和编码, 128-bit 布局 |
| test_ampere_encode (21) | Ampere-specific opcode: HADD2=0x0804, IADD3=0x8212 精确匹配 |
| test_e2e_archs (6) | Factory 创建正确后端, encode-disassemble roundtrip |
| test_industrial (22) | 错误诊断, 10 条指令 roundtrip, 100k 指令性能 (<5s) |
| test_assembler (7) | 完整流水线: ffma R0,R1,R2 → encode → disassemble → text |

## 输出文件问题

`.sabin` 文件使用浑天内部格式, 不是 GPU 可执行的 `.cubin` 或真实 AMD code object.

## 对 t0-gpu 通用运行时的价值

| 组件 | 状态 | t0-gpu 可用性 |
|------|------|-------------|
| Pascal 编码器 | ✅ 真实编码 | 直接用 |
| Volta/Ampere 编码器 | ✅ 真实 opcode | 可用, 需补齐指令变体 |
| Hopper tensor ops | ⚠️ opcode 有, 编码不完整 | 需补 ~800 行 |
| Ada/Blackwell | ⚠️ 骨架 | 需大量补充 |
| AMD RDNA4 | ❌ 纯定义 | 用 t0-gpu 的 rdna3_asm.rs |
| .cubin ELF 生成 | ❌ 无 | 需写 |
| JIT 编译器 | ⚠️ Volta 三寄存器快速路径 | 部分可用 |
| IDeviceBackend 抽象 | ✅ 真实多态设计 | 架构可复用 |
| ILP 调度模型 | ⚠️ 硬件参数数据库 + 基础调度 | 模型数据可用 |
