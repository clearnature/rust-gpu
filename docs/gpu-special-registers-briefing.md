# GPU 编译器对特殊/地址寄存器的建模与保护——调研简报

**范围**：LLVM AMDGPU 后端、ACO（Mesa intel-lgci-fdo-gitlab-mirror 分支，含 GFX1200）、LLVM issue #161759、tinygrad AMD 路径。
**勘误**：题面 pin 的 AMDGPUUsage.rst 提交 `b802b8d7` 经查证为 **2017-02-11 旧版**（394 行，无 Register Conventions 章节）；本简报的寄存器信息取自该文档 main 版 + LLVM/ACO 源码，并逐条注明出处。

---

## Q1. LLVM AMDGPU 明确保留（never allocatable）的寄存器

机制是**双通道**：

1. **`SIRegisterInfo::getReservedRegs()`**（llvm/lib/Target/AMDGPU/SIRegisterInfo.cpp:599）返回 BitVector，Greedy RA 永不分配。清单：
   - `MODE`；`EXEC`（EXEC_LO/HI，HW 编号 126/127）；`FLAT_SCR`（102/103）；`M0`（注释：必须保留，否则 LLVM 不接受其作为 block live-in）；`SRC_VCCZ/SRC_EXECZ/SRC_SCC`；aperture 伪寄存器 `SRC_SHARED_BASE/LIMIT`、`SRC_PRIVATE_BASE/LIMIT`、`SRC_FLAT_SCRATCH_BASE_LO/HI`；`ASYNCcnt/TENSORcnt`；`SRC_POPS_EXITING_WAVE_ID`；`XNACK_MASK`；`LDS_DIRECT`；`TBA/TMA` + `TTMP0-TTMP15`（"Trap Handler registers - support is not implemented in Codegen"）；`SGPR_NULL64`（"it shall never be allocated"）。
   - **动态部分**：HW 索引 ≥ `ST.getMaxNumSGPRs()`（按占用 wave 数收紧）的 SGPR 全部保留（`VCC_LO/HI/VCC` 除外）；需要 spill 时额外扣 4 个 scratch buffer resource SGPR。
2. **TableGen 类层 `isAllocatable=0`**（SIRegisterInfo.td）：`TTMP_32`（ttmp0-15）、`M0_CLASS`、`SCC_CLASS` 都是"可作源操作数、不可分配"的类；`SReg_32_XM0_XEXEC` 仅 `AllocationPriority=0`。

**关键澄清**：**v0 与 user SGPR 都不是 reserved**。v0=workitem_id_x 只是 kernel 入口 live-in（AMDHSA 提供 unpacked 法 v0=X/v1=Y/v2=Z 与 packed 法 v0 位 0:9/10:19/20:29，见 AMDGPUUsage master 版表格 @7040-7115），prologue 拷入 vreg 后 v0 即可被 RA 复用。kernarg_segment_ptr 是 ≤16 个 user SGPR 之一（CP 按启用集合顺序装入连续 SGPR，`.amdhsa_user_sgpr_kernarg_segment_ptr` @21722），现代 ABI 位置不固定；s[0:1] 只是 2017 版文档示例约定（enable_sgpr_kernarg_segment_ptr 示例）。注：main 分支 `SGPR_32 = (add (sequence "SGPR%u",0,105), VCC_LO, VCC_HI)` 已把 VCC 并入普通 SGPR 池（llvm-21 版则不含），VCC 的可分配性随代际演进。

来源：pinned URL（2017 版，仅 ABI 示例）；https://llvm.googlesource.com/llvm-project/+/refs/heads/main/llvm/docs/AMDGPUUsage.rst ；SIRegisterInfo.cpp / SIRegisterInfo.td（llvmorg-21.1.0 与 main）。

## Q2. ACO 分配器的"固定寄存器"机制

- **寄存器类概念极简**：`RegClass`（aco_ir.h:289）只含 `s1..s16 / v1..v16 / 子字 v1b-v8b / linear` 变体；`RegType` 仅 `{sgpr, vgpr}`（284-287）。**特殊寄存器不在任何类里**。
- **排除机制 = 窗口外编号**：可分配窗口是 `PhysRegInterval{PhysReg(ctx.sgpr_start), ctx.sgpr_bounds}`（aco_register_allocation.cpp:261），`sgpr_bounds` 由 `get_addr_regs_from_waves()`（wave 数→压力上限，:1520）且 ≤ `ctx.limit.sgpr`；而 `vcc{106}/m0{124}/sgpr_null{125}/exec{126}/ttmp{112-123}`（aco_ir.h:437-461 常量）全部在窗口外——源码注释原话 **"VCC is outside the bounds"**（:1490）。
- **指令级特判**：`get_reg_specified()`（:1466）允许把 VCC 当目标仅当 `ctx.program->needs_vcc`（:1492）；`is_m0 = rc==s1 && reg==m0 && can_write_m0(instr)`（:1493）；**RDNA4 特例：pseudo-scalar transcendent 禁 VCC 目标**（:1497-1503，GFX1200 直接相关）。
- **固定/预着色操作数**：`Operand::isPrecolored()/isFixed()`；`handle_fixed_operands()`（:2305）先把固定操作数从 `RegisterFile` block/clear（"so fixed operands are not collected by collect_vars()"，:2378），必要时插 parallel copy；`get_reg_for_operand()` 用 `setFixed` 落回（:2428）。
- **软 pin（亲和）**：`assignment::precolor_affinity` + `set_precolor_affinity(PhysReg)`（:57-82）；写 VCC 的指令（v_add_co 等）对 def/use 设 vcc 亲和（:3199-3213），`get_reg_specified` 优先试 affinity.reg（:1874）失败再找空闲；"phis fixed before RA can only be fixed to exec"（:2559）。
- 入口：`register_allocation(Program*, ra_test_policy)`（:3898）。

来源：https://github.com/intel-lgci-fdo-gitlab-mirror/mesa.mesa/blob/93339463/src/amd/compiler/aco_register_allocation.cpp （及同提交 aco_ir.h）。

## Q3. 地址值是否有单独类型/寄存器类、是否防折叠别名

- **LLVM**：`ptr` + addrspace（AMDGPU 映射 0=Generic/Flat、1=Global、2=Constant、3=Local、4=Private、5=Region）。**没有指针专用寄存器类**——`TargetRegisterInfo::getPointerRegClass` 默认 `llvm_unreachable`（TargetRegisterInfo.h:722），AMDGPU 未覆盖；地址值就是普通 SGPR/VGPR vreg，防折叠/别名靠 SSA 语义 + 指令寻址模式合法性检查（offset 立即数范围、addrspace 判定），与寄存器层无关。
- **ACO**：本分支 aco_ir.h **连指针 RegType 都没有**（只有 sgpr/vgpr），基址按 s2/v2 分配，地址计算是普通指令，无专门保护。
- **真正要警惕的是 issue #161759**：`SReg_32 = (add SReg_32_XM0, M0)`（SIRegisterInfo.td:840）——SReg_32 系操作数类把**不可分配的源值 M0/EXEC**（`SReg_32_XM0_XEXEC` 即"去 M0/EXEC 的 SReg_32"）混进类成员，若直接用作 vreg 类有被 RA 选中的风险；issue 建议拆成 `isAllocatable=0` 的 source-only 操作数类。这正是"特殊寄存器不得混入可分配类"的业界共识。

来源：issue https://github.com/llvm/llvm-project/issues/161759 ；SIRegisterInfo.td（main）；TargetRegisterInfo.h。

## Q4. 跨循环/跨阶段存活的基址寄存器：业界标准处理

- **LLVM Greedy：区域分裂 + 循环感知偏置**（RegAllocGreedy.cpp）：`RAGreedy::growRegion()`（:872）对 loop through-blocks 施 `SpillPlacer->addPrefSpill(NewBlocks, Strong=true)`——注释原话 "provide a strong negative bias on through blocks to prevent unwanted liveness on loop backedges"（:914-933）；例外：`SplitAnalysis::looksLikeLoopIV()` 的归纳变量允许 header↔latch 存活（:920-931）；保守切割点 `SplitAnalysis::getLastSplitPoint`（:856）；代价诊断 remark "LoopSpillReloadCopies"（:2883）。
- **备选 remat**：`LiveRangeEdit::rematerializeAt`（LiveRangeEdit.cpp:84）+ `TargetInstrInfo::isTriviallyReMaterializable`（TargetInstrInfo.h:176）；AMDGPU 无自定义 override（SIInstrInfo.cpp 无此函数），便宜的地址计算重算而非 spill 重载。
- **显式 pinning 只用于真特殊寄存器**：LLVM = reserved regs + 预着色固定操作数；ACO = precolor_affinity（软 pin，失败回落）+ isFixed（硬 pin）。基址寄存器不 pin，作为普通 vreg 参与分配，SGPR 压力上限按 wave 数动态收紧。
- **kernarg preload**：kernel descriptor 的 KERNARG_PRELOAD 字段（GFX9+）+ "loaded into consecutive User SGPRs"（AMDGPUUsage @7138-7158）——入口把实参/隐式参数预载 SGPR，减少基址跨循环重载。

来源：RegAllocGreedy.cpp / LiveRangeEdit.cpp / TargetInstrInfo.h（llvm main）；AMDGPUUsage master。

## Q5. tinygrad AMD 路径：委托 LLVM，自己不分配

- **生产路径**：`tinygrad/runtime/support/compiler_llvm.py` 的 `LLVMCompiler` 用 LLVM C API，triple `amdgcn-amd-amdhsa`、processor gfx1100/gfx1201、passes `default<O2>`；`compiler_amd.py` 的 `compile_hip` 走 ROCm COMGR（`amdgcn-amd-amdhsa--gfx1100`）。寄存器分配、v0/tid、user SGPR 全部由 LLVM AMDGPU 后端 ABI 负责（v0=workitem id 由 `enable_vgpr_workitem_id` 控制，tinygrad 不感知）。
- **`renderer/amd/dsl.py` + `runtime/autogen/amd/{rdna3,rdna4}` 是自研指令编码/解码库，不是寄存器分配器**（grep 无 reg_alloc/spill/live-range 代码）。用途：(a) 对照 LLVM MC 测试向量验证编码正确性——test/amd/test_llvm.py 拉取 llvmorg-21.1.0 的 `gfx11_asm_*.s`，test_integration.py 用 `llvm_assemble/llvm_disasm` 往返；(b) SQTT 跟踪反汇编（sqtt.py、viz/serve.py）。
- **寄存器命名**（dsl.py:5-89）：统一 src 编码空间 0-511——s0-105、VCC_LO/HI=106/107、ttmp=108-123、XNACK=104/105、FLAT_SCRATCH=102/103、**NULL=124、M0=125**、EXEC=126/127、内联整型 128-208、浮点常量 240-248、VCCZ/EXECZ/SCC=251-253、LIT=255、v0-255=256-511。注释点明架构差异：**"RDNA has NULL@124/M0@125, CDNA has M0@124/reserved@125"**（RDNA4 与 CDNA 编码不一致，T0 需按 GFX1200 定表）。

来源：/home/yanli/work/9060xt/tinygrad（tinygrad/runtime/support/compiler_llvm.py、compiler_amd.py、tinygrad/renderer/amd/dsl.py、test/amd/）。

---

## 对自研编译器 T0（GFX1200）的三条启示

1. **保留集合双通道建模**：一张 `getReservedRegs` 式位图（MODE/EXEC/M0/TTMP/XNACK/FLAT_SCR/SGPR_NULL + 按 wave 数动态收紧的 SGPR 上限）+ 表驱动 `isAllocatable=0` 的 source-only 操作数类（照 TTMP_32/M0_CLASS/SCC_CLASS 先例）。直接落实 issue #161759 的整改方向：**特殊寄存器可作源、不可分配、不进 vreg 类**。
2. **固定寄存器靠"窗口外编号 + 指令级特判"，不靠类**：ACO 的可分配 SGPR 窗口 `[0, sgpr_bounds)` 天然排除 vcc=106/m0=124/null=125/exec=126-127（注意 RDNA4 指令编码空间是 NULL@124/M0@125，与 ACO 内部编号差 1，需两张表）；写 VCC/M0 的指令用 `needs_vcc`/`can_write_m0` + `precolor_affinity` 特判；**必须复制 RDNA4 pseudo-scalar 禁 VCC 目标的特例**。
3. **基址跨循环不 pin，靠"循环感知 spill 偏置 + remat + 预载"**：对 loop back-edge 施 `addPrefSpill(Strong)` 强负偏置、`looksLikeLoopIV` 例外，地址计算按 remat 重算而非跨迭代存活；kernarg/基址指针入口预载 SGPR。T0 的单缓冲 TEMP / ksub 回环问题应优先用"循环头重算地址 + 负偏置防跨迭代存活"解决，而不是显式 pin 寄存器。

---

## 验证记录

- 2026-08-28：`python3 tests/verify_briefing_claims.py` → **RESULT: PASS（57 项断言通过 / 0 失败，exit 0）**。该测试分三层：① 交付文件完整性与六个章节存在性；② 简报内关键锚点（寄存器编号 vcc{106}/m0{124}/sgpr_null{125}/exec{126}/NULL@124/M0@125、函数名 get_reg_specified/handle_fixed_operands/getReservedRegs/addPrefSpill 等）；③ 逐条对源码交叉核实（aco_register_allocation.cpp、aco_ir.h、SIRegisterInfo.cpp/.td、RegAllocGreedy.cpp、TargetRegisterInfo.h、AMDGPUUsage-master.rst、tinygrad dsl.py）。测试文件：`tests/verify_briefing_claims.py`。
