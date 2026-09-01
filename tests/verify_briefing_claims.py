#!/usr/bin/env python3
"""Verify the factual claims in docs/gpu-special-registers-briefing.md against the
actual reference sources (LLVM / ACO / tinygrad). Fails (exit 1) if any anchor is
missing from a source file or any required section is missing from the briefing.

Usage: python3 tests/verify_briefing_claims.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIEF = os.path.join(ROOT, "docs", "gpu-special-registers-briefing.md")
REFS = "/tmp/t0-research-refs"
TINYGRAD = "/home/yanli/work/9060xt/tinygrad"

fails = []

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(label)

def file_has(path, needle):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return needle in f.read()
    except FileNotFoundError:
        return False

def has_any(path, needles):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return any(n in text for n in needles)
    except FileNotFoundError:
        return False

print("== 1. Deliverable integrity ==")
check("briefing file exists", os.path.isfile(BRIEF))
if os.path.isfile(BRIEF):
    b = open(BRIEF, encoding="utf-8").read()
    for sec in ["## Q1.", "## Q2.", "## Q3.", "## Q4.", "## Q5.", "## 对自研编译器 T0（GFX1200）的三条启示"]:
        check(f"section present: {sec}", sec in b)

print("== 2. Briefing content anchors (numbers / function names) ==")
for anchor in ["vcc{106}", "m0{124}", "sgpr_null{125}", "exec{126}", "NULL=124、M0=125",
               "get_reg_specified", "handle_fixed_operands", "precolor_affinity",
               "getReservedRegs", "SReg_32_XM0_XEXEC", "isAllocatable=0",
               "addPrefSpill", "looksLikeLoopIV", "LoopSpillReloadCopies",
               "getPointerRegClass", "amdgcn-amd-amdhsa", "enable_vgpr_workitem_id"]:
    check(f"briefing mentions: {anchor}", anchor in b)

print("== 3. Claim cross-check against sources ==")
SRC = [
    # (label, path, required substring)
    ("ACO: VCC outside allocatable bounds", f"{REFS}/aco_register_allocation.cpp", "VCC is outside the bounds"),
    ("ACO: get_reg_specified signature", f"{REFS}/aco_register_allocation.cpp", "get_reg_specified(ra_ctx& ctx"),
    ("ACO: handle_fixed_operands", f"{REFS}/aco_register_allocation.cpp", "handle_fixed_operands(ra_ctx& ctx"),
    ("ACO: RA entry register_allocation(Program*)", f"{REFS}/aco_register_allocation.cpp", "register_allocation(Program* program"),
    ("ACO: needs_vcc special case", f"{REFS}/aco_register_allocation.cpp", "needs_vcc"),
    ("ACO: can_write_m0 special case", f"{REFS}/aco_register_allocation.cpp", "can_write_m0"),
    ("ACO: RDNA4 pseudo-scalar VCC-dst ban", f"{REFS}/aco_register_allocation.cpp", "valu_pseudo_scalar_trans"),
    ("Greedy: addPrefSpill strong bias on through-blocks", f"{REFS}/RegAllocGreedy.cpp", "addPrefSpill"),
    ("ACO ir: m0=124", f"{REFS}/aco_ir.h", "static constexpr PhysReg m0{124}"),
    ("ACO ir: vcc=106", f"{REFS}/aco_ir.h", "static constexpr PhysReg vcc{106}"),
    ("ACO ir: sgpr_null=125", f"{REFS}/aco_ir.h", "static constexpr PhysReg sgpr_null{125}"),
    ("ACO ir: exec=126", f"{REFS}/aco_ir.h", "static constexpr PhysReg exec{126}"),
    ("ACO ir: RegType has no pointer member", f"{REFS}/aco_ir.h", "enum class RegType"),
    ("LLVM: getReservedRegs", f"{REFS}/SIRegisterInfo.cpp", "BitVector SIRegisterInfo::getReservedRegs"),
    ("LLVM: SGPR_NULL64 reserved (never allocated)", f"{REFS}/SIRegisterInfo.cpp", "Reserve null register"),
    ("LLVM: TTMP0-15 reserved", f"{REFS}/SIRegisterInfo.cpp", "TTMP14_TTMP15"),
    ("LLVM: XNACK_MASK reserved", f"{REFS}/SIRegisterInfo.cpp", "XNACK_MASK"),
    ("LLVM td: SReg_32_XM0_XEXEC class", f"{REFS}/SIRegisterInfo-main.td", "def SReg_32_XM0_XEXEC : SIRegisterClass"),
    ("LLVM td: SReg_32 = SReg_32_XM0 + M0", f"{REFS}/SIRegisterInfo-main.td", "(add SReg_32_XM0, M0)"),
    ("LLVM td: isAllocatable=0 classes", f"{REFS}/SIRegisterInfo-main.td", "let isAllocatable = 0"),
    ("LLVM td: TTMP_32 non-allocatable class", f"{REFS}/SIRegisterInfo-main.td", '(sequence "TTMP%u", 0, 15)'),
    ("LLVM td: main SGPR_32 includes VCC_LO/HI", f"{REFS}/SIRegisterInfo-main.td", "VCC_LO, VCC_HI)> {"),
    ("Greedy: growRegion", f"{REFS}/RegAllocGreedy.cpp", "RAGreedy::growRegion"),
    ("Greedy: looksLikeLoopIV exception", f"{REFS}/RegAllocGreedy.cpp", "looksLikeLoopIV"),
    ("Greedy: LoopSpillReloadCopies remark", f"{REFS}/RegAllocGreedy.cpp", "LoopSpillReloadCopies"),
    ("LLVM: getPointerRegClass default unreachable", f"{REFS}/TargetRegisterInfo.h", "Target didn't implement getPointerRegClass"),
    ("doc: user sgpr kernarg ptr directive", f"{REFS}/AMDGPUUsage-master.rst", ".amdhsa_user_sgpr_kernarg_segment_ptr"),
    ("doc: enable_vgpr_workitem_id", f"{REFS}/AMDGPUUsage-master.rst", "enable_vgpr_workitem_id"),
    ("tinygrad dsl: RDNA NULL@124/M0@125 vs CDNA", f"{TINYGRAD}/tinygrad/renderer/amd/dsl.py", "RDNA has NULL@124/M0@125, CDNA has M0@124/reserved@125"),
    ("tinygrad dsl: VCC_LO=106 name table", f"{TINYGRAD}/tinygrad/renderer/amd/dsl.py", '106: "VCC_LO"'),
    ("tinygrad dsl: NULL=124 / M0=125", f"{TINYGRAD}/tinygrad/renderer/amd/dsl.py", '124: "NULL", 125: "M0"'),
]
for label, path, needle in SRC:
    check(f"{label}", file_has(path, needle), detail=f"needle not found in {path}")

# RegType member check (no pointer regtype in this ACO fork)
regtype = open(f"{REFS}/aco_ir.h", encoding="utf-8").read()
m = re.search(r"enum class RegType \{(.*?)\n\}", regtype, re.S)
check("ACO RegType body is only sgpr/vgpr", bool(m) and not re.search(r"p[0-9]", m.group(1)),
      detail=m.group(1).strip() if m else "enum not found")

print()
if fails:
    print(f"RESULT: FAIL ({len(fails)} failed)")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS (all anchors verified)")
