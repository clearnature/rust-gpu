# GFX12 编码器 API 参考

> **目标硬件**: AMD RX 9060 XT (gfx1200, RDNA4)  
> **模块**: `rdna3_asm::gfx11` (编码), `rdna3_disasm` (反汇编)  
> **验证状态**: 367 tests pass, llvm-mc 交叉验证

---

## 1. 概述

T0-GPU 的 GFX12 编码器提供两层 API:

1. **编码器** (`src/rdna3_asm.rs`): 将指令参数编码为 `[u32]` 机器码
2. **反汇编器** (`src/rdna3_disasm.rs`): 将 `[u32]` 机器码解码为汇编文本

所有 GFX1200 (RDNA4) 指令通过 `gfx11::*_gfx1200()` 函数编码。
GFX1100 (RDNA3) 指令使用对应的无后缀函数。

---

## 2. SMEM — 标量内存

```rust
// 统一 helper (L1.4 重构)
fn gfx12_smem(dst: u8, base: u8, offset: u32, size: u32) -> [u32; 2]
// word0 = 0xF4000000 | (size << 12) | (dst << 6) | (base / 2)
// word1 = 0xF8000000 | (offset & 0xFFFFFF)

// 公共 API
s_load_dword_gfx1200(dst: u8, base: u8, offset: u32) -> [u32; 2]   // s_load_b32
s_load_dwordx2_gfx1200(dst: u8, base: u8, offset: u32) -> [u32; 2] // s_load_b64 (dst 必须 2-aligned)
s_load_dwordx4_gfx1200(dst: u8, base: u8, offset: u32) -> [u32; 2] // s_load_b128 (dst 必须 4-aligned)
```

**约束**:
- `base` 必须 2-aligned (SGPR pair)
- B64: `dst` 必须 2-aligned; B128: `dst` 必须 4-aligned
- `offset`: 24 位无符号 (0..16M)

**示例**:
```rust
use t0_gpu::rdna3_asm::gfx11;
let words = gfx11::s_load_dwordx2_gfx1200(4, 0, 0x10); // s_load_b64 s[4:5], s[0:1], 0x10
```

---

## 3. VGLOBAL — 全局内存 (96 位格式)

### 3.1 Load

```rust
// 统一 helper (L1.4 重构)
fn gfx12_vglobal_load(base_op: u32, vdst: u8, vaddr: u8, offset: i32) -> [u32; 3]
// word0 = base_op | 0x7C
// word1 = vdst
// word2 = (offset << 8) | vaddr

// 公共 API
global_load_dword_gfx1200(vdst, vaddr, offset) -> [u32; 3]    // global_load_b32
global_load_dwordx2_gfx1200(vdst, vaddr, offset) -> [u32; 3]  // global_load_b64
global_load_dwordx4_gfx1200(vdst, vaddr, offset) -> [u32; 3]  // global_load_b128
global_load_ushort_gfx1200(vdst, vaddr, offset) -> [u32; 3]   // global_load_u16
```

### 3.2 Store

```rust
fn gfx12_vglobal_store(base_op: u32, vaddr: u8, vsrc: u8, offset: i32) -> [u32; 3]
// word0 = base_op | 0x7C
// word1 = (vsrc / 2) | ((vsrc & 1) << 23)  // ⚠️ 已知: 与 llvm-mc word1 格式有差异
// word2 = (offset << 8) | vaddr

global_store_dword_gfx1200(vaddr, vsrc, offset) -> [u32; 3]    // global_store_b32
global_store_dwordx2_gfx1200(vaddr, vsrc, offset) -> [u32; 3]  // global_store_b64
global_store_dwordx4_gfx1200(vaddr, vsrc, offset) -> [u32; 3]  // global_store_b128
global_store_short_gfx1200(vaddr, vsrc, offset) -> [u32; 3]    // global_store_b16
```

### 3.3 Atomic

```rust
fn gfx12_vglobal_atomic(base_op: u32, vaddr: u8, vdata: u8, offset: i32, vdst: Option<u8>) -> [u32; 3]
// word0 = base_op | (vdata << 1) | 0x7C
// word1 = if rtn { vdst | (1 << 16) } else { 0 }
// word2 = (offset << 8) | vaddr

global_atomic_add_u32_gfx1200(vdst, vaddr, vdata, offset) -> [u32; 3]       // u32 + return
global_atomic_add_u32_no_rtn_gfx1200(vaddr, vdata, offset) -> [u32; 3]     // u32, fire-and-forget
global_atomic_add_f32_gfx1200(vdst, vaddr, vdata, offset) -> [u32; 3]      // f32 + return
global_atomic_add_f32_no_rtn_gfx1200(vaddr, vdata, offset) -> [u32; 3]     // f32, fire-and-forget
```

**base_op 常量**:
| 指令 | base_op |
|------|---------|
| load_b32 | 0xEE050000 |
| load_b64 | 0xEE054000 |
| load_b128 | 0xEE05C000 |
| load_u16 | 0xEE048000 |
| store_b32 | 0xEE068000 |
| store_b64 | 0xEE06C000 |
| store_b128 | 0xEE074000 |
| store_b16 | 0xEE064000 |
| atomic_u32_rtn | 0xEE0D4000 |
| atomic_u32_nortn | 0xEE054000 |
| atomic_f32_rtn | 0xEE154000 |
| atomic_f32_nortn | 0xEE0D4000 |

---

## 4. VOP3P — WMMA 矩阵指令

```rust
fn vop3p_mai_word1(va: u8, vb: u8, vc: u8) -> u32
// word1 = 0x1C000000 | (va+256) | ((vb+256) << 9) | ((vc+256) << 18)

v_wmma_f32_16x16x16_bf16(vdst, va, vb, vc) -> [u32; 2]  // BF16→F32
v_wmma_f32_16x16x16_f16(vdst, va, vb, vc) -> [u32; 2]   // F16→F32
v_wmma_bf16_16x16x16_bf16(vdst, va, vb, vc) -> [u32; 2] // BF16→BF16
v_wmma_f16_16x16x16_f16(vdst, va, vb, vc) -> [u32; 2]   // F16→F16
```

---

## 5. SOPP — 标量程序控制

GFX12 SOPP opcode 表 (与 GFX11 不同!):

| OP | GFX12 指令 | 编码 |
|----|-----------|------|
| 0x09 | s_waitcnt (legacy bitfield) | 0xBF89xxxx |
| 0x14 | s_barrier_wait | 0xBF94xxxx |
| 0x20 | s_branch | 0xBFA0xxxx |
| 0x21 | s_cbranch_scc0 | 0xBFA1xxxx |
| 0x30 | s_endpgm | 0xBFB0xxxx |
| 0x3D | s_barrier | 0xBFBDxxxx |
| 0x40 | s_wait_loadcnt | 0xBFC0xxxx |
| 0x41 | s_wait_storecnt | 0xBFC1xxxx |
| 0x45 | s_wait_dscnt | 0xBFC5xxxx |
| 0x46 | s_wait_dscnt | 0xBFC6xxxx |
| 0x47 | s_wait_kmcnt | 0xBFC7xxxx |

**Waitcnt API**:
```rust
s_wait_loadcnt(n: u8) -> u32   // 0xBFC00000 | n
s_wait_storecnt(n: u8) -> u32  // 0xBFC10000 | n
s_wait_dscnt(n: u8) -> u32     // 0xBFC60000 | n
s_wait_kmcnt(n: u8) -> u32     // 0xBFC70000 | n
s_wait_tensorcnt(n: u8) -> u32 // 0xBFCB0000 | n (GFX1250 only)
s_wait_asynccnt(n: u8) -> u32  // 0xBFCA0000 | n (GFX1250 only)
```

**Barrier (GFX12)**:
```rust
// GFX12 不支持 s_barrier — 使用 signal + wait 组合:
// s_barrier_signal -1 (SOP1: 0xBE804EC1)
// s_barrier_wait -1   (SOPP: 0xBF94FFFF)
```

---

## 6. 反汇编器

```rust
use t0_gpu::rdna3_disasm::disasm;

// 解码机器码为汇编文本
let code: &[u32] = &[0xF4002100, 0xF8000010, 0xBFC00000, 0xBFB00000];
let asm = disasm(code, true);  // true = GFX12
// → "s_load_b64 s[4:5], s[0:1], 0x10\ns_wait_loadcnt 0\ns_endpgm\n"

// 单条指令解码
let (text, n_words) = disasm_insn(&code[0..2], true);
// → ("s_load_b64 s[4:5], s[0:1], 0x10", 2)
```

**支持格式**: SOPP, SOP1, SOP2, SOPK, SMEM, VOP1, VOP2, VOP3, VOPC, VOP3P, DS, Flat, VGlobal

---

## 7. 验证状态

| 测试类别 | 测试数 | 状态 |
|----------|--------|------|
| SMEM 编码 | 4 | ✅ |
| VGLOBAL FLAT 编码 | 12 | ✅ |
| VGLOBAL Atomic 编码 | 5 | ✅ |
| Waitcnt 编码 | 4 | ✅ |
| Barrier 编码 | 1 | ✅ |
| WMMA 编码 | 5 | ✅ |
| 反汇编器 | 17 | ✅ |
| Round-trip (text) | 30 | ✅ |
| Round-trip (binary via llvm-mc) | 7 | ✅ |
| Round-trip (ignored) | 3 | ⚠️ 见已知限制 |
| **总计** | **367** | **0 failed** |

### 已知限制

1. **VGLOBAL store word1 格式**: encoder 使用 bits[22:16] 编码 vsrc, llvm-mc 使用 bits[25:23]。Text round-trip 正确。
2. **s_barrier_signal SOP1 sdst**: disassembler 输出寄存器号而非立即数 (-1)。
3. **Atomic nortn 歧义**: `global_atomic_add_u32_no_rtn` 与 `global_load_b64` 共享 w0_base (0xEE054000)。

---

## 8. 文件结构

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/rdna3_asm.rs` | ~3800 | ISA 编码器 + Rdna3Assembler |
| `src/rdna3_disasm.rs` | ~1770 | ISA 反汇编器 + round-trip 测试 |
| `src/rdna3_code_object.rs` | ~500 | ELF/HSA code object 生成 |
| `src/wmma_db.rs` | ~600 | WMMA intrinsic 数据库 |
| `src/t0/isa_probe.rs` | ~1000 | llvm-mc 编码探测工具 |
