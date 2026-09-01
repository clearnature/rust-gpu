//! WMMA Intrinsic Database — Query by (target, M, N, K, A_type, B_type, D_type)
//!
//! Encoding data verified via llvm-mc:
//!   /opt/llvm-23/bin/llvm-mc --triple=amdgcn -mcpu=gfx1200 --show-encoding
//!   /opt/llvm-23/bin/llvm-mc --triple=amdgcn -mcpu=gfx1250 --show-encoding
//!
//! Reference:
//!   TritonAMDGPUTransforms/WmmaGroup.cpp — upstream intrinsic map
//!   sass-assembler/src/sass/backends/amd_rdna4.h — RDNA4 ISA table
//!
//! ## GFX1200 (RDNA4) vs GFX1250 (RDNA4.5) operand widths
//!
//! | Type class       | A/B VGPRs (K=16) | A/B VGPRs (K=32) | C/D VGPRs | Notes          |
//! |------------------|-------------------|-------------------|-----------|----------------|
//! | f16/bf16 → f32   | 4                 | 8 (gfx1250 only)  | 8         | Most common    |
//! | f16/bf16 → self  | 4                 | 8 (gfx1250 only)  | 4         | Saves VGPRs    |
//! | iu8 → i32        | 2                 | 8 (gfx1250 K=64)  | 8         | INT8           |
//! | iu4 → i32        | 1 (K=16)          | 2 (K=32)          | 8         | INT4           |
//! | fp8/bf8 → f32    | 2 (K=16)          | 8 (gfx1250 K=64)  | 8         | RDNA4 FP8/BF8  |
//!
//! ## Encoding format (VOP3P-MAI, 2 dwords)
//!
//! ```text
//! word0 = 0xCCXX4000 | (vdst & 0x7F)
//!   [22:16] = opcode (type-specific)
//!   [15:8]  = 0x40 (fixed VOP3P-MAI prefix)
//!   [6:0]   = VDST base register
//!
//! word1 = (src0+256) | ((src1+256) << 9) | ((src2+256) << 18) | 0x1C000000
//!   [8:0]   = SRC0 (VGPR + 256)
//!   [17:9]  = SRC1 (VGPR + 256)
//!   [26:18] = SRC2 (VGPR + 256)
//!   [31:27] = 0b11100 (VOP3P-MAI modifier)
//! ```

use std::fmt;

/// Input element type for WMMA operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WmmaType {
    /// 16-bit floating point (IEEE 754 half)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit floating point (accumulator only)
    F32,
    /// Unsigned 8-bit integer
    IU8,
    /// Unsigned 4-bit integer
    IU4,
    /// 8-bit floating point (FP8 E4M3 format, RDNA4 WMMA)
    FP8,
    /// 8-bit brain floating point (BF8 E5M2 format, RDNA4 WMMA)
    BF8,
}

impl fmt::Display for WmmaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WmmaType::F16 => write!(f, "f16"),
            WmmaType::BF16 => write!(f, "bf16"),
            WmmaType::F32 => write!(f, "f32"),
            WmmaType::IU8 => write!(f, "iu8"),
            WmmaType::IU4 => write!(f, "iu4"),
            WmmaType::FP8 => write!(f, "fp8"),
            WmmaType::BF8 => write!(f, "bf8"),
        }
    }
}

/// GPU target generation for WMMA lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WmmaTarget {
    /// RDNA4 (gfx1200) — K=16 only for bf16/f16; K=32 for iu4
    GFX1200,
    /// RDNA4.5 (gfx1250) — adds K=32 bf16/f16, K=64 iu8
    GFX1250,
}

impl WmmaTarget {
    /// Create from target GFX string.
    pub fn from_gfx(gfx: &str) -> Option<Self> {
        if gfx.starts_with("gfx125") {
            Some(WmmaTarget::GFX1250)
        } else if gfx.starts_with("gfx12") {
            Some(WmmaTarget::GFX1200)
        } else {
            None // GFX1100 and below not in this DB
        }
    }
}

/// Complete descriptor for a WMMA intrinsic's binary encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WmmaIntrinsicInfo {
    /// Assembly mnemonic (e.g. "v_wmma_f32_16x16x16_bf16")
    pub mnemonic: &'static str,
    /// M dimension (always 16 for current RDNA WMMA)
    pub m: u8,
    /// N dimension (always 16 for current RDNA WMMA)
    pub n: u8,
    /// K dimension (reduction axis depth per instruction)
    pub k: u8,
    /// Number of consecutive VGPRs for A operand
    pub a_vgprs: u8,
    /// Number of consecutive VGPRs for B operand
    pub b_vgprs: u8,
    /// Number of consecutive VGPRs for C/D (accumulator) operand
    pub cd_vgprs: u8,
    /// Input element type for A
    pub a_type: WmmaType,
    /// Input element type for B
    pub b_type: WmmaType,
    /// Accumulator element type (D/C)
    pub d_type: WmmaType,
    /// word0 base value (OR with vdst bits [6:0])
    pub word0_base: u32,
    /// LLVM intrinsic name for reference
    pub llvm_intrinsic: &'static str,
}

impl WmmaIntrinsicInfo {
    /// Compute full word0 for given VDST base register.
    #[inline]
    pub fn word0(&self, vdst: u8) -> u32 {
        self.word0_base | (vdst as u32 & 0x7F)
    }

    /// Compute word1 for given source VGPR base registers.
    /// All sources are encoded as VGPR + 256 in the VOP3P-MAI format.
    #[inline]
    pub fn word1(&self, va: u8, vb: u8, vc: u8) -> u32 {
        let src0 = va as u32 + 256;
        let src1 = vb as u32 + 256;
        let src2 = vc as u32 + 256;
        0x1C000000u32 | src0 | (src1 << 9) | (src2 << 18)
    }

    /// Encode the full 2-dword instruction.
    #[inline]
    pub fn encode(&self, vdst: u8, va: u8, vb: u8, vc: u8) -> [u32; 2] {
        [self.word0(vdst), self.word1(va, vb, vc)]
    }

    /// Human-readable signature: "f32_16x16x16_bf16" etc.
    pub fn signature(&self) -> String {
        format!("{}_{}x{}x{}_{}",
                self.d_type, self.m, self.n, self.k,
                if self.a_type == self.b_type {
                    self.a_type.to_string()
                } else {
                    format!("{}_{}", self.a_type, self.b_type)
                })
    }
}

impl fmt::Display for WmmaIntrinsicInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}x{}x{} {}→{}, A={}V B={}V CD={}V, word0=0x{:08X})",
               self.mnemonic, self.m, self.n, self.k,
               self.a_type, self.d_type,
               self.a_vgprs, self.b_vgprs, self.cd_vgprs,
               self.word0_base)
    }
}

// ═══════════════════════════════════════════════════════════════
// Encoding constants — llvm-mc verified
// ═══════════════════════════════════════════════════════════════

// word0 base values (OR with VDST in bits [6:0])
// All share 0xCCxx4000 where xx = opcode in bits [22:16]

// GFX1200 K=16 opcodes
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_F16:   u32 = 0xCC404000; // gfx1200: opcode=0x00
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_BF16:  u32 = 0xCC414000; // gfx1200: opcode=0x01
#[allow(non_upper_case_globals)]
const OP_F16_16X16X16_F16:   u32 = 0xCC424000; // gfx1200: opcode=0x02
#[allow(non_upper_case_globals)]
const OP_BF16_16X16X16_BF16: u32 = 0xCC434000; // gfx1200: opcode=0x03
#[allow(non_upper_case_globals)]
const OP_I32_16X16X16_IU8:   u32 = 0xCC444000; // gfx1200: opcode=0x04
#[allow(non_upper_case_globals)]
const OP_I32_16X16X16_IU4:   u32 = 0xCC454000; // gfx1200: opcode=0x05
#[allow(non_upper_case_globals)]
const OP_I32_16X16X32_IU4:   u32 = 0xCC4A4000; // gfx1200: opcode=0x0A

// GFX1200 K=16 FP8/BF8 opcodes (llvm-mc verified)
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_FP8_FP8: u32 = 0xCC464000; // gfx1200: opcode=0x06
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_FP8_BF8: u32 = 0xCC474000; // gfx1200: opcode=0x07
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_BF8_FP8: u32 = 0xCC484000; // gfx1200: opcode=0x08
#[allow(non_upper_case_globals)]
const OP_F32_16X16X16_BF8_BF8: u32 = 0xCC494000; // gfx1200: opcode=0x09

// GFX1250 K=32/K=64 opcodes (different bit layout — bit16=0)
#[allow(non_upper_case_globals)]
const OP_F32_16X16X32_F16:   u32 = 0xCC600000; // gfx1250: opcode=0x20
#[allow(non_upper_case_globals)]
const OP_F32_16X16X32_BF16:  u32 = 0xCC620000; // gfx1250: opcode=0x22
#[allow(non_upper_case_globals)]
const OP_I32_16X16X64_IU8:   u32 = 0xCC720000; // gfx1250: opcode=0x32

// GFX1250 K=64 FP8/BF8 opcodes (llvm-mc verified)
#[allow(non_upper_case_globals)]
const OP_F32_16X16X64_FP8_FP8: u32 = 0xCC6A0000; // gfx1250: opcode=0x2A
#[allow(non_upper_case_globals)]
const OP_F32_16X16X64_FP8_BF8: u32 = 0xCC6B0000; // gfx1250: opcode=0x2B
#[allow(non_upper_case_globals)]
const OP_F32_16X16X64_BF8_FP8: u32 = 0xCC6C0000; // gfx1250: opcode=0x2C
#[allow(non_upper_case_globals)]
const OP_F32_16X16X64_BF8_BF8: u32 = 0xCC6D0000; // gfx1250: opcode=0x2D
#[allow(non_upper_case_globals)]
const OP_F16_16X16X64_FP8_FP8: u32 = 0xCC6E0000; // gfx1250: opcode=0x2E
#[allow(non_upper_case_globals)]
const OP_F16_16X16X64_FP8_BF8: u32 = 0xCC6F0000; // gfx1250: opcode=0x2F
#[allow(non_upper_case_globals)]
const OP_F16_16X16X64_BF8_FP8: u32 = 0xCC700000; // gfx1250: opcode=0x30
#[allow(non_upper_case_globals)]
const OP_F16_16X16X64_BF8_BF8: u32 = 0xCC710000; // gfx1250: opcode=0x31

// ═══════════════════════════════════════════════════════════════
// GFX1200 intrinsic table
// ═══════════════════════════════════════════════════════════════

/// All WMMA intrinsics available on GFX1200 (RDNA4).
/// 7 intrinsics, all K=16 except iu4 K=32.
pub const GFX1200_INTRINSICS: &[WmmaIntrinsicInfo] = &[
    // ── f32 accumulator (most common, 8 VGPRs output) ──
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_f16",
        m: 16, n: 16, k: 16,
        a_vgprs: 4, b_vgprs: 4, cd_vgprs: 8,
        a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_F16,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.f16",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_bf16",
        m: 16, n: 16, k: 16,
        a_vgprs: 4, b_vgprs: 4, cd_vgprs: 8,
        a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_BF16,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.bf16",
    },

    // ── Native-type accumulator (4 VGPRs output, saves registers) ──
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f16_16x16x16_f16",
        m: 16, n: 16, k: 16,
        a_vgprs: 4, b_vgprs: 4, cd_vgprs: 4,
        a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F16,
        word0_base: OP_F16_16X16X16_F16,
        llvm_intrinsic: "llvm.amdgcn.wmma.f16.16x16x16.f16",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_bf16_16x16x16_bf16",
        m: 16, n: 16, k: 16,
        a_vgprs: 4, b_vgprs: 4, cd_vgprs: 4,
        a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::BF16,
        word0_base: OP_BF16_16X16X16_BF16,
        llvm_intrinsic: "llvm.amdgcn.wmma.bf16.16x16x16.bf16",
    },

    // ── Integer ──
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_i32_16x16x16_iu8",
        m: 16, n: 16, k: 16,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::IU8, b_type: WmmaType::IU8, d_type: WmmaType::F32,
        word0_base: OP_I32_16X16X16_IU8,
        llvm_intrinsic: "llvm.amdgcn.wmma.i32.16x16x16.iu8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_i32_16x16x16_iu4",
        m: 16, n: 16, k: 16,
        a_vgprs: 1, b_vgprs: 1, cd_vgprs: 8,
        a_type: WmmaType::IU4, b_type: WmmaType::IU4, d_type: WmmaType::F32,
        word0_base: OP_I32_16X16X16_IU4,
        llvm_intrinsic: "llvm.amdgcn.wmma.i32.16x16x16.iu4",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_i32_16x16x32_iu4",
        m: 16, n: 16, k: 32,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::IU4, b_type: WmmaType::IU4, d_type: WmmaType::F32,
        word0_base: OP_I32_16X16X32_IU4,
        llvm_intrinsic: "llvm.amdgcn.wmma.i32.16x16x32.iu4",
    },

    // ── FP8/BF8 (RDNA4 K=16, f32 accumulator only) ──
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_fp8_fp8",
        m: 16, n: 16, k: 16,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::FP8, b_type: WmmaType::FP8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_FP8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_fp8_bf8",
        m: 16, n: 16, k: 16,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::FP8, b_type: WmmaType::BF8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_FP8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_bf8_fp8",
        m: 16, n: 16, k: 16,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::BF8, b_type: WmmaType::FP8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_BF8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x16_bf8_bf8",
        m: 16, n: 16, k: 16,
        a_vgprs: 2, b_vgprs: 2, cd_vgprs: 8,
        a_type: WmmaType::BF8, b_type: WmmaType::BF8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X16_BF8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8",
    },
];

// ═══════════════════════════════════════════════════════════════
// GFX1250 additional intrinsics (extends GFX1200)
// ═══════════════════════════════════════════════════════════════

/// GFX1250-only WMMA intrinsics (K=32 bf16/f16, K=64 iu8/fp8/bf8).
/// GFX1250 supports all GFX1200 intrinsics PLUS these.
pub const GFX1250_EXTRA_INTRINSICS: &[WmmaIntrinsicInfo] = &[
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x32_f16",
        m: 16, n: 16, k: 32,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X32_F16,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x32.f16",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x32_bf16",
        m: 16, n: 16, k: 32,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X32_BF16,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x32.bf16",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_i32_16x16x64_iu8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::IU8, b_type: WmmaType::IU8, d_type: WmmaType::F32,
        word0_base: OP_I32_16X16X64_IU8,
        llvm_intrinsic: "llvm.amdgcn.wmma.i32.16x16x64.iu8",
    },

    // ── K=64 FP8/BF8 (GFX1250, f32 and f16 accumulators) ──
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x64_fp8_fp8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::FP8, b_type: WmmaType::FP8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X64_FP8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x64_fp8_bf8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::FP8, b_type: WmmaType::BF8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X64_FP8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x64_bf8_fp8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::BF8, b_type: WmmaType::FP8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X64_BF8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f32_16x16x64_bf8_bf8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 8,
        a_type: WmmaType::BF8, b_type: WmmaType::BF8, d_type: WmmaType::F32,
        word0_base: OP_F32_16X16X64_BF8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f16_16x16x64_fp8_fp8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 4,
        a_type: WmmaType::FP8, b_type: WmmaType::FP8, d_type: WmmaType::F16,
        word0_base: OP_F16_16X16X64_FP8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f16_16x16x64_fp8_bf8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 4,
        a_type: WmmaType::FP8, b_type: WmmaType::BF8, d_type: WmmaType::F16,
        word0_base: OP_F16_16X16X64_FP8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f16_16x16x64_bf8_fp8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 4,
        a_type: WmmaType::BF8, b_type: WmmaType::FP8, d_type: WmmaType::F16,
        word0_base: OP_F16_16X16X64_BF8_FP8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8",
    },
    WmmaIntrinsicInfo {
        mnemonic: "v_wmma_f16_16x16x64_bf8_bf8",
        m: 16, n: 16, k: 64,
        a_vgprs: 8, b_vgprs: 8, cd_vgprs: 4,
        a_type: WmmaType::BF8, b_type: WmmaType::BF8, d_type: WmmaType::F16,
        word0_base: OP_F16_16X16X64_BF8_BF8,
        llvm_intrinsic: "llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8",
    },
];

// ═══════════════════════════════════════════════════════════════
// Query API
// ═══════════════════════════════════════════════════════════════

/// Lookup key for WMMA intrinsic query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WmmaQuery {
    pub target: WmmaTarget,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub a_type: WmmaType,
    pub b_type: WmmaType,
    pub d_type: WmmaType,
}

/// Look up a specific WMMA intrinsic by exact (target, M, N, K, A, B, D).
/// Returns None if no matching intrinsic exists for the target.
pub fn lookup_exact(query: &WmmaQuery) -> Option<&'static WmmaIntrinsicInfo> {
    // Search GFX1250 extras first (higher K preferred when both exist)
    if query.target == WmmaTarget::GFX1250 {
        if let Some(info) = find_in(GFX1250_EXTRA_INTRINSICS, query) {
            return Some(info);
        }
    }
    // Search GFX1200 base table (shared by both targets)
    find_in(GFX1200_INTRINSICS, query)
}

/// Look up WMMA intrinsics by (target, M, N, A, B, D) without specifying K.
/// Returns all matching intrinsics sorted by K descending (prefer larger K).
pub fn lookup_by_shape(
    target: WmmaTarget, m: u8, n: u8,
    a_type: WmmaType, b_type: WmmaType, d_type: WmmaType,
) -> Vec<&'static WmmaIntrinsicInfo> {
    let mut results: Vec<&'static WmmaIntrinsicInfo> = GFX1200_INTRINSICS.iter()
        .filter(|info| matches_shape(info, m, n, a_type, b_type, d_type))
        .collect();

    if target == WmmaTarget::GFX1250 {
        for info in GFX1250_EXTRA_INTRINSICS {
            if matches_shape(info, m, n, a_type, b_type, d_type) {
                results.push(info);
            }
        }
    }

    // Sort by K descending — prefer wider K for throughput
    results.sort_by(|a, b| b.k.cmp(&a.k));
    results
}

/// Get all intrinsics available for a given target.
pub fn all_for_target(target: WmmaTarget) -> Vec<&'static WmmaIntrinsicInfo> {
    let mut all: Vec<&'static WmmaIntrinsicInfo> = GFX1200_INTRINSICS.iter().collect();
    if target == WmmaTarget::GFX1250 {
        all.extend(GFX1250_EXTRA_INTRINSICS.iter());
    }
    all
}

/// Count of intrinsics per target.
pub fn intrinsic_count(target: WmmaTarget) -> usize {
    GFX1200_INTRINSICS.len()
        + if target == WmmaTarget::GFX1250 { GFX1250_EXTRA_INTRINSICS.len() } else { 0 }
}

// ── Internal helpers ──

fn find_in(
    table: &'static [WmmaIntrinsicInfo],
    query: &WmmaQuery,
) -> Option<&'static WmmaIntrinsicInfo> {
    table.iter().find(|info|
        info.m == query.m
            && info.n == query.n
            && info.k == query.k
            && info.a_type == query.a_type
            && info.b_type == query.b_type
            && info.d_type == query.d_type
    )
}

fn matches_shape(
    info: &WmmaIntrinsicInfo,
    m: u8, n: u8,
    a_type: WmmaType, b_type: WmmaType, d_type: WmmaType,
) -> bool {
    info.m == m && info.n == n
        && info.a_type == a_type && info.b_type == b_type && info.d_type == d_type
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Encoding verification (llvm-mc ground truth) ──

    #[test]
    fn test_gfx1200_bf16_k16_encoding() {
        // llvm-mc: v_wmma_f32_16x16x16_bf16 v[0:7], v[8:11], v[16:19], v[24:31]
        //   → encoding: [0x00,0x40,0x41,0xcc, 0x08,0x21,0x62,0x1c]
        //   → word0=0xCC414000, word1=0x1C622108
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("bf16 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC414000, "word0 mismatch for bf16 K=16");
        assert_eq!(w1, 0x1C622108, "word1 mismatch for bf16 K=16");
    }

    #[test]
    fn test_gfx1200_f16_k16_encoding() {
        // llvm-mc: v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[16:19], v[24:31]
        //   → encoding: [0x00,0x40,0x40,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("f16 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC404000, "word0 mismatch for f16 K=16");
        assert_eq!(w1, 0x1C622108, "word1 mismatch for f16 K=16");
    }

    #[test]
    fn test_gfx1200_bf16_bf16_k16_encoding() {
        // llvm-mc: v_wmma_bf16_16x16x16_bf16 v[0:3], v[8:11], v[16:19], v[24:27]
        //   → encoding: [0x00,0x40,0x43,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::BF16,
        };
        let info = lookup_exact(&q).expect("bf16→bf16 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC434000, "word0 mismatch for bf16→bf16");
        assert_eq!(w1, 0x1C622108, "word1 mismatch for bf16→bf16");
        assert_eq!(info.cd_vgprs, 4, "bf16→bf16 uses 4 VGPR accumulator");
    }

    #[test]
    fn test_gfx1200_f16_f16_k16_encoding() {
        // llvm-mc: v_wmma_f16_16x16x16_f16 v[0:3], v[8:11], v[16:19], v[24:27]
        //   → encoding: [0x00,0x40,0x42,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F16,
        };
        let info = lookup_exact(&q).expect("f16→f16 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC424000, "word0 mismatch for f16→f16");
        assert_eq!(w1, 0x1C622108, "word1 mismatch for f16→f16");
    }

    #[test]
    fn test_gfx1200_iu8_k16_encoding() {
        // llvm-mc: v_wmma_i32_16x16x16_iu8 v[0:7], v[32:33], v[40:41], v[48:55]
        //   → encoding: [0x00,0x40,0x44,0xcc, 0x20,0x51,0xc2,0x1c]
        //   → word0=0xCC444000, word1=0x1CC25120
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::IU8, b_type: WmmaType::IU8, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("iu8 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 32, 40, 48);
        assert_eq!(w0, 0xCC444000, "word0 mismatch for iu8 K=16");
        assert_eq!(w1, 0x1CC25120, "word1 mismatch for iu8 K=16");
        assert_eq!(info.a_vgprs, 2, "iu8 A uses 2 VGPRs");
        assert_eq!(info.b_vgprs, 2, "iu8 B uses 2 VGPRs");
    }

    #[test]
    fn test_gfx1200_iu4_k16_encoding() {
        // llvm-mc: v_wmma_i32_16x16x16_iu4 v[0:7], v8, v16, v[24:31]
        //   → encoding: [0x00,0x40,0x45,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::IU4, b_type: WmmaType::IU4, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("iu4 K=16 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC454000, "word0 mismatch for iu4 K=16");
        assert_eq!(info.a_vgprs, 1, "iu4 K=16 A uses 1 VGPR");
    }

    #[test]
    fn test_gfx1200_iu4_k32_encoding() {
        // llvm-mc: v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[16:17], v[24:31]
        //   → encoding: [0x00,0x40,0x4a,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 32,
            a_type: WmmaType::IU4, b_type: WmmaType::IU4, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("iu4 K=32 must exist on gfx1200");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC4A4000, "word0 mismatch for iu4 K=32");
        assert_eq!(info.a_vgprs, 2, "iu4 K=32 A uses 2 VGPRs");
    }

    #[test]
    fn test_gfx1250_bf16_k32_encoding() {
        // llvm-mc: v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[24:31]
        //   → encoding: [0x00,0x00,0x62,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1250,
            m: 16, n: 16, k: 32,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("bf16 K=32 must exist on gfx1250");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC620000, "word0 mismatch for gfx1250 bf16 K=32");
        assert_eq!(w1, 0x1C622108, "word1 mismatch for gfx1250 bf16 K=32");
        assert_eq!(info.a_vgprs, 8, "gfx1250 K=32 uses 8 VGPRs for A");
    }

    #[test]
    fn test_gfx1250_f16_k32_encoding() {
        // llvm-mc: v_wmma_f32_16x16x32_f16 v[0:7], v[8:15], v[16:23], v[24:31]
        //   → encoding: [0x00,0x00,0x60,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1250,
            m: 16, n: 16, k: 32,
            a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("f16 K=32 must exist on gfx1250");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC600000, "word0 mismatch for gfx1250 f16 K=32");
    }

    #[test]
    fn test_gfx1250_iu8_k64_encoding() {
        // llvm-mc: v_wmma_i32_16x16x64_iu8 v[0:7], v[8:15], v[16:23], v[24:31]
        //   → encoding: [0x00,0x00,0x72,0xcc, 0x08,0x21,0x62,0x1c]
        let q = WmmaQuery {
            target: WmmaTarget::GFX1250,
            m: 16, n: 16, k: 64,
            a_type: WmmaType::IU8, b_type: WmmaType::IU8, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).expect("iu8 K=64 must exist on gfx1250");
        let [w0, w1] = info.encode(0, 8, 16, 24);
        assert_eq!(w0, 0xCC720000, "word0 mismatch for gfx1250 iu8 K=64");
        assert_eq!(info.k, 64, "K=64 for gfx1250 iu8");
    }

    // ── Negative tests ──

    #[test]
    fn test_gfx1200_no_k32_bf16() {
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 32,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        };
        assert!(lookup_exact(&q).is_none(), "bf16 K=32 NOT available on gfx1200");
    }

    #[test]
    fn test_gfx1200_no_k32_f16() {
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 32,
            a_type: WmmaType::F16, b_type: WmmaType::F16, d_type: WmmaType::F32,
        };
        assert!(lookup_exact(&q).is_none(), "f16 K=32 NOT available on gfx1200");
    }

    #[test]
    fn test_gfx1200_no_k64_iu8() {
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 64,
            a_type: WmmaType::IU8, b_type: WmmaType::IU8, d_type: WmmaType::F32,
        };
        assert!(lookup_exact(&q).is_none(), "iu8 K=64 NOT available on gfx1200");
    }

    // ── Shape lookup ──

    #[test]
    fn test_lookup_by_shape_prefers_larger_k() {
        // For iu4 on gfx1200, both K=16 and K=32 exist.
        // lookup_by_shape should return K=32 first.
        let results = lookup_by_shape(
            WmmaTarget::GFX1200, 16, 16,
            WmmaType::IU4, WmmaType::IU4, WmmaType::F32,
        );
        assert!(results.len() >= 2, "should have K=16 and K=32 iu4");
        assert_eq!(results[0].k, 32, "first result should prefer K=32");
        assert_eq!(results[1].k, 16, "second result should be K=16");
    }

    #[test]
    fn test_gfx1250_has_more_intrinsics() {
        assert_eq!(intrinsic_count(WmmaTarget::GFX1200), 11);
        assert_eq!(intrinsic_count(WmmaTarget::GFX1250), 22);
    }

    #[test]
    fn test_all_intrinsic_count() {
        let all = all_for_target(WmmaTarget::GFX1250);
        assert_eq!(all.len(), 22, "GFX1250 should have 22 WMMA intrinsics");
    }

    // ── High-register encoding test ──

    #[test]
    fn test_high_register_encoding() {
        // llvm-mc: v_wmma_f32_16x16x16_bf16 v[0:7], v[64:67], v[72:75], v[80:87]
        //   → encoding: [0x00,0x40,0x41,0xcc, 0x40,0x91,0x42,0x1d]
        //   → word0=0xCC414000, word1=0x1D429140
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).unwrap();
        let [w0, w1] = info.encode(0, 64, 72, 80);
        assert_eq!(w0, 0xCC414000, "word0 for high regs");
        assert_eq!(w1, 0x1D429140, "word1 for high regs (va=64,vb=72,vc=80)");
    }

    // ── Signature display ──

    #[test]
    fn test_signature_format() {
        let q = WmmaQuery {
            target: WmmaTarget::GFX1200,
            m: 16, n: 16, k: 16,
            a_type: WmmaType::BF16, b_type: WmmaType::BF16, d_type: WmmaType::F32,
        };
        let info = lookup_exact(&q).unwrap();
        assert_eq!(info.signature(), "f32_16x16x16_bf16");
    }

    // ── GFX1250 target detection ──

    #[test]
    fn test_target_from_gfx() {
        assert_eq!(WmmaTarget::from_gfx("gfx1200"), Some(WmmaTarget::GFX1200));
        assert_eq!(WmmaTarget::from_gfx("gfx1201"), Some(WmmaTarget::GFX1200));
        assert_eq!(WmmaTarget::from_gfx("gfx1250"), Some(WmmaTarget::GFX1250));
        assert_eq!(WmmaTarget::from_gfx("gfx1251"), Some(WmmaTarget::GFX1250));
        assert_eq!(WmmaTarget::from_gfx("gfx1100"), None);
    }
}
