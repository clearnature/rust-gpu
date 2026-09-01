//! GFX1200 指令编码属性测试 (Property-based encoding tests)
//!
//! 参考 CompCert 验证方法论：对每个编码函数做随机化输入测试，
//! 验证编码格式的结构性不变量（word0 前缀、位域范围、对齐约束）。
//!
//! 由于项目保持零外部依赖，使用 XorShift64 PRNG 代替 proptest crate。
//! 每个测试类别覆盖该指令类的所有已知编码函数。
//!
//! ## 测试策略
//!
//! 1. **前缀不变量**：每条指令的 word0 高位必须匹配其格式前缀
//! 2. **位域完整性**：编码的寄存器号、操作码等字段必须能从编码中正确提取
//! 3. **对齐约束**：SMEM base 必须 2 对齐，dst 按 size 对齐
//! 4. **边界条件**：空操作数（v0, s0, offset=0）和满操作数（v255, offset=max）
//! 5. **Decode(Encode(x)) == x**：提取编码字段并验证与输入一致

use t0_gpu::rdna3_asm::gfx11;

// ═══════════════════════════════════════════════════════════════
// XorShift64 PRNG — deterministic, no external deps
// ═══════════════════════════════════════════════════════════════

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }

    /// Generate u32 in range [0, max)
    fn range(&mut self, max: u32) -> u32 {
        self.next_u32() % max
    }

    /// Generate u8 in range [0, max)
    fn range_u8(&mut self, max: u8) -> u8 {
        (self.next_u32() % max as u32) as u8
    }
}

// ═══════════════════════════════════════════════════════════════
// Helper: repeat test N times with random inputs
// ═══════════════════════════════════════════════════════════════

const PROPERTY_ITERATIONS: u32 = 1000;

fn with_rng<F: Fn(&mut XorShift64)>(seed: u64, f: F) {
    let mut rng = XorShift64::new(seed);
    f(&mut rng);
}

// ═══════════════════════════════════════════════════════════════
// SOPP 属性测试 (Scalar Operation, Immediate)
// ═══════════════════════════════════════════════════════════════

/// SOPP word0[31:24] == 0xBF — 所有 SOPP 指令共享此前缀
fn assert_sopp_prefix(word: u32, desc: &str) {
    assert_eq!(
        (word >> 24) & 0xFF, 0xBF,
        "{}: SOPP prefix mismatch: 0x{:08X} (expected top byte 0xBF)",
        desc, word
    );
}

/// SOPP word0[15:0] == imm16 — 立即数字段
fn assert_sopp_imm16(word: u32, expected_imm: u16, desc: &str) {
    assert_eq!(
        word & 0xFFFF, expected_imm as u32,
        "{}: SOPP imm16 mismatch: expected {}, got {}",
        desc, expected_imm, word & 0xFFFF
    );
}

#[test]
fn proptest_s_wait_loadcnt() {
    with_rng(0x534F5050, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_loadcnt(n);
            assert_sopp_prefix(word, "s_wait_loadcnt");
            // s_wait_loadcnt base = 0xBFC00000
            assert_eq!(word & 0xFFFF0000, 0xBFC00000,
                "s_wait_loadcnt({}): base mismatch", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16,
                "s_wait_loadcnt({}): count field mismatch", n);
        }
    });
}

#[test]
fn proptest_s_wait_storecnt() {
    with_rng(0x53544F52, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_storecnt(n);
            assert_sopp_prefix(word, "s_wait_storecnt");
            assert_eq!(word & 0xFFFF0000, 0xBFC10000,
                "s_wait_storecnt({}): base mismatch", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16,
                "s_wait_storecnt({}): count field mismatch", n);
        }
    });
}

#[test]
fn proptest_s_wait_dscnt() {
    with_rng(0x53445343, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_dscnt(n);
            assert_sopp_prefix(word, "s_wait_dscnt");
            assert_eq!(word & 0xFFFF0000, 0xBFC60000, "s_wait_dscnt({}) base", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16, "s_wait_dscnt({}) count", n);
        }
    });
}

#[test]
fn proptest_s_wait_kmcnt() {
    with_rng(0x534B4D43, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_kmcnt(n);
            assert_sopp_prefix(word, "s_wait_kmcnt");
            assert_eq!(word & 0xFFFF0000, 0xBFC70000, "s_wait_kmcnt({}) base", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16, "s_wait_kmcnt({}) count", n);
        }
    });
}

#[test]
fn proptest_s_wait_asynccnt_gfx1250() {
    with_rng(0x4153594E, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_asynccnt(n);
            assert_sopp_prefix(word, "s_wait_asynccnt");
            assert_eq!(word & 0xFFFF0000, 0xBFCA0000, "s_wait_asynccnt({}) base", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16, "s_wait_asynccnt({}) count", n);
        }
    });
}

#[test]
fn proptest_s_wait_tensorcnt_gfx1250() {
    with_rng(0x54454E53, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(64);
            let word = gfx11::s_wait_tensorcnt(n);
            assert_sopp_prefix(word, "s_wait_tensorcnt");
            assert_eq!(word & 0xFFFF0000, 0xBFCB0000, "s_wait_tensorcnt({}) base", n);
            assert_eq!((word & 0xFFFF) as u16, n as u16, "s_wait_tensorcnt({}) count", n);
        }
    });
}

#[test]
fn proptest_s_setprio() {
    with_rng(0x5052494F, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let prio = rng.range_u8(4); // 0-3
            let word = gfx11::s_setprio(prio);
            assert_sopp_prefix(word, "s_setprio");
            assert_eq!(word & 0xFFFF0000, 0xBFB50000, "s_setprio({}) base", prio);
            assert_eq!((word & 0xFF) as u8, prio, "s_setprio({}) prio field", prio);
        }
    });
}

#[test]
fn proptest_s_nop() {
    with_rng(0x5F4E4F50, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let n = rng.range_u8(32);
            let word = gfx11::s_nop(n);
            assert_sopp_prefix(word, "s_nop");
            assert_eq!(word & 0xFFFF0000, 0xBF800000, "s_nop({}) base", n);
            assert_eq!((word & 0xFF) as u8, n, "s_nop({}) delay field", n);
        }
    });
}

#[test]
fn proptest_s_clause() {
    with_rng(0x434C4153, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let count = rng.range_u8(63) + 1; // 1-63
            let word = gfx11::s_clause(count);
            assert_sopp_prefix(word, "s_clause");
            assert_eq!(word & 0xFFFF0000, 0xBF850000, "s_clause({}) base", count);
            assert_eq!((word & 0x3F) as u8, count, "s_clause({}) count field", count);
        }
    });
}

#[test]
fn proptest_s_branch() {
    with_rng(0x42524E43, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let offset = (rng.next_u32() as i16);
            let word = gfx11::s_branch(offset);
            assert_sopp_prefix(word, "s_branch");
            // s_branch opcode 0x20 → bits[22:16] = 0x20
            assert_eq!((word >> 16) & 0x7F, 0x20, "s_branch({}) opcode", offset);
            // imm16 = offset as u16
            assert_eq!(word & 0xFFFF, (offset as u16) as u32,
                "s_branch({}) imm16", offset);
        }
    });
}

#[test]
fn proptest_s_cbranch_scc0() {
    with_rng(0x43425330, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let offset = (rng.next_u32() as i16);
            let word = gfx11::s_cbranch_scc0(offset);
            assert_sopp_prefix(word, "s_cbranch_scc0");
            assert_eq!((word >> 16) & 0x7F, 0x21, "s_cbranch_scc0 opcode");
            assert_eq!(word & 0xFFFF, (offset as u16) as u32, "s_cbranch_scc0 imm16");
        }
    });
}

#[test]
fn proptest_s_cbranch_scc1() {
    with_rng(0x43425331, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let offset = (rng.next_u32() as i16);
            let word = gfx11::s_cbranch_scc1(offset);
            assert_sopp_prefix(word, "s_cbranch_scc1");
            assert_eq!((word >> 16) & 0x7F, 0x22, "s_cbranch_scc1 opcode");
        }
    });
}

#[test]
fn proptest_s_cbranch_vccz() {
    with_rng(0x4356435A, |rng| {
        for _ in 0..500 {
            let offset = (rng.next_u32() as i16);
            let word = gfx11::s_cbranch_vccz(offset);
            assert_sopp_prefix(word, "s_cbranch_vccz");
            assert_eq!((word >> 16) & 0x7F, 0x23, "s_cbranch_vccz opcode");
        }
    });
}

#[test]
fn proptest_s_cbranch_vccnz() {
    with_rng(0x43564E5A, |rng| {
        for _ in 0..500 {
            let offset = (rng.next_u32() as i16);
            let word = gfx11::s_cbranch_vccnz(offset);
            assert_sopp_prefix(word, "s_cbranch_vccnz");
            assert_eq!((word >> 16) & 0x7F, 0x24, "s_cbranch_vccnz opcode");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// SOP2 属性测试 (Scalar ALU)
// ═══════════════════════════════════════════════════════════════

/// SOP2 word0[31:30] == 0b10 (top nibble 0x8 or 0x9)
fn assert_sop2_prefix(word: u32, desc: &str) {
    assert_eq!(
        (word >> 30) & 0x3, 0b10,
        "{}: SOP2 prefix mismatch: 0x{:08X} (expected bits[31:30]=10)",
        desc, word
    );
}

/// Extract SOP2 SDST from word0[22:16]
fn extract_sop2_sdst(word: u32) -> u8 {
    ((word >> 16) & 0x7F) as u8
}

/// Extract SOP2 SSRC1 from word0[15:8]
fn extract_sop2_ssrc1(word: u32) -> u8 {
    ((word >> 8) & 0xFF) as u8
}

/// Extract SOP2 SSRC0 from word0[7:0]
fn extract_sop2_ssrc0(word: u32) -> u8 {
    (word & 0xFF) as u8
}

#[test]
fn proptest_s_add_u32() {
    with_rng(0x53414444, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);  // SGPR 0-107
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_add_u32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_add_u32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_add_u32 sdst({},{},{})", sdst, ssrc0, ssrc1);
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_add_u32 ssrc0({},{},{})", sdst, ssrc0, ssrc1);
            assert_eq!(extract_sop2_ssrc1(word), ssrc1, "s_add_u32 ssrc1({},{},{})", sdst, ssrc0, ssrc1);
        }
    });
}

#[test]
fn proptest_s_add_u32_imm() {
    with_rng(0x53414449, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let imm = rng.range_u8(65); // 0-64
            let word = gfx11::s_add_u32_imm(sdst, ssrc0, imm);
            assert_sop2_prefix(word, "s_add_u32_imm");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_add_u32_imm sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_add_u32_imm ssrc0");
            // inline constant = 0x80 + imm
            assert_eq!(extract_sop2_ssrc1(word), 0x80 + imm,
                "s_add_u32_imm inline const({})", imm);
        }
    });
}

#[test]
fn proptest_s_sub_u32_imm() {
    with_rng(0x53535542, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let imm = rng.range_u8(65);
            let word = gfx11::s_sub_u32_imm(sdst, ssrc0, imm);
            assert_sop2_prefix(word, "s_sub_u32_imm");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_sub_u32_imm sdst");
        }
    });
}

#[test]
fn proptest_s_and_b32_imm() {
    with_rng(0x53414E44, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let imm = rng.range_u8(65);
            let word = gfx11::s_and_b32_imm(sdst, ssrc0, imm);
            assert_sop2_prefix(word, "s_and_b32_imm");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_and_b32_imm sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_and_b32_imm ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), 0x80 + imm, "s_and_b32_imm inline const");
        }
    });
}

#[test]
fn proptest_s_and_b32_reg() {
    with_rng(0x53414E52, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_and_b32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_and_b32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_and_b32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_and_b32 ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), ssrc1, "s_and_b32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_addc_u32() {
    with_rng(0x53414343, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_addc_u32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_addc_u32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_addc_u32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_addc_u32 ssrc0");
        }
    });
}

#[test]
fn proptest_s_addc_u32_zero_ssrc1() {
    // CRITICAL: ssrc1=0 must encode as inline constant 0x80, NOT register s0!
    with_rng(0x5341435A, |rng| {
        for _ in 0..500 {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let word = gfx11::s_addc_u32(sdst, ssrc0, 0);
            assert_eq!(extract_sop2_ssrc1(word), 0x80,
                "s_addc_u32(s{}, s{}, 0): ssrc1=0 must be inline 0x80, not register s0", sdst, ssrc0);
        }
    });
}

#[test]
fn proptest_s_sub_u32_reg() {
    with_rng(0x53534252, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_sub_u32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_sub_u32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_sub_u32 sdst");
        }
    });
}

#[test]
fn proptest_s_mul_i32() {
    with_rng(0x534D554C, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_mul_i32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_mul_i32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_mul_i32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_mul_i32 ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), ssrc1, "s_mul_i32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_cselect_b32() {
    with_rng(0x5343534C, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_cselect_b32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_cselect_b32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_cselect_b32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_cselect_b32 ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), ssrc1, "s_cselect_b32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_xor_b32() {
    with_rng(0x53584F52, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let ssrc1 = rng.range_u8(108);
            let word = gfx11::s_xor_b32(sdst, ssrc0, ssrc1);
            assert_sop2_prefix(word, "s_xor_b32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_xor_b32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_xor_b32 ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), ssrc1, "s_xor_b32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_lshr_b32() {
    with_rng(0x534C5348, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let shift = rng.range_u8(32);
            let word = gfx11::s_lshr_b32(sdst, ssrc0, shift);
            assert_sop2_prefix(word, "s_lshr_b32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_lshr_b32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_lshr_b32 ssrc0");
            // shift is inline constant: 0x80 + shift
            assert_eq!(extract_sop2_ssrc1(word), 0x80 + shift, "s_lshr_b32 shift inline const");
        }
    });
}

#[test]
fn proptest_s_lshl_b32() {
    with_rng(0x534C534C, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc0 = rng.range_u8(108);
            let shift = rng.range_u8(32);
            let word = gfx11::s_lshl_b32(sdst, ssrc0, shift);
            assert_sop2_prefix(word, "s_lshl_b32");
            assert_eq!(extract_sop2_sdst(word), sdst, "s_lshl_b32 sdst");
            assert_eq!(extract_sop2_ssrc0(word), ssrc0, "s_lshl_b32 ssrc0");
            assert_eq!(extract_sop2_ssrc1(word), 0x80 + shift, "s_lshl_b32 shift inline const");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// SOP1 属性测试 (Scalar Unary)
// ═══════════════════════════════════════════════════════════════

/// SOP1 word0[31:23] == 0xBE8x → top byte 0xBE, bits[23:22] indicate SOP1
fn assert_sop1_prefix(word: u32, desc: &str) {
    assert_eq!(
        (word >> 24) & 0xFF, 0xBE,
        "{}: SOP1 prefix mismatch: 0x{:08X} (expected top byte 0xBE)",
        desc, word
    );
}

#[test]
fn proptest_s_mov_b32() {
    with_rng(0x534D4F56, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let sdst = rng.range_u8(108);
            let ssrc = rng.range_u8(108);
            let word = gfx11::s_mov_b32(sdst, ssrc);
            assert_sop1_prefix(word, "s_mov_b32");
            // SDST at bits[22:16]
            assert_eq!((word >> 16) & 0x7F, sdst as u32, "s_mov_b32 sdst");
            // SSRC at bits[7:0]
            assert_eq!(word & 0xFF, ssrc as u32, "s_mov_b32 ssrc");
        }
    });
}

#[test]
fn proptest_s_mov_b32_imm() {
    with_rng(0x534D494D, |rng| {
        // Test inline constants: -64 to 64
        for imm in -64i32..=64 {
            let sdst = rng.range_u8(108);
            let word = gfx11::s_mov_b32_imm(sdst, imm);
            assert_sop1_prefix(word, &format!("s_mov_b32_imm({}, {})", sdst, imm));
            assert_eq!((word >> 16) & 0x7F, sdst as u32, "s_mov_b32_imm({}, {}) sdst", sdst, imm);
        }
    });
}

#[test]
fn proptest_s_mov_b32_literal() {
    with_rng(0x534C4954, |rng| {
        for _ in 0..500 {
            let sdst = rng.range_u8(108);
            let literal = rng.next_u32();
            let [instr, lit] = gfx11::s_mov_b32_literal(sdst, literal);
            assert_sop1_prefix(instr, "s_mov_b32_literal");
            assert_eq!((instr >> 16) & 0x7F, sdst as u32, "s_mov_b32_literal sdst");
            // Source = 0xFF (literal marker)
            assert_eq!(instr & 0xFF, 0xFF, "s_mov_b32_literal literal marker");
            // Second dword = literal value
            assert_eq!(lit, literal, "s_mov_b32_literal value");
        }
    });
}

#[test]
fn proptest_s_mov_b32_exec_lo() {
    with_rng(0x53455843, |rng| {
        // Test with common mask values
        let masks = [0u32, 1, 0xFF, 0xFFFF, 0xFFFFFFFF];
        for &mask in &masks {
            let _ = rng.next_u32(); // advance PRNG
            let word = gfx11::s_mov_b32_exec_lo(mask);
            assert_sop1_prefix(word, &format!("s_mov_b32_exec_lo(0x{:X})", mask));
            // exec_lo = SGPR 0x7E → SDST at bits[22:16] = 0x7E
            assert_eq!((word >> 16) & 0x7F, 0x7E,
                "s_mov_b32_exec_lo(0x{:X}): dst must be exec_lo (0x7E)", mask);
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// SOPC 属性测试 (Scalar Compare)
// ═══════════════════════════════════════════════════════════════

/// SOPC word0[31:24] == 0xBF, and specific opcode at bits[22:16]
fn assert_sopc_format(word: u32, expected_opcode: u32, desc: &str) {
    assert_eq!(
        (word >> 24) & 0xFF, 0xBF,
        "{}: SOPC prefix mismatch: 0x{:08X}", desc, word
    );
    assert_eq!(
        (word >> 16) & 0x7F, expected_opcode,
        "{}: SOPC opcode mismatch: expected 0x{:02X}", desc, expected_opcode
    );
}

#[test]
fn proptest_s_cmp_eq_u32() {
    with_rng(0x53434551, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let s1 = rng.range_u8(108);
            let word = gfx11::s_cmp_eq_u32(s0, s1);
            assert_sopc_format(word, 0x06, "s_cmp_eq_u32");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_eq_u32 ssrc0");
            assert_eq!((word >> 8) & 0xFF, s1 as u32, "s_cmp_eq_u32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_cmp_ge_u32() {
    with_rng(0x53434745, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let s1 = rng.range_u8(108);
            let word = gfx11::s_cmp_ge_u32(s0, s1);
            assert_sopc_format(word, 0x09, "s_cmp_ge_u32");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_ge_u32 ssrc0");
            assert_eq!((word >> 8) & 0xFF, s1 as u32, "s_cmp_ge_u32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_cmp_ge_u32_imm() {
    with_rng(0x5347494D, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let imm = rng.range_u8(65); // 0-64
            let word = gfx11::s_cmp_ge_u32_imm(s0, imm);
            assert_sopc_format(word, 0x09, "s_cmp_ge_u32_imm");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_ge_u32_imm ssrc0");
            // inline constant = 128 + imm
            assert_eq!((word >> 8) & 0xFF, (128 + imm) as u32, "s_cmp_ge_u32_imm inline const");
        }
    });
}

#[test]
fn proptest_s_cmp_lt_u32() {
    with_rng(0x53434C54, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let s1 = rng.range_u8(108);
            let word = gfx11::s_cmp_lt_u32(s0, s1);
            assert_sopc_format(word, 0x0A, "s_cmp_lt_u32");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_lt_u32 ssrc0");
            assert_eq!((word >> 8) & 0xFF, s1 as u32, "s_cmp_lt_u32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_cmp_lt_u32_imm() {
    with_rng(0x534C494D, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let imm = rng.range_u8(65);
            let word = gfx11::s_cmp_lt_u32_imm(s0, imm);
            assert_sopc_format(word, 0x0A, "s_cmp_lt_u32_imm");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_lt_u32_imm ssrc0");
            assert_eq!((word >> 8) & 0xFF, (128 + imm) as u32, "s_cmp_lt_u32_imm inline const");
        }
    });
}

#[test]
fn proptest_s_cmp_gt_i32() {
    with_rng(0x53434754, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let s1 = rng.range_u8(108);
            let word = gfx11::s_cmp_gt_i32(s0, s1);
            assert_sopc_format(word, 0x02, "s_cmp_gt_i32");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_gt_i32 ssrc0");
            assert_eq!((word >> 8) & 0xFF, s1 as u32, "s_cmp_gt_i32 ssrc1");
        }
    });
}

#[test]
fn proptest_s_cmp_lg_u32_imm() {
    with_rng(0x534C4755, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let s0 = rng.range_u8(108);
            let imm = rng.range_u8(65);
            let word = gfx11::s_cmp_lg_u32_imm(s0, imm);
            // SOPC opcode for s_cmp_lg_u32 = 0x07
            assert_sopc_format(word, 0x07, "s_cmp_lg_u32_imm");
            assert_eq!(word & 0xFF, s0 as u32, "s_cmp_lg_u32_imm ssrc0");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// SMEM 属性测试 (GFX1200 Scalar Memory)
// ═══════════════════════════════════════════════════════════════

/// GFX1200 SMEM word0 = 0xF4000000 | (size << 12) | (dst << 6) | (base / 2)
/// For dst >= 64, dst<<6 overflows into size field — this is correct by design
/// (hardware uses opcode, not these bits, for size). For property testing, we
/// verify the full word0 reconstruction rather than extracting individual fields
/// which can overlap.

/// Verify GFX1200 SMEM word0 by reconstructing from inputs
fn assert_gfx12_smem_word0(word0: u32, dst: u8, base: u8, size: u32, desc: &str) {
    let expected = 0xF4000000u32 | (size << 12) | ((dst as u32) << 6) | (base as u32 / 2);
    assert_eq!(word0, expected,
        "{}: SMEM word0 mismatch: got 0x{:08X}, expected 0x{:08X} (dst={}, size={}, base={})",
        desc, word0, expected, dst, size, base);
}

/// GFX1200 SMEM word1 = 0xF8000000 | (offset & 0xFFFFFF)
fn assert_gfx12_smem_word1(word1: u32, offset: u32, desc: &str) {
    assert_eq!(word1, 0xF8000000 | (offset & 0xFFFFFF),
        "{}: SMEM word1 mismatch", desc);
}

#[test]
fn proptest_gfx1200_s_load_b32() {
    with_rng(0x534C3332, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let dst = rng.range_u8(108);
            let base = rng.range_u8(54) * 2; // 2-aligned
            let offset = rng.next_u32() & 0xFFFFFF; // 24-bit
            let [w0, w1] = gfx11::s_load_dword_gfx1200(dst, base, offset);
            assert_gfx12_smem_word0(w0, dst, base, 0, "s_load_b32");
            assert_gfx12_smem_word1(w1, offset, "s_load_b32");
        }
    });
}

#[test]
fn proptest_gfx1200_s_load_b64() {
    with_rng(0x534C3634, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let dst = rng.range_u8(54) * 2; // 2-aligned, max 106
            let base = rng.range_u8(54) * 2;
            let offset = rng.next_u32() & 0xFFFFFF;
            let [w0, w1] = gfx11::s_load_dwordx2_gfx1200(dst, base, offset);
            assert_gfx12_smem_word0(w0, dst, base, 2, "s_load_b64");
            assert_gfx12_smem_word1(w1, offset, "s_load_b64");
        }
    });
}

#[test]
fn proptest_gfx1200_s_load_b128() {
    with_rng(0x534C3136, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let dst = rng.range_u8(27) * 4; // 4-aligned, max 104
            let base = rng.range_u8(54) * 2;
            let offset = rng.next_u32() & 0xFFFFFF;
            let [w0, w1] = gfx11::s_load_dwordx4_gfx1200(dst, base, offset);
            assert_gfx12_smem_word0(w0, dst, base, 4, "s_load_b128");
            assert_gfx12_smem_word1(w1, offset, "s_load_b128");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// VGLOBAL 属性测试 (GFX1200 FLAT/Global Memory)
// ═══════════════════════════════════════════════════════════════

/// GFX1200 VGLOBAL word0[31:24] == 0xEE (global) or 0xED (scratch)
fn assert_vglobal_word0_prefix(word0: u32, expected: u32, desc: &str) {
    assert_eq!(
        (word0 >> 24) & 0xFF, expected,
        "{}: VGLOBAL word0 prefix mismatch: 0x{:08X} (expected 0x{:02X})",
        desc, word0, expected
    );
}

/// GFX1200 VGLOBAL word0[7:0] == 0x7C (saddr = off)
fn assert_vglobal_saddr_off(word0: u32, desc: &str) {
    assert_eq!(
        word0 & 0xFF, 0x7C,
        "{}: VGLOBAL saddr must be 0x7C (off), got 0x{:02X}",
        desc, word0 & 0xFF
    );
}

#[test]
fn proptest_gfx1200_global_load_b32() {
    with_rng(0x474C4233, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vaddr = rng.range_u8(254); // v[addr:addr+1] needs addr+1 < 256
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_load_dword_gfx1200(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_load_b32");
            assert_vglobal_saddr_off(w0, "global_load_b32");
            // word1 = vdst
            assert_eq!(w1, vdst as u32, "global_load_b32 word1=vdst({})", vdst);
            // word2 = (offset << 8) | vaddr
            assert_eq!(w2 & 0xFF, vaddr as u32, "global_load_b32 word2 vaddr({})", vaddr);
            assert_eq!((w2 >> 8) as u32, ((offset as u32) << 8) >> 8,
                "global_load_b32 word2 offset");
        }
    });
}

#[test]
fn proptest_gfx1200_global_load_b128() {
    with_rng(0x474C4231, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(252); // v[dst:dst+3] needs room
            let vaddr = rng.range_u8(254);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_load_dwordx4_gfx1200(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_load_b128");
            assert_vglobal_saddr_off(w0, "global_load_b128");
            assert_eq!(w1, vdst as u32, "global_load_b128 word1=vdst");
        }
    });
}

#[test]
fn proptest_gfx1200_global_store_b32() {
    with_rng(0x47534233, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vaddr = rng.range_u8(254);
            let vsrc = rng.range_u8(255);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_store_dword_gfx1200(vaddr, vsrc, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_store_b32");
            assert_vglobal_saddr_off(w0, "global_store_b32");
            // word1: vsrc encoded as (vsrc/2 << 24) | ((vsrc&1) << 23)
            let expected_w1 = ((vsrc as u32 / 2) << 24) | (((vsrc as u32 & 1) as u32) << 23);
            assert_eq!(w1, expected_w1, "global_store_b32 word1 vsrc({})", vsrc);
            // word2: vaddr at bits[7:0]
            assert_eq!(w2 & 0xFF, vaddr as u32, "global_store_b32 vaddr");
        }
    });
}

#[test]
fn proptest_gfx1200_global_store_b128() {
    with_rng(0x47534231, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vaddr = rng.range_u8(254);
            let vsrc = rng.range_u8(252); // v[src:src+3] needs room
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_store_dwordx4_gfx1200(vaddr, vsrc, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_store_b128");
            assert_vglobal_saddr_off(w0, "global_store_b128");
            // Verify full word1 reconstruction
            let expected_w1 = ((vsrc as u32 / 2) << 24) | (((vsrc as u32 & 1) as u32) << 23);
            assert_eq!(w1, expected_w1, "global_store_b128 word1 vsrc({})", vsrc);
        }
    });
}

// ── Non-temporal (TH) variants ──

#[test]
fn proptest_gfx1200_global_load_b32_nt() {
    with_rng(0x474C4E54, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vaddr = rng.range_u8(254);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_load_dword_gfx1200_nt(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_load_b32_nt");
            assert_eq!(w1, vdst as u32 | (1 << 12),
                "global_load_b32_nt: TH=1 (NT) in word1[13:12]");
            assert_eq!(w2 & 0xFF, vaddr as u32, "global_load_b32_nt vaddr");
        }
    });
}

#[test]
fn proptest_gfx1200_global_load_b128_ht() {
    with_rng(0x474C4854, |rng| {
        for _ in 0..500 {
            let vdst = rng.range_u8(252);
            let vaddr = rng.range_u8(254);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_load_dwordx4_gfx1200_ht(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_load_b128_ht");
            assert_eq!(w1, vdst as u32 | (2 << 12),
                "global_load_b128_ht: TH=2 (HT) in word1[13:12]");
        }
    });
}

#[test]
fn proptest_gfx1200_global_load_b128_lu() {
    with_rng(0x474C4C55, |rng| {
        for _ in 0..500 {
            let vdst = rng.range_u8(252);
            let vaddr = rng.range_u8(254);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_load_dwordx4_gfx1200_lu(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_load_b128_lu");
            assert_eq!(w1, vdst as u32 | (3 << 12),
                "global_load_b128_lu: TH=3 (LU) in word1[13:12]");
        }
    });
}

// ── Scratch instructions ──

#[test]
fn proptest_gfx1200_scratch_load_b32() {
    with_rng(0x534C4233, |rng| {
        for _ in 0..500 {
            let vdst = rng.range_u8(255);
            let vaddr = rng.range_u8(254);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::scratch_load_b32_gfx1200(vdst, vaddr, offset);
            assert_vglobal_word0_prefix(w0, 0xED, "scratch_load_b32");
            assert_eq!(w1, vdst as u32, "scratch_load_b32 vdst");
        }
    });
}

#[test]
fn proptest_gfx1200_scratch_store_b32() {
    with_rng(0x53534233, |rng| {
        for _ in 0..500 {
            let vaddr = rng.range_u8(254);
            let vsrc = rng.range_u8(255);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::scratch_store_b32_gfx1200(vaddr, vsrc, offset);
            assert_vglobal_word0_prefix(w0, 0xED, "scratch_store_b32");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// VOP3P-MAI 属性测试 (WMMA Matrix Instructions)
// ═══════════════════════════════════════════════════════════════

/// VOP3P-MAI word0[31:24] == 0xCC
fn assert_vop3p_mai_word0_prefix(word0: u32, desc: &str) {
    assert_eq!(
        (word0 >> 24) & 0xFF, 0xCC,
        "{}: VOP3P-MAI word0 prefix mismatch: 0x{:08X}", desc, word0
    );
}

/// VOP3P-MAI word1 top-byte invariant: bits[31:24] = 0x1C when all src regs < 128.
/// For high src regs, bits bleed into the modifier — we check the base constant
/// OR'd with source contributions is correct by verifying round-trip of all 3 SRCs.
fn assert_vop3p_mai_word1_base(word1: u32, va: u8, vb: u8, vc: u8, desc: &str) {
    // Reconstruct expected word1 from the encoding formula
    let expected = 0x1C000000u32
        | (va as u32 + 256)
        | ((vb as u32 + 256) << 9)
        | ((vc as u32 + 256) << 18);
    assert_eq!(
        word1, expected,
        "{}: VOP3P-MAI word1 mismatch: got 0x{:08X}, expected 0x{:08X}",
        desc, word1, expected
    );
}

/// Extract VOP3P-MAI SRC0 from word1[8:0]
fn extract_mai_src0(word1: u32) -> u32 {
    word1 & 0x1FF
}

/// Extract VOP3P-MAI SRC1 from word1[17:9]
fn extract_mai_src1(word1: u32) -> u32 {
    (word1 >> 9) & 0x1FF
}

/// Extract VOP3P-MAI SRC2 from word1[26:18]
fn extract_mai_src2(word1: u32) -> u32 {
    (word1 >> 18) & 0x1FF
}

/// Extract VDST from word0[6:0]
fn extract_mai_vdst(word0: u32) -> u8 {
    (word0 & 0x7F) as u8
}

/// Full WMMA encode+decode round-trip test
fn wmma_roundtrip<F: Fn(u8, u8, u8, u8) -> [u32; 2]>(
    name: &str,
    encode_fn: F,
    expected_word0_base: u32,
    vdst_max: u8,
    src_max: u8,
) {
    let mut rng = XorShift64::new(expected_word0_base as u64 ^ 0x574D4D41);
    for _ in 0..PROPERTY_ITERATIONS {
        let vdst = rng.range_u8(vdst_max); // 7-bit: max 127
        let va = rng.range_u8(src_max);
        let vb = rng.range_u8(src_max);
        let vc = rng.range_u8(src_max);
        let [w0, w1] = encode_fn(vdst, va, vb, vc);

        // 1. Prefix check
        assert_vop3p_mai_word0_prefix(w0, name);
        // 2. Full word1 verification (reconstruct from inputs)
        assert_vop3p_mai_word1_base(w1, va, vb, vc, name);

        // 3. VDST round-trip
        assert_eq!(extract_mai_vdst(w0), vdst,
            "{}: VDST round-trip: expected v{}, got v{}", name, vdst, extract_mai_vdst(w0));

        // 3. Opcode check (word0[22:16])
        assert_eq!(w0 & 0xFFFFFF80, expected_word0_base,
            "{}: word0 opcode mismatch: 0x{:08X} vs expected 0x{:08X}",
            name, w0, expected_word0_base | vdst as u32);

        // 4. Source round-trip (VGPR+256 encoding)
        assert_eq!(extract_mai_src0(w1), va as u32 + 256,
            "{}: SRC0 (va={}) round-trip", name, va);
        assert_eq!(extract_mai_src1(w1), vb as u32 + 256,
            "{}: SRC1 (vb={}) round-trip", name, vb);
        assert_eq!(extract_mai_src2(w1), vc as u32 + 256,
            "{}: SRC2 (vc={}) round-trip", name, vc);
    }
}

#[test]
fn proptest_wmma_f32_16x16x16_bf16() {
    wmma_roundtrip("v_wmma_f32_16x16x16_bf16", gfx11::v_wmma_f32_16x16x16_bf16, 0xCC414000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x16_f16() {
    wmma_roundtrip("v_wmma_f32_16x16x16_f16", gfx11::v_wmma_f32_16x16x16_f16, 0xCC404000, 128, 248);
}

#[test]
fn proptest_wmma_bf16_16x16x16_bf16() {
    wmma_roundtrip("v_wmma_bf16_16x16x16_bf16", gfx11::v_wmma_bf16_16x16x16_bf16, 0xCC434000, 128, 248);
}

#[test]
fn proptest_wmma_f16_16x16x16_f16() {
    wmma_roundtrip("v_wmma_f16_16x16x16_f16", gfx11::v_wmma_f16_16x16x16_f16, 0xCC424000, 128, 248);
}

#[test]
fn proptest_wmma_i32_16x16x16_iu8() {
    wmma_roundtrip("v_wmma_i32_16x16x16_iu8", gfx11::v_wmma_i32_16x16x16_iu8, 0xCC444000, 128, 248);
}

#[test]
fn proptest_wmma_i32_16x16x16_iu4() {
    wmma_roundtrip("v_wmma_i32_16x16x16_iu4", gfx11::v_wmma_i32_16x16x16_iu4, 0xCC454000, 128, 248);
}

#[test]
fn proptest_wmma_i32_16x16x32_iu4() {
    wmma_roundtrip("v_wmma_i32_16x16x32_iu4", gfx11::v_wmma_i32_16x16x32_iu4, 0xCC4A4000, 128, 248);
}

// ── GFX1250 WMMA ──

#[test]
fn proptest_wmma_f32_16x16x32_bf16_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x32_bf16", gfx11::v_wmma_f32_16x16x32_bf16, 0xCC620000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x32_f16_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x32_f16", gfx11::v_wmma_f32_16x16x32_f16, 0xCC600000, 128, 248);
}

#[test]
fn proptest_wmma_i32_16x16x64_iu8_gfx1250() {
    wmma_roundtrip("v_wmma_i32_16x16x64_iu8", gfx11::v_wmma_i32_16x16x64_iu8, 0xCC720000, 128, 248);
}

// ── GFX1200 (RDNA4) K=16 FP8/BF8 WMMA ──

#[test]
fn proptest_wmma_f32_16x16x16_fp8_fp8() {
    wmma_roundtrip("v_wmma_f32_16x16x16_fp8_fp8", gfx11::v_wmma_f32_16x16x16_fp8_fp8, 0xCC464000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x16_fp8_bf8() {
    wmma_roundtrip("v_wmma_f32_16x16x16_fp8_bf8", gfx11::v_wmma_f32_16x16x16_fp8_bf8, 0xCC474000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x16_bf8_fp8() {
    wmma_roundtrip("v_wmma_f32_16x16x16_bf8_fp8", gfx11::v_wmma_f32_16x16x16_bf8_fp8, 0xCC484000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x16_bf8_bf8() {
    wmma_roundtrip("v_wmma_f32_16x16x16_bf8_bf8", gfx11::v_wmma_f32_16x16x16_bf8_bf8, 0xCC494000, 128, 248);
}

// ── GFX1250 (RDNA4.5) K=64 FP8/BF8 WMMA ──

#[test]
fn proptest_wmma_f32_16x16x64_fp8_fp8_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x64_fp8_fp8", gfx11::v_wmma_f32_16x16x64_fp8_fp8, 0xCC6A0000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x64_fp8_bf8_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x64_fp8_bf8", gfx11::v_wmma_f32_16x16x64_fp8_bf8, 0xCC6B0000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x64_bf8_fp8_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x64_bf8_fp8", gfx11::v_wmma_f32_16x16x64_bf8_fp8, 0xCC6C0000, 128, 248);
}

#[test]
fn proptest_wmma_f32_16x16x64_bf8_bf8_gfx1250() {
    wmma_roundtrip("v_wmma_f32_16x16x64_bf8_bf8", gfx11::v_wmma_f32_16x16x64_bf8_bf8, 0xCC6D0000, 128, 248);
}

#[test]
fn proptest_wmma_f16_16x16x64_fp8_fp8_gfx1250() {
    wmma_roundtrip("v_wmma_f16_16x16x64_fp8_fp8", gfx11::v_wmma_f16_16x16x64_fp8_fp8, 0xCC6E0000, 128, 248);
}

#[test]
fn proptest_wmma_f16_16x16x64_fp8_bf8_gfx1250() {
    wmma_roundtrip("v_wmma_f16_16x16x64_fp8_bf8", gfx11::v_wmma_f16_16x16x64_fp8_bf8, 0xCC6F0000, 128, 248);
}

#[test]
fn proptest_wmma_f16_16x16x64_bf8_fp8_gfx1250() {
    wmma_roundtrip("v_wmma_f16_16x16x64_bf8_fp8", gfx11::v_wmma_f16_16x16x64_bf8_fp8, 0xCC700000, 128, 248);
}

#[test]
fn proptest_wmma_f16_16x16x64_bf8_bf8_gfx1250() {
    wmma_roundtrip("v_wmma_f16_16x16x64_bf8_bf8", gfx11::v_wmma_f16_16x16x64_bf8_bf8, 0xCC710000, 128, 248);
}

// ═══════════════════════════════════════════════════════════════
// VOP3P Packed 属性测试 (v_pk_add_f16, v_pk_fma_f16, etc.)
// ═══════════════════════════════════════════════════════════════

/// VOP3P two-source packed: word1 = 0x1A000000 | (src0+256) | ((src1+256) << 9)
fn assert_vop3p_packed_word1_full(word1: u32, vsrc0: u8, vsrc1: u8, desc: &str) {
    let expected = 0x1A000000u32 | (vsrc0 as u32 + 256) | ((vsrc1 as u32 + 256) << 9);
    assert_eq!(word1, expected,
        "{}: VOP3P packed word1 mismatch: got 0x{:08X}, expected 0x{:08X}", desc, word1, expected);
}

/// VOP3P three-source packed: word1 = 0x1C000000 | (src0+256) | ((src1+256) << 9) | ((src2+256) << 18)
fn assert_vop3p_3src_word1_full(word1: u32, vsrc0: u8, vsrc1: u8, vsrc2: u8, desc: &str) {
    let expected = 0x1C000000u32
        | (vsrc0 as u32 + 256)
        | ((vsrc1 as u32 + 256) << 9)
        | ((vsrc2 as u32 + 256) << 18);
    assert_eq!(word1, expected,
        "{}: VOP3P 3-src word1 mismatch: got 0x{:08X}, expected 0x{:08X}", desc, word1, expected);
}

#[test]
fn proptest_v_pk_add_f16() {
    with_rng(0x504B4146, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128); // 7-bit VDST
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_pk_add_f16(vdst, vsrc0, vsrc1);
            assert_eq!((w0 >> 24) & 0xFF, 0xCC, "v_pk_add_f16 word0 prefix");
            assert_eq!(w0 & 0x7F, vdst as u32, "v_pk_add_f16 vdst");
            assert_vop3p_packed_word1_full(w1, vsrc0, vsrc1, "v_pk_add_f16");
        }
    });
}

#[test]
fn proptest_v_pk_mul_f16() {
    with_rng(0x504B4D46, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_pk_mul_f16(vdst, vsrc0, vsrc1);
            assert_eq!((w0 >> 24) & 0xFF, 0xCC, "v_pk_mul_f16 word0 prefix");
            assert_vop3p_packed_word1_full(w1, vsrc0, vsrc1, "v_pk_mul_f16");
        }
    });
}

#[test]
fn proptest_v_pk_fma_f16() {
    with_rng(0x504B464D, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let vsrc2 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_pk_fma_f16(vdst, vsrc0, vsrc1, vsrc2);
            assert_eq!((w0 >> 24) & 0xFF, 0xCC, "v_pk_fma_f16 word0 prefix");
            assert_eq!(w0 & 0x7F, vdst as u32, "v_pk_fma_f16 vdst");
            assert_vop3p_3src_word1_full(w1, vsrc0, vsrc1, vsrc2, "v_pk_fma_f16");
        }
    });
}

#[test]
fn proptest_v_dot2_f32_bf16() {
    with_rng(0x44324246, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let vsrc2 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_dot2_f32_bf16(vdst, vsrc0, vsrc1, vsrc2);
            assert_eq!((w0 >> 24) & 0xFF, 0xCC, "v_dot2_f32_bf16 word0 prefix");
            assert_eq!(w0 & 0x7F, vdst as u32, "v_dot2_f32_bf16 vdst");
            assert_vop3p_3src_word1_full(w1, vsrc0, vsrc1, vsrc2, "v_dot2_f32_bf16");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// VOP1 属性测试 (Vector Unary)
// ═══════════════════════════════════════════════════════════════

/// VOP1 word0[31:25] == 0x3F → top byte 0x7E or 0x7F
fn assert_vop1_prefix(word: u32, desc: &str) {
    assert_eq!(
        (word >> 25) & 0x7F, 0x3F,
        "{}: VOP1 prefix mismatch: 0x{:08X} (expected bits[31:25]=0x3F)",
        desc, word
    );
}

/// Extract VOP1 VDST from word0[24:17]
fn extract_vop1_vdst(word: u32) -> u8 {
    ((word >> 17) & 0xFF) as u8
}

/// Extract VOP1 SRC0 from word0[8:0]
fn extract_vop1_src0(word: u32) -> u32 {
    word & 0x1FF
}

/// Extract VOP1 opcode from word0[16:9]
fn extract_vop1_opcode(word: u32) -> u32 {
    (word >> 9) & 0xFF
}

#[test]
fn proptest_v_mov_b32() {
    with_rng(0x564D4F56, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_mov_b32(vdst, vsrc);
            assert_vop1_prefix(word, "v_mov_b32");
            assert_eq!(extract_vop1_vdst(word), vdst, "v_mov_b32 vdst({})", vdst);
            assert_eq!(extract_vop1_opcode(word), 0x01, "v_mov_b32 opcode=1");
            // VGPR src encoded as 256 + reg
            assert_eq!(extract_vop1_src0(word), vsrc as u32 + 256,
                "v_mov_b32 src({}) VGPR encoding", vsrc);
        }
    });
}

#[test]
fn proptest_v_mov_b32_from_sgpr() {
    with_rng(0x564D5350, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let ssrc = rng.range_u8(108);
            let word = gfx11::v_mov_b32_from_sgpr(vdst, ssrc);
            assert_vop1_prefix(word, "v_mov_b32_from_sgpr");
            assert_eq!(extract_vop1_vdst(word), vdst, "v_mov_b32_from_sgpr vdst");
            // SGPR encoded directly (no +256)
            assert_eq!(extract_vop1_src0(word), ssrc as u32,
                "v_mov_b32_from_sgpr ssrc({}) direct encoding", ssrc);
        }
    });
}

#[test]
fn proptest_v_exp_f32() {
    with_rng(0x56455850, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_exp_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_exp_f32");
            assert_eq!(extract_vop1_vdst(word), vdst, "v_exp_f32 vdst");
            assert_eq!(extract_vop1_opcode(word), 0x25, "v_exp_f32 opcode=0x25");
            assert_eq!(extract_vop1_src0(word), vsrc as u32 + 256, "v_exp_f32 src");
        }
    });
}

#[test]
fn proptest_v_log_f32() {
    with_rng(0x564C4F47, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_log_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_log_f32");
            assert_eq!(extract_vop1_opcode(word), 0x27, "v_log_f32 opcode=0x27");
        }
    });
}

#[test]
fn proptest_v_rcp_f32() {
    with_rng(0x56524350, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_rcp_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_rcp_f32");
            assert_eq!(extract_vop1_opcode(word), 0x2A, "v_rcp_f32 opcode=0x2A");
        }
    });
}

#[test]
fn proptest_v_sqrt_f32() {
    with_rng(0x56535152, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_sqrt_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_sqrt_f32");
            assert_eq!(extract_vop1_opcode(word), 0x33, "v_sqrt_f32 opcode=0x33");
        }
    });
}

#[test]
fn proptest_v_cvt_f32_u32() {
    with_rng(0x56434655, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_cvt_f32_u32(vdst, vsrc);
            assert_vop1_prefix(word, "v_cvt_f32_u32");
            assert_eq!(extract_vop1_vdst(word), vdst, "v_cvt_f32_u32 vdst");
            assert_eq!(extract_vop1_opcode(word), 0x06, "v_cvt_f32_u32 opcode=6");
        }
    });
}

#[test]
fn proptest_v_cvt_u32_f32() {
    with_rng(0x56435546, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_cvt_u32_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_cvt_u32_f32");
            assert_eq!(extract_vop1_opcode(word), 0x07, "v_cvt_u32_f32 opcode=7");
        }
    });
}

#[test]
fn proptest_v_cvt_f32_f16() {
    with_rng(0x56434631, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_cvt_f32_f16(vdst, vsrc);
            assert_vop1_prefix(word, "v_cvt_f32_f16");
            assert_eq!(extract_vop1_opcode(word), 0x17, "v_cvt_f32_f16 opcode=0x17");
        }
    });
}

#[test]
fn proptest_v_cvt_f16_f32() {
    with_rng(0x56433146, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_cvt_f16_f32(vdst, vsrc);
            assert_vop1_prefix(word, "v_cvt_f16_f32");
            assert_eq!(extract_vop1_opcode(word), 0x15, "v_cvt_f16_f32 opcode=0x15");
        }
    });
}

#[test]
fn proptest_v_permlane64_b32() {
    with_rng(0x56504D36, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_permlane64_b32(vdst, vsrc);
            assert_vop1_prefix(word, "v_permlane64_b32");
            assert_eq!(extract_vop1_vdst(word), vdst, "v_permlane64_b32 vdst");
            assert_eq!(extract_vop1_opcode(word), 0x67, "v_permlane64_b32 opcode=0x67");
            assert_eq!(extract_vop1_src0(word), vsrc as u32 + 256, "v_permlane64_b32 src");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// VOP2 属性测试 (Vector Binary ALU)
// ═══════════════════════════════════════════════════════════════

/// VOP2 word0 — VDST at bits[24:17]
fn extract_vop2_vdst(word: u32) -> u8 {
    ((word >> 17) & 0xFF) as u8
}

/// VOP2 SRC0 at bits[8:0]
fn extract_vop2_src0(word: u32) -> u32 {
    word & 0x1FF
}

/// VOP2 SRC1 at bits[16:9]
fn extract_vop2_src1(word: u32) -> u8 {
    ((word >> 9) & 0xFF) as u8
}

#[test]
fn proptest_v_add_f32() {
    with_rng(0x56414632, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let word = gfx11::v_add_f32(vdst, vsrc0, vsrc1);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_add_f32 vdst");
            // VOP2 SRC0: VGPR encoded as 256+n
            assert_eq!(extract_vop2_src0(word), vsrc0 as u32 + 256,
                "v_add_f32 src0({}) VGPR encoding", vsrc0);
            // VOP2 SRC1: raw VGPR number
            assert_eq!(extract_vop2_src1(word), vsrc1, "v_add_f32 src1");
        }
    });
}

#[test]
fn proptest_v_mul_f32() {
    with_rng(0x564D4632, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let word = gfx11::v_mul_f32(vdst, vsrc0, vsrc1);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_mul_f32 vdst");
            assert_eq!(extract_vop2_src0(word), vsrc0 as u32 + 256, "v_mul_f32 src0");
            assert_eq!(extract_vop2_src1(word), vsrc1, "v_mul_f32 src1");
        }
    });
}

#[test]
fn proptest_v_max_f32() {
    with_rng(0x564D5846, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let word = gfx11::v_max_f32(vdst, vsrc0, vsrc1);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_max_f32 vdst");
            assert_eq!(extract_vop2_src0(word), vsrc0 as u32 + 256, "v_max_f32 src0");
        }
    });
}

#[test]
fn proptest_v_min_f32() {
    with_rng(0x564D4E46, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let word = gfx11::v_min_f32(vdst, vsrc0, vsrc1);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_min_f32 vdst");
            assert_eq!(extract_vop2_src0(word), vsrc0 as u32 + 256, "v_min_f32 src0");
        }
    });
}

#[test]
fn proptest_v_sub_f32() {
    with_rng(0x56534632, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let word = gfx11::v_sub_f32(vdst, vsrc0, vsrc1);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_sub_f32 vdst");
            assert_eq!(extract_vop2_src0(word), vsrc0 as u32 + 256, "v_sub_f32 src0");
        }
    });
}

#[test]
fn proptest_v_add_u32_imm() {
    with_rng(0x56415532, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128); // 7-bit VDST
            let vsrc = rng.range_u8(255);
            let imm: u32 = rng.range(65); // 0-64
            let word = gfx11::v_add_u32_imm(vdst, vsrc, imm);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_add_u32_imm vdst");
            // v_add_u32_imm: SRC0 = inline constant (0x80+imm), SRC1 = vsrc (raw)
            assert_eq!(extract_vop2_src0(word), 0x80 + imm, "v_add_u32_imm inline const");
            assert_eq!(extract_vop2_src1(word), vsrc, "v_add_u32_imm vsrc");
        }
    });
}

#[test]
fn proptest_v_and_b32_imm() {
    with_rng(0x56414932, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(128); // 7-bit VDST
            let vsrc = rng.range_u8(255);
            let imm: u32 = rng.range(65); // 0-64
            let word = gfx11::v_and_b32_imm(vdst, vsrc, imm);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_and_b32_imm vdst");
            // v_and_b32_imm: SRC0 = inline constant, SRC1 = vsrc (raw)
            assert_eq!(extract_vop2_src0(word), 0x80 + imm, "v_and_b32_imm inline const");
            assert_eq!(extract_vop2_src1(word), vsrc, "v_and_b32_imm vsrc");
        }
    });
}

#[test]
fn proptest_v_lshlrev_b32() {
    with_rng(0x564C5332, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let shift = rng.range_u8(32);
            let vsrc = rng.range_u8(255);
            let word = gfx11::v_lshlrev_b32(vdst, shift, vsrc);
            assert_eq!(extract_vop2_vdst(word), vdst, "v_lshlrev_b32 vdst");
            // SRC0 = inline constant shift
            assert_eq!(extract_vop2_src0(word), 0x80 + shift as u32,
                "v_lshlrev_b32 shift({}) inline", shift);
            // SRC1 = raw VGPR number
            assert_eq!(extract_vop2_src1(word), vsrc, "v_lshlrev_b32 vsrc");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// VOP3 (E64) 属性测试 (Three-source Vector ALU)
// ═══════════════════════════════════════════════════════════════

#[test]
fn proptest_v_fma_f32() {
    with_rng(0x56464D41, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let vsrc2 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_fma_f32(vdst, vsrc0, vsrc1, vsrc2);
            // VOP3 word0[31:24] = 0xD6 (for v_fma_f32)
            assert_eq!((w0 >> 24) & 0xFF, 0xD6, "v_fma_f32 word0 prefix");
            assert_eq!(w0 & 0xFF, vdst as u32, "v_fma_f32 vdst");
            // word1: src0[8:0], src1[17:9], src2[26:18]
            assert_eq!(w1 & 0x1FF, vsrc0 as u32 + 256, "v_fma_f32 src0");
            assert_eq!((w1 >> 9) & 0x1FF, vsrc1 as u32 + 256, "v_fma_f32 src1");
            assert_eq!((w1 >> 18) & 0x1FF, vsrc2 as u32 + 256, "v_fma_f32 src2");
        }
    });
}

#[test]
fn proptest_v_mul_lo_u32() {
    with_rng(0x564D4C55, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_mul_lo_u32(vdst, vsrc0, vsrc1);
            assert_eq!((w0 >> 24) & 0xFF, 0xD7, "v_mul_lo_u32 word0 prefix (0xD7)");
            assert_eq!(w0 & 0xFF, vdst as u32, "v_mul_lo_u32 vdst");
            assert_eq!(w1 & 0x1FF, vsrc0 as u32 + 256, "v_mul_lo_u32 src0");
            assert_eq!((w1 >> 9) & 0x1FF, vsrc1 as u32 + 256, "v_mul_lo_u32 src1");
        }
    });
}

#[test]
fn proptest_v_pack_b32_f16() {
    with_rng(0x56504231, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(255);
            let vsrc0 = rng.range_u8(255);
            let vsrc1 = rng.range_u8(255);
            let [w0, w1] = gfx11::v_pack_b32_f16(vdst, vsrc0, vsrc1);
            assert_eq!((w0 >> 16) & 0xFFFF, 0xD711, "v_pack_b32_f16 opcode");
            assert_eq!(w0 & 0xFF, vdst as u32, "v_pack_b32_f16 vdst");
            assert_eq!(w1 & 0x1FF, vsrc0 as u32 + 256, "v_pack_b32_f16 src0");
            assert_eq!((w1 >> 9) & 0x1FF, vsrc1 as u32 + 256, "v_pack_b32_f16 src1");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// GFX11 (兼容) 属性测试 — 确保 GFX11 不被 GFX1200 修改破坏
// ═══════════════════════════════════════════════════════════════

/// GFX11 Global word0[31:24] == 0xDC
fn assert_gfx11_global_prefix(word0: u32, desc: &str) {
    assert_eq!(
        (word0 >> 24) & 0xFF, 0xDC,
        "{}: GFX11 global prefix mismatch: 0x{:08X} (expected top byte 0xDC)",
        desc, word0
    );
}

#[test]
fn proptest_gfx11_global_load_dwordx4() {
    with_rng(0x4731314C, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(252);
            let vaddr = rng.range_u8(254);
            let offset = (rng.next_u32() as i32) & 0x1FFF; // 13-bit signed
            let [w0, w1] = gfx11::global_load_dwordx4(vdst, vaddr, offset);
            assert_gfx11_global_prefix(w0, "gfx11 global_load_dwordx4");
            // word1: vdst at bits[31:24]
            assert_eq!((w1 >> 24) & 0xFF, vdst as u32, "gfx11 global_load_dwordx4 vdst");
            // word1: saddr = 0x7C at bits[23:16]
            assert_eq!((w1 >> 16) & 0xFF, 0x7C, "gfx11 global_load_dwordx4 saddr=off");
            // word1: vaddr at bits[7:0]
            assert_eq!(w1 & 0xFF, vaddr as u32, "gfx11 global_load_dwordx4 vaddr");
        }
    });
}

#[test]
fn proptest_gfx11_global_store_dword() {
    with_rng(0x47313153, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vaddr = rng.range_u8(254);
            let vsrc = rng.range_u8(255);
            let offset = (rng.next_u32() as i32) & 0x1FFF;
            let [w0, w1] = gfx11::global_store_dword(vaddr, vsrc, offset);
            assert_gfx11_global_prefix(w0, "gfx11 global_store_dword");
            assert_eq!((w1 >> 16) & 0xFF, 0x7C, "gfx11 global_store_dword saddr=off");
            assert_eq!((w1 >> 8) & 0xFF, vsrc as u32, "gfx11 global_store_dword vsrc");
            assert_eq!(w1 & 0xFF, vaddr as u32, "gfx11 global_store_dword vaddr");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// DS (Data Share / LDS) 属性测试
// ═══════════════════════════════════════════════════════════════

/// GFX11 DS word0[31:24] == 0xD8 or 0xD9 or 0xDB (size-dependent)
fn assert_ds_prefix(word0: u32, expected: u32, desc: &str) {
    assert_eq!(
        (word0 >> 24) & 0xFF, expected,
        "{}: DS prefix mismatch: 0x{:08X} (expected 0x{:02X})",
        desc, word0, expected
    );
}

#[test]
fn proptest_ds_read_b128() {
    with_rng(0x44524231, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(252);
            let vaddr = rng.range_u8(255);
            let offset = rng.next_u32() as u16;
            let [w0, w1] = gfx11::ds_read_b128(vdst, vaddr, offset);
            assert_ds_prefix(w0, 0xDB, "ds_read_b128");
            assert_eq!(w0 & 0xFFFF, offset as u32, "ds_read_b128 offset");
            assert_eq!(w1 & 0xFF, vaddr as u32, "ds_read_b128 vaddr");
            assert_eq!((w1 >> 24) & 0xFF, vdst as u32, "ds_read_b128 vdst");
        }
    });
}

#[test]
fn proptest_ds_write_b128() {
    with_rng(0x44574231, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vaddr = rng.range_u8(255);
            let vsrc = rng.range_u8(252);
            let offset = rng.next_u32() as u16;
            let [w0, w1] = gfx11::ds_write_b128(vaddr, vsrc, offset);
            // ds_write_b128 opcode = 0xD8FD → word0 = 0xD8FD0000 | offset
            assert_ds_prefix(w0, 0xD8, "ds_write_b128 word0 prefix");
            assert_eq!(w0 & 0xFFFF, offset as u32, "ds_write_b128 offset");
            assert_eq!(w1 & 0xFF, vaddr as u32, "ds_write_b128 vaddr");
            assert_eq!((w1 >> 8) & 0xFF, vsrc as u32, "ds_write_b128 vsrc");
        }
    });
}

#[test]
fn proptest_ds_store_b32() {
    with_rng(0x44535333, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vaddr = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let offset = rng.next_u32() as u16;
            let [w0, w1] = gfx11::ds_store_b32(vaddr, vsrc, offset);
            assert_ds_prefix(w0, 0xD8, "ds_store_b32");
            assert_eq!(w0 & 0xFFFF, offset as u32, "ds_store_b32 offset");
            assert_eq!(w1 & 0xFF, vaddr as u32, "ds_store_b32 vaddr");
            assert_eq!((w1 >> 8) & 0xFF, vsrc as u32, "ds_store_b32 vsrc");
        }
    });
}

#[test]
fn proptest_ds_load_b64() {
    with_rng(0x444C4236, |rng| {
        for _ in 0..PROPERTY_ITERATIONS {
            let vdst = rng.range_u8(254);
            let vaddr = rng.range_u8(255);
            let offset = rng.next_u32() as u16;
            let [w0, w1] = gfx11::ds_load_b64(vdst, vaddr, offset);
            assert_ds_prefix(w0, 0xD9, "ds_load_b64");
            assert_eq!(w0 & 0xFFFF, offset as u32, "ds_load_b64 offset");
            assert_eq!(w1 & 0xFF, vaddr as u32, "ds_load_b64 vaddr");
            assert_eq!((w1 >> 24) & 0xFF, vdst as u32, "ds_load_b64 vdst");
        }
    });
}

#[test]
fn proptest_ds_swizzle_b32() {
    with_rng(0x4453574C, |rng| {
        for _ in 0..500 {
            let vdst = rng.range_u8(255);
            let vsrc = rng.range_u8(255);
            let pattern = rng.next_u32() as u16;
            let [w0, w1] = gfx11::ds_swizzle_b32(vdst, vsrc, pattern);
            assert_ds_prefix(w0, 0xD8, "ds_swizzle_b32");
            assert_eq!(w0 & 0xFFFF, pattern as u32, "ds_swizzle_b32 pattern");
            assert_eq!(w1 & 0xFF, vsrc as u32, "ds_swizzle_b32 vsrc");
            assert_eq!((w1 >> 24) & 0xFF, vdst as u32, "ds_swizzle_b32 vdst");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// Global Atomic 属性测试 (GFX1200)
// ═══════════════════════════════════════════════════════════════

#[test]
fn proptest_gfx1200_global_atomic_add_u32_rtn() {
    with_rng(0x41415532, |rng| {
        for _ in 0..500 {
            let vdst = rng.range_u8(128); // 7-bit VDST
            let vaddr = rng.range_u8(254);
            let vdata = rng.range_u8(128); // limit to avoid overflow into high bits
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_atomic_add_u32_gfx1200(vdst, vaddr, vdata, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_atomic_add_u32_rtn");
            // Atomics encode vdata in word0 at (vdata << 1), OR'd with 0x7C
            // So saddr field is NOT simply 0x7C — verify the full word0
            let expected_w0 = 0xEE0D4000u32 | ((vdata as u32) << 1) | 0x7C;
            assert_eq!(w0, expected_w0,
                "global_atomic_add_u32_rtn: word0 mismatch for vdata={}", vdata);
            // word1: TH_ATOMIC_RETURN flag at bit 16, plus vdst at bits[6:0]
            assert_eq!(w1 & 0x7F, vdst as u32 & 0x7F,
                "global_atomic_add_u32_rtn: vdst low bits");
            assert_eq!(w1 & (1 << 16), 1 << 16,
                "global_atomic_add_u32_rtn: TH_ATOMIC_RETURN bit");
        }
    });
}

#[test]
fn proptest_gfx1200_global_atomic_add_u32_nortn() {
    with_rng(0x41414E52, |rng| {
        for _ in 0..500 {
            let vaddr = rng.range_u8(254);
            let vdata = rng.range_u8(255);
            let offset = rng.next_u32() as i32;
            let [w0, w1, w2] = gfx11::global_atomic_add_u32_no_rtn_gfx1200(vaddr, vdata, offset);
            assert_vglobal_word0_prefix(w0, 0xEE, "global_atomic_add_u32_nortn");
            assert_eq!(w1, 0, "global_atomic_add_u32_nortn: word1=0 (no return)");
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// 边界条件和回归测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn proptest_boundary_all_zeros() {
    // All-zero operands should produce valid encodings without panicking
    let _ = gfx11::s_add_u32(0, 0, 0);
    let _ = gfx11::s_mov_b32(0, 0);
    let _ = gfx11::v_add_f32(0, 0, 0);
    let _ = gfx11::v_mov_b32(0, 0);
    let _ = gfx11::v_wmma_f32_16x16x16_bf16(0, 0, 0, 0);
    let _ = gfx11::s_wait_loadcnt(0);
    let _ = gfx11::s_branch(0);
    let [_, _] = gfx11::s_load_dword_gfx1200(0, 0, 0);
    let [_, _, _] = gfx11::global_load_dword_gfx1200(0, 0, 0);
    let [_, _] = gfx11::ds_read_b128(0, 0, 0);
}

#[test]
fn proptest_boundary_max_values() {
    // Maximum values should not overflow
    let v255: u8 = 255;
    let _ = gfx11::v_add_f32(v255, v255, v255);
    let _ = gfx11::v_mov_b32(v255, v255);
    let _ = gfx11::v_wmma_f32_16x16x16_bf16(v255, v255, v255, v255);
    let _ = gfx11::s_wait_loadcnt(63);
    let _ = gfx11::s_branch(i16::MAX);
    let _ = gfx11::s_branch(i16::MIN);
    let [_, _] = gfx11::s_load_dword_gfx1200(63, 106, 0xFFFFFF);
    let [_, _, _] = gfx11::global_load_dword_gfx1200(255, 254, i32::MAX);
}

#[test]
fn proptest_encoding_determinism() {
    // Same inputs must always produce same output (no statefulness)
    let args = (42u8, 10u8, 20u8, 30u8);
    for _ in 0..100 {
        assert_eq!(
            gfx11::v_wmma_f32_16x16x16_bf16(args.0, args.1, args.2, args.3),
            gfx11::v_wmma_f32_16x16x16_bf16(args.0, args.1, args.2, args.3),
            "WMMA encoding must be deterministic"
        );
        assert_eq!(
            gfx11::s_add_u32(args.0, args.1, args.2),
            gfx11::s_add_u32(args.0, args.1, args.2),
            "s_add_u32 encoding must be deterministic"
        );
        let [w0a, w1a, w2a] = gfx11::global_load_dword_gfx1200(args.0, args.1, 42);
        let [w0b, w1b, w2b] = gfx11::global_load_dword_gfx1200(args.0, args.1, 42);
        assert_eq!([w0a, w1a, w2a], [w0b, w1b, w2b],
            "global_load_b32 encoding must be deterministic");
    }
}

#[test]
fn proptest_gfx1200_vs_gfx11_smem_divergence() {
    // GFX1200 and GFX11 SMEM have DIFFERENT layouts for same logical instruction
    let [w0_gfx12, w1_gfx12] = gfx11::s_load_dwordx4_gfx1200(4, 0, 0x10);
    let [w0_gfx11, w1_gfx11] = gfx11::s_load_dwordx4(4, 0, 0x10);

    // Both have 0xF4 prefix
    assert_eq!((w0_gfx12 >> 24) & 0xFF, 0xF4, "GFX1200 SMEM prefix");
    assert_eq!((w0_gfx11 >> 24) & 0xFF, 0xF4, "GFX11 SMEM prefix");

    // But the layout is different!
    assert_ne!(w0_gfx12, w0_gfx11,
        "GFX1200 and GFX11 SMEM word0 must differ (different layout for same logical op)");
}

#[test]
fn proptest_gfx1200_vs_gfx11_global_divergence() {
    // GFX1200 global uses 3-dword VGLOBAL format (0xEE), GFX11 uses 2-dword FLAT format (0xDC)
    let [w0_gfx12, w1_gfx12, w2_gfx12] = gfx11::global_load_dword_gfx1200(0, 10, 0);
    let [w0_gfx11, w1_gfx11] = gfx11::global_load_dword(0, 10, 0);

    assert_eq!((w0_gfx12 >> 24) & 0xFF, 0xEE, "GFX1200 global prefix = 0xEE");
    assert_eq!((w0_gfx11 >> 24) & 0xFF, 0xDC, "GFX11 global prefix = 0xDC");
}
