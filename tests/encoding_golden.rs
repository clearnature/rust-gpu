//! 编码器金标准锁（llvm-mc 反汇编对拍，三件套方法论）
//!
//! 出处：2026-11-05 全量覆盖审计——285 个真编码器中 144 个无 llvm-mc 金
//! 标准锁，本文件锁定命脉级子集。每条 = 编码函数 + 代表性参数 → 字序列
//! （llvm-mc 23.1.4 /opt/rocm 字节流反汇编实证语义）→ 测试断言逐字匹配。
//! 编码器改动任何 bit 即红灯；金标准注释即该编码的 LLVM 权威语义。
//! 本表已实锤并修复 4 个 RDNA3→RDNA4 遗留编码 bug：
//!   ds_bpermute_b32（旧 opcode 52）、v_max_f32_imm0（旧 VOP3 形式）、
//!   ds_add_f32（op 22 + GDS=1——会打穿全局内存！）、ds_max_f32（同病，
//!   GFX12 改名 ds_max_num_f32）——均为死路径编码器，金标准锁是唯一防线。
//! 注：global_load_async_to_lds_* 为 GFX1250 专属，金标准以 gfx1250 反汇编。
//! 教训：llvm-mc --disassemble 输入=字节流小端；系统 llvm-mc 不支持
//! gfx1200，用 /opt/rocm/llvm/bin 版。
//!
//! 生成方式：tests/encoding_golden_gen.rs 产出字 → llvm-mc 反汇编 → bake。
//! 用法：cargo test --release --features rocm --test encoding_golden

use t0_gpu::rdna3_asm::gfx11;

/// 金标准表：(描述, 期望字序列, llvm-mc 反汇编语义)
const GOLDENS: &[(&str, &[u32], &str)] = &[
    ("ds_load_b32(1,2,0)", &[0xD8D80000, 0x01000002], "ds_load_b32 v1, v2"),
    ("ds_load_b32(4,8,64)", &[0xD8D80040, 0x04000008], "ds_load_b32 v4, v8 offset:64"),
    ("ds_read_b128(4,2,0)", &[0xDBFC0000, 0x04000002], "ds_load_b128 v[4:7], v2"),
    ("ds_store_b32(2,3,0)", &[0xD8340000, 0x00000302], "ds_store_b32 v2, v3"),
    ("ds_store_b64(2,4,8)", &[0xD9340008, 0x00000402], "ds_store_b64 v2, v[4:5] offset:8"),
    ("ds_store_b128(2,4,0)", &[0xDB7C0000, 0x00000402], "ds_store_b128 v2, v[4:7]"),
    ("ds_load_u16(1,2,0)", &[0xD8F00000, 0x01000002], "ds_load_u16 v1, v2"),
    ("ds_bpermute_b32(1,2,3)", &[0xDACC0000, 0x01000302], "ds_bpermute_b32 v1, v2, v3"),
    ("ds_swizzle_b32(4,3,0x401F)", &[0xD8D4401F, 0x04000003], "ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,16)"),
    ("ds_add_f32(2,3,0)", &[0xD8540000, 0x00000302], "ds_add_f32 v2, v3"),
    ("ds_max_f32(2,3,0)", &[0xD84C0000, 0x00000302], "ds_max_num_f32 v2, v3"),
    ("v_cndmask_b32(4,1,2)", &[0x02080501], "v_cndmask_b32_e32 v4, v1, v2, vcc_lo"),
    ("v_cmp_lt_u32(1,2)", &[0x7C920501], "v_cmp_lt_u32_e32 vcc_lo, v1, v2"),
    ("v_add_co_u32(3,10,1,2)", &[0xD7000A03, 0x00020501], "v_add_co_u32 v3, s10, v1, v2"),
    ("v_add_co_ci_u32(3,10,1,2,11)", &[0xD5200A03, 0x002E0501], "v_add_co_ci_u32_e64 v3, s10, v1, v2, s11"),
    ("v_max_f32_imm0(4,3)", &[0xD5160004, 0x02010103], "v_max_num_f32_e64 v4, v3, 0"),
    ("v_dual_add_f32_mul_f32(0,1,2,3,4,5)", &[0xC9060501, 0x00020B04], "v_dual_add_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5"),
    ("global_load_async_to_lds_b128(4,2,0)", &[0xEE18807C, 0x00000004, 0x00000002], "global_load_async_to_lds_b128 v4, v[2:3], off"),
    ("scratch_load_b128_gfx1200(4,2,0)", &[0xED05C07C, 0x00000004, 0x00000002], "scratch_load_b128 v[4:7], off, off"),
];

#[test]
fn encoders_match_llvm_mc_goldens() {
    assert!(!GOLDENS.is_empty(), "golden table must not be empty");
    let cases: Vec<(&str, Vec<u32>)> = vec![
        ("ds_load_b32(1,2,0)", gfx11::ds_load_b32(1, 2, 0).to_vec()),
        ("ds_load_b32(4,8,64)", gfx11::ds_load_b32(4, 8, 64).to_vec()),
        ("ds_read_b128(4,2,0)", gfx11::ds_read_b128(4, 2, 0).to_vec()),
        ("ds_store_b32(2,3,0)", gfx11::ds_store_b32(2, 3, 0).to_vec()),
        ("ds_store_b64(2,4,8)", gfx11::ds_store_b64(2, 4, 8).to_vec()),
        ("ds_store_b128(2,4,0)", gfx11::ds_store_b128(2, 4, 0).to_vec()),
        ("ds_load_u16(1,2,0)", gfx11::ds_load_u16(1, 2, 0).to_vec()),
        ("ds_bpermute_b32(1,2,3)", gfx11::ds_bpermute_b32(1, 2, 3).to_vec()),
        ("ds_swizzle_b32(4,3,0x401F)", gfx11::ds_swizzle_b32(4, 3, 0x401F).to_vec()),
        ("ds_add_f32(2,3,0)", gfx11::ds_add_f32(2, 3, 0).to_vec()),
        ("ds_max_f32(2,3,0)", gfx11::ds_max_f32(2, 3, 0).to_vec()),
        ("v_cndmask_b32(4,1,2)", vec![gfx11::v_cndmask_b32(4, 1, 2)]),
        ("v_cmp_lt_u32(1,2)", vec![gfx11::v_cmp_lt_u32(1, 2)]),
        ("v_add_co_u32(3,10,1,2)", gfx11::v_add_co_u32(3, 10, 1, 2).to_vec()),
        ("v_add_co_ci_u32(3,10,1,2,11)", gfx11::v_add_co_ci_u32(3, 10, 1, 2, 11).to_vec()),
        ("v_max_f32_imm0(4,3)", gfx11::v_max_f32_imm0(4, 3).to_vec()),
        ("v_dual_add_f32_mul_f32(0,1,2,3,4,5)", gfx11::v_dual_add_f32_mul_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("global_load_async_to_lds_b128(4,2,0)", gfx11::global_load_async_to_lds_b128(4, 2, 0).to_vec()),
        ("scratch_load_b128_gfx1200(4,2,0)", gfx11::scratch_load_b128_gfx1200(4, 2, 0).to_vec()),
    ];
    for (desc, actual) in &cases {
        let golden = GOLDENS.iter().find(|(g, _, _)| g == desc)
            .unwrap_or_else(|| panic!("golden 缺失: {desc}"));
        assert_eq!(&golden.1.to_vec(), actual,
            "{desc}: 编码偏离 llvm-mc 金标准（LLVM 语义: {}）", golden.2);
    }
    for (g, _, _) in GOLDENS {
        assert!(cases.iter().any(|(d, _)| d == g), "金标准无对拍用例: {g}");
    }
}
