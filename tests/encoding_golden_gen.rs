//! 金标准生成器（临时）——产出字序列供 llvm-mc 反汇编对拍
//! 运行：cargo test --release --features rocm --test encoding_golden_gen -- --nocapture
//! 生成后由脚本 bake 进 tests/encoding_golden.rs，本文件删除。

use t0_gpu::rdna3_asm::gfx11;

#[test]
fn gen_golden_words() {
    let mut out: Vec<String> = Vec::new();
    let mut push = |name: &str, words: &[u32]| {
        let hex: Vec<String> = words.iter().map(|w| format!("0x{:08X}", w)).collect();
        out.push(format!("{}|{}", name, hex.join(",")));
    };

    // ── DS 族（GEMM LDS 数据通路核心）──
    push("ds_load_b32(1,2,0)", &gfx11::ds_load_b32(1, 2, 0));
    push("ds_load_b32(4,8,64)", &gfx11::ds_load_b32(4, 8, 64));
    push("ds_read_b128(4,2,0)", &gfx11::ds_read_b128(4, 2, 0));
    push("ds_store_b32(2,3,0)", &gfx11::ds_store_b32(2, 3, 0));
    push("ds_store_b64(2,4,8)", &gfx11::ds_store_b64(2, 4, 8));
    push("ds_store_b128(2,4,0)", &gfx11::ds_store_b128(2, 4, 0));
    push("ds_load_u16(1,2,0)", &gfx11::ds_load_u16(1, 2, 0));
    push("ds_bpermute_b32(1,2,3)", &gfx11::ds_bpermute_b32(1, 2, 3));
    push("ds_swizzle_b32(4,3,0x401F)", &gfx11::ds_swizzle_b32(4, 3, 0x401F));

    // ── 转换/掩码/进位族 ──
    push("v_cndmask_b32(4,1,2)", &[gfx11::v_cndmask_b32(4, 1, 2)]);
    push("v_cmp_lt_u32(1,2)", &[gfx11::v_cmp_lt_u32(1, 2)]);
    push("v_add_co_u32(3,10,1,2)", &gfx11::v_add_co_u32(3, 10, 1, 2));
    push("v_add_co_ci_u32(3,10,1,2,11)", &gfx11::v_add_co_ci_u32(3, 10, 1, 2, 11));
    push("v_max_f32_imm0(4,3)", &gfx11::v_max_f32_imm0(4, 3));

    // ── 双发射族（性能路径）──
    push("v_dual_add_f32_mul_f32(0,1,2,3,4,5)",
         &gfx11::v_dual_add_f32_mul_f32(0, 1, 2, 3, 4, 5));

    // ── 协同加载/spill 通路 ──
    push("global_load_async_to_lds_b128(4,2,0)", &gfx11::global_load_async_to_lds_b128(4, 2, 0));
    push("scratch_load_b128_gfx1200(4,2,0)", &gfx11::scratch_load_b128_gfx1200(4, 2, 0));

    let text = out.join("\n");
    std::fs::write("/tmp/enc_words.txt", text).unwrap();
    eprintln!("[gen] {} 条字序列 → /tmp/enc_words.txt", out.len());
}
