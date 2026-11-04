//! TileGemm/gemm_gen 契约断言（TDD 回归锁，纯 CPU 无 GPU）
//!
//! 出处标注：
//!   [GPU]  本机 RX 9060 XT fullcheck 实测（twop_fullcheck/swap correctness）
//!   [事故] 2026-11-07/08：grid 虚标（41.1TF 伪影）、wg=256 非法组合（WT 从未加载）
//!   [数学] 维度恒等式（tile_ir.rs 文档注释）
//!
//! 用法：cargo test --release --features rocm --test tile_gemm_contract

use t0_gpu::t0::gemm_gen::{self, GemmConfig};
use t0_gpu::t0::tile_ir::{self, TileGemm};

/// 全部 preset 构造器（新加 preset 必须登记到本列表——恒等式与 LDS 契约全覆盖）
fn all_presets() -> Vec<TileGemm> {
    vec![
        TileGemm::tile_32x64_k16(),
        TileGemm::tile_64x64_k16(),
        TileGemm::tile_64x64_k32(),
        TileGemm::tile_64x64_k64(),
        TileGemm::tile_32x128_k16(),
        TileGemm::tile_128x64_k16(),
        TileGemm::tile_128x64_k32(),
        TileGemm::tile_128x64_k64(),
        TileGemm::tile_64x128_k16(),
        TileGemm::tile_64x128_k32(),
        TileGemm::tile_128x128_k16(),
        TileGemm::tile_128x128_k32(),
        TileGemm::tile_128x128_k48(),
        TileGemm::tile_128x128_k64(),
        TileGemm::tile_128x128_k16_swap(),
        TileGemm::tile_128x128_k32_swap(),
        TileGemm::tile_256x64_k32_wgp(),
        TileGemm::tile_256x64_k64_wgp(),
        TileGemm::tile_32x32_k16(),
        TileGemm::tile_128x32_k16(),
        TileGemm::tile_64x256_k32(),
        TileGemm::tile_128x256_k32(),
    ]
}

/// [数学] 维度恒等式：tile_m = n_waves × rows_per_wave = n_waves × n_row_blocks × 16；
/// wg_size = n_waves × 32；k_sub_steps = tile_k / 16
#[test]
fn preset_dimension_identities() {
    for spec in all_presets() {
        let n_waves = spec.waves_per_wg.unwrap_or(spec.tile_m / 32);
        let rpw = spec.tile_m / n_waves;
        let nrb = rpw / 16;
        assert_eq!(spec.wg_size(), n_waves * 32,
            "{} wg_size 恒等式破坏", spec.name());
        assert_eq!(spec.n_row_blocks(), nrb,
            "{} n_row_blocks 恒等式破坏", spec.name());
        assert_eq!(spec.rows_per_wave(), rpw, "{} rows_per_wave 恒等式", spec.name());
        assert_eq!(spec.k_sub_steps(), spec.tile_k / 16, "{} k_sub", spec.name());
        // WMMA 16 行粒度
        assert_eq!(rpw % 16, 0, "{} rows_per_wave 必须是 16 的倍数（WMMA 行粒度）", spec.name());
    }
}

/// [数学] LDS 总账自洽：lds_total = 双缓冲 × (X 区 + WT 区) + acc_swap 区
///（group/persistent 附加项另有公式；本断言锁非 persistent 主项）
#[test]
fn lds_total_self_consistent() {
    for spec in all_presets() {
        let gemm_lds = if spec.double_buffer { spec.lds_per_buffer() * 2 } else { spec.lds_per_buffer() };
        let expect = gemm_lds + if spec.acc_swap { spec.acc_swap_region_size() } else { 0 };
        assert_eq!(spec.lds_total(), expect, "{} LDS 总账不自洽", spec.name());
    }
}

/// [GPU] 网格覆盖枚举：对边界/对齐/大尺寸网格点，派发网格必须覆盖全部输出
/// 且 grid_x 是 wg_size 整数倍（HSA grid 单位 = work-item）
#[test]
fn grid_coverage_enumeration() {
    let configs = [GemmConfig::tile_128x64_k16(), GemmConfig::tile_128x64_k32(), GemmConfig::tile_64x64_k32()];
    let dims = [1u32, 31, 32, 33, 63, 64, 127, 128, 255, 256, 4095, 4096];
    for cfg in &configs {
        for &m in &dims {
            for &n in &dims {
                let (gx, gy) = gemm_gen::compute_grid_auto(cfg, m, n);
                assert_eq!(gx % cfg.wg_size, 0,
                    "{} grid_x={gx} 非 wg={} 整数倍 (m={m},n={n})", cfg.name(), cfg.wg_size);
                let tiles_m = (m + cfg.tile_m - 1) / cfg.tile_m;
                let tiles_n = (n + cfg.tile_n - 1) / cfg.tile_n;
                assert_eq!(gy, tiles_m, "grid_y 必须等于 M 方向 tile 数");
                let wgs_n = gx / cfg.wg_size;
                assert_eq!(wgs_n, tiles_n, "N 方向 tile 数（覆盖 {n} 需 {tiles_n}）");
                assert!(wgs_n * cfg.tile_n >= n, "网格未覆盖 N={n}");
                assert!(tiles_m * cfg.tile_m >= m, "网格未覆盖 M={m}");
            }
        }
    }
}

/// [事故回归 2026-11-08] kernarg_m 44B 逐字段布局：
/// X@0 WT@8 Y@16 K@24 N@28 split_k_shift@32 y_split_stride@36 M@40
///（split_k_shift 由 split_k 映射 1→0, 2→1, 4→2, 8→3, 16→4）
#[test]
fn build_kernargs_m_field_layout() {
    let mut spec = TileGemm::tile_64x64_k16();
    spec.split_k = 4; // → shift=2
    let (x, wt, y, k, n, m) = (0xA000u64, 0xB000u64, 0xC000u64, 64u32, 128u32, 32u32);
    let ka = tile_ir::build_kernargs_m(x, wt, y, k, n, m, &spec);
    assert_eq!(ka.len(), 44);
    let rd_u64 = |o: usize| u64::from_le_bytes(ka[o..o + 8].try_into().unwrap());
    let rd_u32 = |o: usize| u32::from_le_bytes(ka[o..o + 4].try_into().unwrap());
    assert_eq!(rd_u64(0), x, "arg0 X@0");
    assert_eq!(rd_u64(8), wt, "arg1 WT@8");
    assert_eq!(rd_u64(16), y, "arg2 Y@16");
    assert_eq!(rd_u32(24), k, "arg3 K@24");
    assert_eq!(rd_u32(28), n, "arg4 N@28");
    assert_eq!(rd_u32(32), 2, "arg5 split_k_shift@32（split_k=4 → 2）");
    assert_eq!(rd_u32(36), 0, "arg6 y_split_stride@36（caller 负责，恒 0）");
    assert_eq!(rd_u32(40), m, "arg7 M@40（OOB 边界检查必需）");
}

/// [事故回归 2026-11-08] gemm_gen 40B 布局：split_k>1 时 y_split_stride=m*n*4
///（每分区写独立平面）；split_k=1 时 0
#[test]
fn gemm_build_kernargs_split_stride_semantics() {
    let mut no_sk = GemmConfig::tile_128x64_k16(); no_sk.split_k = None;
    let ka1 = gemm_gen::build_kernargs(1, 2, 3, 64, 64, 64, &no_sk);
    assert_eq!(ka1.len(), 40);
    assert_eq!(u32::from_le_bytes(ka1[36..40].try_into().unwrap()), 0,
        "split_k=1 时 y_split_stride 必须 0");

    let mut sk2 = GemmConfig::tile_128x64_k16(); sk2.split_k = Some(2);
    let ka2 = gemm_gen::build_kernargs(1, 2, 3, 64, 64, 256, &sk2);
    assert_eq!(u32::from_le_bytes(ka2[36..40].try_into().unwrap()), 256 * 64 * 4,
        "split_k=2 时 y_split_stride = M*N*4");
}

/// [语义锁] f32→bf16 截断语义（tile_ir::f32_to_bf16）：
/// 高 16 位直取（truncation，非 RNE）——调用方契约：GMEM 载入数据用本函数时
/// 与 GPU 侧 bf16 解读一致即可；ignis bf16_matmul 用的是 RNE 变体（独立实现）
#[test]
fn f32_to_bf16_truncation_semantics() {
    let f = tile_ir::f32_to_bf16;
    assert_eq!(f(1.0), 0x3F80);
    assert_eq!(f(-1.0), 0xBF80);
    assert_eq!(f(0.0), 0);
    assert_eq!(f(f32::INFINITY), 0x7F80);
    assert_eq!(f(f32::NEG_INFINITY), 0xFF80);
    // 截断：1.0 + 2^-9（fp32 可表示，bf16 尾数只有 7 位 → 戕断回 1.0）
    let slightly_over = f32::from_bits(1.0f32.to_bits() + 0x00FF);
    assert_eq!(f(slightly_over), 0x3F80, "截断语义：低位尾数直接丢弃");
    // NaN 位型保持（高 16 位携带 quiet bit）
    let nan = f32::NAN;
    assert_eq!(f(nan), (nan.to_bits() >> 16) as u16);
    // 往返：bf16→f32→bf16 幂等
    for bits in [0x3F80u16, 0xBF80, 0x007F, 0x8000] {
        assert_eq!(f(tile_ir::bf16_to_f32(bits)), bits, "bf16 往返幂等 {bits:#06x}");
    }
}

/// [事故回归] probe_lane_mask：单 lane EXEC 掩码（lane & 31 回绕——32 位 s_and_b64 限制）
#[test]
fn probe_lane_mask_semantics() {
    assert_eq!(tile_ir::probe_lane_mask(0), 1);
    assert_eq!(tile_ir::probe_lane_mask(31), 1 << 31);
    assert_eq!(tile_ir::probe_lane_mask(33), 1 << 1, "lane 33 回绕到 lane 1（文档注释语义）");
}

/// [事故回归 2026-11-08] auto_select 全域抽样：产出必须可行
///（每线程 GMEM load ≥1 + VGPR ≤256——wg=256×128x64 曾生成 WT 从未加载的内核）
#[test]
fn auto_select_domain_sample_feasible() {
    let ms = [16u32, 31, 64, 128, 256, 1024, 4096];
    let ns = [32u32, 64, 128, 256, 4096];
    let ks = [16u32, 32, 64, 256, 4096];
    let mut n_checked = 0usize;
    for &m in &ms { for &n in &ns { for &k in &ks {
        let cfg = gemm_gen::auto_select(m, k, n);
        assert!(cfg.is_feasible(),
            "auto_select({m},{k},{n}) → {} 不可行", cfg.name());
        n_checked += 1;
    }}}
    eprintln!("[auto_select_domain] checked {n_checked} combos");
    assert!(n_checked >= 100, "抽样覆盖不足：{n_checked}");
}
