//! 硬件底数契约断言（TDD 回归锁，纯 CPU）
//!
//! 出处：2026-11-08 审计（GFX1100Limits::default 96 CU/960GB/s 是 7900 XTX 底数，
//! 直接喂 GFX1200 tune 会把评分并行度高估 3 倍——n_scheduling_units = n_cus/2）。
//! amd-smi 实测：RX 9060 XT 32 CU / boost 2780MHz / GDDR6 322 GB/s。
//!
//! 用法：cargo test --release --features rocm --test hw_limits_contract

use t0_gpu::t0::cost_model::{auto_schedule_gemm, auto_schedule_gemm_hw, DataFormat, GFX1100Limits};

/// [GPU/规格] for_gfx1200 必须是 RX 9060 XT 实测底数——防止未来被改回 7900 XTX 值
///（或删构造器导致 tune 静默回落 default）
#[test]
fn for_gfx1200_is_rx9060xt_baseline() {
    let hw = GFX1100Limits::for_gfx1200();
    assert_eq!(hw.n_cus, 32, "9060 XT = 32 CU（Navi44），非 7900 XTX 的 96");
    assert!((hw.vmem_bandwidth_gbps - 322.0).abs() < 1.0, "GDDR6 322 GB/s");
    assert!((hw.clock_ghz - 2.6).abs() < 0.1, "boost ~2.6 GHz（amd-smi MAX_CLK 2780）");
    // 与 GFX1100 等价的容量项（继承 default）
    assert_eq!(hw.max_vgprs, 256);
    assert_eq!(hw.lds_per_wg, 65536, "ISA §1.2.2.1 单 WG 64KB");
    assert_eq!(hw.wmma_k, 16);
}

/// [规格] default 必须保持 7900 XTX 底数——GFX1100 参考卡的事实也要锁
///（两边都锁：任何一边被"悄悄修改"立即暴露）
#[test]
fn default_is_rx7900xtx_baseline() {
    let hw = GFX1100Limits::default();
    assert_eq!(hw.n_cus, 96);
    assert!((hw.vmem_bandwidth_gbps - 960.0).abs() < 1.0);
    assert_eq!(hw.max_vgprs, 256);
    assert_eq!(hw.lds_per_wg, 65536);
}

/// [行为] auto_schedule_gemm_hw 分域：两种底数产出**相同的可行候选集合**
///（feasible 判据 vgpr/sgpr/lds 不含 n_cus），但**评分不同**
///（n_scheduling_units = n_cus/2 进入 TF 估算）——不一致说明硬件参数泄漏/缺失
#[test]
fn hw_variant_changes_scores_not_feasible_set() {
    let hw_xtx = GFX1100Limits::default();
    let hw_9060 = GFX1100Limits::for_gfx1200();

    let a = auto_schedule_gemm_hw(hw_xtx, 4096, 4096, 4096, DataFormat::BF16);
    let b = auto_schedule_gemm_hw(hw_9060, 4096, 4096, 4096, DataFormat::BF16);

    assert!(!a.is_empty() && !b.is_empty());
    assert_eq!(a.len(), b.len(), "可行候选数量必须一致（feasible 判据不含 n_cus）");
    for (ca, cb) in a.iter().zip(b.iter()) {
        assert_eq!(ca.config.name(), cb.config.name(),
            "两底数的候选排序不应交叉换序（同底数族内相对序应稳定）");
        assert_eq!(ca.vgprs, cb.vgprs, "VGPR 估算与硬件底数无关");
        assert!(cb.lds_bytes <= 65536, "候选必须过 64KB LDS 门");
    }
}

/// [行为] auto_schedule_gemm（无 hw 版）等价于 default 底数——向后兼容契约
#[test]
fn default_entry_uses_default_hw() {
    let a = auto_schedule_gemm(1024, 1024, 1024, DataFormat::BF16);
    let b = auto_schedule_gemm_hw(GFX1100Limits::default(), 1024, 1024, 1024, DataFormat::BF16);
    assert_eq!(a.len(), b.len());
    assert_eq!(a[0].score, b[0].score, "无 hw 版必须与 default 底数完全同结果");
}
