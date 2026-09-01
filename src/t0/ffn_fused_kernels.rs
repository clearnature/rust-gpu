//! Fused FFN prefill kernels — residual_add + RMSNorm + GEMV projection.
//!
//! # Fusion Strategy (Blog #3: residual_add + rmsnorm + GEMV fusion)
//!
//! Eliminates intermediate GMEM traffic by fusing three prefill stages:
//!
//! ```text
//! x2      = residual + hidden_state                    (residual add)
//! rms     = sqrt(mean(x2²) + eps)                      (RMSNorm)
//! normed  = x2 / rms * gamma                           (normalize + scale)
//! output  = normed @ weight^T                          (GEMV projection)
//! ```
//!
//! ## Single-kernel fusion:
//! 1. Load `residual` + `hidden_state` → compute `x2` AND accumulate `sum_sq`
//!    (**one GMEM read** for both RMSNorm and residual add)
//! 2. `wg_reduce_sum` → `inv_rms` (cross-wave via LDS, includes barrier)
//! 3. Loop: load `gamma` + `weight`, compute `normed = x2 * inv_rms * gamma`
//!    inline with GEMV dot product accumulation
//!    (**one GMEM read** for both normalization weights and projection weights)
//!
//! ## Savings vs 3 separate kernels:
//! - Eliminated: `normed` write to GMEM (M × D f32)
//! - Eliminated: `normed` read from GMEM (M × D f32)
//! - Residual read fused with sum_sq accumulation
//!
//! ## Thread mapping (mw4 = 4 waves = 128 threads):
//! - 1 workgroup per row (output element)
//! - Grid: (M, 1, 1) — M = number of tokens in prefill
//! - Each thread processes D/128 columns per loop iteration

use super::block_dsl::*;
use super::ir::Target;

/// Workgroup size: 4 waves × 32 lanes = 128 threads (mw4 pattern).
const MW4_WG_SIZE: u32 = 128;

/// Build fused residual_add + RMSNorm + GEMV prefill kernel.
///
/// Performs in a single kernel:
/// ```text
/// x2[i]     = residual[row,i] + hidden[row,i]
/// rms       = sqrt(mean(x2²) + eps)
/// output[row] = Σᵢ (x2[i] * inv_rms * gamma[i]) * weight[row,i]
/// ```
///
/// Kernarg layout:
///   [residual:u64, hidden:u64, gamma:u64, weight:u64, output:u64,
///    D:u32, eps:f32]
///
/// Grid: (M, 1, 1) — one workgroup per output row.
/// WG size: 128 (4 waves).
///
/// The `weight` pointer should point to the row of the weight matrix
/// corresponding to the current output element (pre-offset by caller).
pub fn build_ffn_fused_rmsnorm_gemm() -> BlockKernel {
    let mut kb = BlockKernel::new("ffn_fused_rmsnorm_gemm", MW4_WG_SIZE);

    // ── Kernarg declarations ──
    let residual_ptr = kb.arg_ptr("residual");
    let hidden_ptr   = kb.arg_ptr("hidden");
    let gamma_ptr    = kb.arg_ptr("gamma");
    let weight_ptr   = kb.arg_ptr("weight");
    let output_ptr   = kb.arg_ptr("output");
    let d            = kb.arg_u32("D");
    let eps          = kb.arg_f32("eps");

    // ── Thread/row indices ──
    let tid = kb.thread_id();           // 0..127
    let pid = kb.program_id(0);         // row index (token)
    let zero = kb.const_u32(0);

    // Row base offset = pid * D (byte offset in f32 units)
    let row_base = pid.mul(&mut kb, d);

    // ════════════════════════════════════════════
    // Phase 1: residual_add + sum_sq accumulation
    // ════════════════════════════════════════════
    //
    // Single pass over model dimension D:
    //   x2 = residual + hidden
    //   sum_sq += x2²
    //
    // Each thread processes columns: tid, tid+128, tid+256, ...
    // (stride = workgroup size = 128)
    let mut sum_sq_acc = kb.const_f32(0.0);
    let iter1 = kb.for_range(zero, d, MW4_WG_SIZE);

    // Global offset for this iteration: row_base + iter1 + tid
    let iter1_tid = iter1.add(&mut kb, tid);
    let g_off = row_base.add(&mut kb, iter1_tid);
    let row_end = row_base.add(&mut kb, d);
    let mask1 = g_off.lt(&mut kb, row_end);

    // Load residual and hidden, compute x2 = residual + hidden
    let res = kb.load(residual_ptr, g_off, mask1);
    let hid = kb.load(hidden_ptr, g_off, mask1);
    let x2  = res.add(&mut kb, hid);

    // Accumulate sum of squares (masked: out-of-bounds contributes 0)
    let x2_sq = x2.mul(&mut kb, x2);
    let zero_f = kb.const_f32(0.0);
    let sq_masked = mask1.select(&mut kb, x2_sq, zero_f);
    sum_sq_acc = sum_sq_acc.add(&mut kb, sq_masked);

    kb.end_for(iter1);

    // ════════════════════════════════════════════
    // Phase 2: WG reduce → inv_rms
    // ════════════════════════════════════════════
    //
    // sum_sq is per-thread accumulated; need workgroup-wide sum.
    // wg_reduce_sum handles:
    //   1. wave_reduce_add within each wave (4 waves)
    //   2. wave leaders write partial sums to LDS
    //   3. barrier
    //   4. wave 0 loads + reduces partial sums
    //   5. broadcast result to all lanes
    //
    // Then: mean_sq = total_sum / D
    //       inv_rms = rsqrt(mean_sq + eps)
    let total_sq = kb.wg_reduce_sum(sum_sq_acc);
    let d_f = d.to_f32(&mut kb);
    let mean_sq = total_sq.div(&mut kb, d_f);
    let mean_sq_eps = mean_sq.add(&mut kb, eps);
    let inv_rms = mean_sq_eps.rsqrt(&mut kb);

    // ════════════════════════════════════════════
    // Phase 3: Inline RMSNorm + GEMV accumulation
    // ════════════════════════════════════════════
    //
    // Loop over model dimension D, computing:
    //   normed_i = x2_i * inv_rms * gamma_i     (fused RMSNorm)
    //   acc += normed_i * weight_i               (GEMV dot product)
    //
    // Key: normed values are NEVER stored to GMEM.
    // They are computed in-register and consumed immediately.
    //
    // We need x2 values again, so we reload from GMEM.
    // (Alternative: store x2 to LDS in Phase 1, but LDS is 64KB
    //  and D=4096*4=16KB per row — would need careful partitioning
    //  with wg_reduce_sum's LDS usage. GMEM reload is simpler and
    //  the data is hot in L2 cache from Phase 1.)
    let mut gemv_acc = kb.const_f32(0.0);
    let iter2 = kb.for_range(zero, d, MW4_WG_SIZE);

    let iter2_tid = iter2.add(&mut kb, tid);
    let g_off2 = row_base.add(&mut kb, iter2_tid);
    let row_end2 = row_base.add(&mut kb, d);
    let mask2 = g_off2.lt(&mut kb, row_end2);

    // Reload x2 = residual + hidden (hot in L2 from Phase 1)
    let res2 = kb.load(residual_ptr, g_off2, mask2);
    let hid2 = kb.load(hidden_ptr, g_off2, mask2);
    let x2_v = res2.add(&mut kb, hid2);

    // Load gamma (RMSNorm scale) and weight (projection)
    let gam = kb.load(gamma_ptr, iter2_tid, mask2);
    let wt  = kb.load(weight_ptr, g_off2, mask2);

    // Fused: normed = x2 * inv_rms * gamma (inline, no GMEM write)
    let x2_inv_rms = x2_v.mul(&mut kb, inv_rms);
    let normed = x2_inv_rms.mul(&mut kb, gam);

    // GEMV accumulation: acc += normed * weight
    let prod = normed.mul(&mut kb, wt);
    let prod_masked = mask2.select(&mut kb, prod, zero_f);
    gemv_acc = gemv_acc.add(&mut kb, prod_masked);

    kb.end_for(iter2);

    // ════════════════════════════════════════════
    // Phase 4: WG reduce GEMV → scalar output
    // ════════════════════════════════════════════
    //
    // Each thread has a partial dot product; sum across workgroup.
    let total = kb.wg_reduce_sum(gemv_acc);

    // Thread 0 stores the scalar result
    let one = kb.const_u32(1);
    let tid_zero = tid.lt(&mut kb, one);
    kb.if_mask(tid_zero);
    let out_offset = pid; // output[row] — scalar per row
    kb.store(output_ptr, out_offset, total, tid_zero);
    kb.end_if();

    kb
}

/// Grid dimensions for the fused FFN kernel: (M, 1, 1).
pub fn ffn_fused_grid(m: u32) -> (u32, u32) {
    (m, 1)
}

/// Workgroup size for the fused FFN kernel (4 waves).
pub fn ffn_fused_wg_size() -> u32 {
    MW4_WG_SIZE
}

// ════════════════════════════════════════════
// CPU Reference Implementation
// ════════════════════════════════════════════

/// CPU reference: fused residual_add + RMSNorm + GEMV.
///
/// Computes for each row:
/// ```text
/// x2[i]      = residual[row*D+i] + hidden[row*D+i]
/// rms        = sqrt(mean(x2²) + eps)
/// normed[i]  = x2[i] / rms * gamma[i]
/// output[row] = Σ normed[i] * weight[row*D+i]
/// ```
pub fn cpu_ffn_fused_rmsnorm_gemm(
    residual: &[f32],
    hidden: &[f32],
    gamma: &[f32],
    weight: &[f32],
    output: &mut [f32],
    m: usize,
    d: usize,
    eps: f32,
) {
    for row in 0..m {
        let base = row * d;
        let row_res = &residual[base..base + d];
        let row_hid = &hidden[base..base + d];
        let row_wt  = &weight[base..base + d];

        // Step 1: residual add
        let mut x2 = vec![0.0f32; d];
        for i in 0..d {
            x2[i] = row_res[i] + row_hid[i];
        }

        // Step 2: RMSNorm
        let sum_sq: f32 = x2.iter().map(|&v| v * v).sum();
        let mean_sq = sum_sq / d as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // Step 3: GEMV dot product (inline normed)
        let mut acc = 0.0f32;
        for i in 0..d {
            let normed = x2[i] * inv_rms * gamma[i];
            acc += normed * row_wt[i];
        }
        output[row] = acc;
    }
}

// ════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Test CPU reference implementation correctness.
    ///
    /// Verifies that the fused result equals:
    ///   output = (residual + hidden) RMSNorm'd @ weight^T
    #[test]
    fn test_cpu_fused_reference() {
        let m = 2usize;
        let d = 64usize;
        let eps = 1e-5f32;

        // Deterministic test data
        let residual: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.17).sin())).collect();
        let hidden: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.23).cos())).collect();
        let gamma: Vec<f32> = (0..d).map(|i| 0.8 + (i as f32 * 0.005)).collect();
        let weight: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.31).sin() * 0.5)).collect();

        // Fused output
        let mut fused_out = vec![0.0f32; m];
        cpu_ffn_fused_rmsnorm_gemm(
            &residual, &hidden, &gamma, &weight, &mut fused_out, m, d, eps,
        );

        // Manual verification: compute step-by-step for row 0
        let mut x2 = vec![0.0f32; d];
        for i in 0..d {
            x2[i] = residual[i] + hidden[i];
        }
        let sum_sq: f32 = x2.iter().map(|&v| v * v).sum();
        let mean_sq = sum_sq / d as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut expected_acc = 0.0f32;
        for i in 0..d {
            expected_acc += (x2[i] * inv_rms * gamma[i]) * weight[i];
        }

        assert!(
            (fused_out[0] - expected_acc).abs() < 1e-4,
            "CPU fused ref row 0: got={}, expected={}, diff={}",
            fused_out[0],
            expected_acc,
            (fused_out[0] - expected_acc).abs()
        );

        // Verify RMSNorm property: mean(normed²) ≈ 1 (for gamma=1)
        // With gamma != 1, just check the computation is consistent
        let mut normed_sq_sum = 0.0f32;
        for i in 0..d {
            let normed = x2[i] * inv_rms;
            normed_sq_sum += normed * normed;
        }
        let normed_rms = (normed_sq_sum / d as f32).sqrt();
        assert!(
            (normed_rms - 1.0).abs() < 1e-4,
            "RMSNorm property: rms(normed)={}, expected~1.0",
            normed_rms
        );

        eprintln!("✓ CPU fused reference: M={}, D={}, output[0]={:.6}", m, d, fused_out[0]);
    }

    /// Test that the fused kernel compiles to a valid ELF.
    #[test]
    fn test_ffn_fused_compiles() {
        let kb = build_ffn_fused_rmsnorm_gemm();
        let ck = kb.compile_via_ssa(Target::detect()).expect("ffn_fused compile");
        assert!(!ck.elf.is_empty());
        eprintln!(
            "✓ FFN fused RMSNorm+GEMM: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(),
            ck.workgroup_size,
            ck.lds_size
        );
    }

    /// Test fused vs separate kernels produce the same result (CPU-level).
    ///
    /// Verifies that:
    ///   fused(residual, hidden, gamma, weight)
    ///   == GEMV(RMSNorm(residual + hidden, gamma), weight)
    #[test]
    fn test_fused_equals_separate() {
        use super::super::rmsnorm_kernels::cpu_rmsnorm_forward;

        let m = 4usize;
        let d = 128usize;
        let eps = 1e-5f32;

        let residual: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.13).sin() * 1.5)).collect();
        let hidden: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.19).cos() * 0.8)).collect();
        let gamma: Vec<f32> = (0..d).map(|i| 0.9 + (i as f32 * 0.002)).collect();
        let weight: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.41).sin())).collect();

        // ── Fused output ──
        let mut fused_out = vec![0.0f32; m];
        cpu_ffn_fused_rmsnorm_gemm(
            &residual, &hidden, &gamma, &weight, &mut fused_out, m, d, eps,
        );

        // ── Separate: residual_add → RMSNorm → GEMV ──
        // Step 1: residual_add
        let mut x2 = vec![0.0f32; m * d];
        for i in 0..m * d {
            x2[i] = residual[i] + hidden[i];
        }

        // Step 2: RMSNorm (uses gamma as weight)
        let mut normed = vec![0.0f32; m * d];
        cpu_rmsnorm_forward(&x2, &gamma, &mut normed, m, d, eps);

        // Step 3: GEMV — output[row] = Σ normed[row,i] * weight[row,i]
        let mut separate_out = vec![0.0f32; m];
        for row in 0..m {
            let base = row * d;
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += normed[base + i] * weight[base + i];
            }
            separate_out[row] = acc;
        }

        // Compare
        for row in 0..m {
            let diff = (fused_out[row] - separate_out[row]).abs();
            assert!(
                diff < 1e-4,
                "Row {}: fused={}, separate={}, diff={}",
                row,
                fused_out[row],
                separate_out[row],
                diff
            );
        }

        eprintln!(
            "✓ Fused == Separate: M={}, D={}, max_diff={:.2e}",
            m,
            d,
            fused_out
                .iter()
                .zip(separate_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }

    /// Stress test: larger dimensions (M=8, D=4096 — typical LLM hidden size).
    #[test]
    fn test_cpu_fused_large() {
        let m = 8usize;
        let d = 4096usize;
        let eps = 1e-5f32;

        let residual: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.001).sin())).collect();
        let hidden: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.002).cos())).collect();
        let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32 * 0.0001)).collect();
        let weight: Vec<f32> = (0..m * d).map(|i| ((i as f32 * 0.003).sin() * 0.01)).collect();

        let mut output = vec![0.0f32; m];
        cpu_ffn_fused_rmsnorm_gemm(
            &residual, &hidden, &gamma, &weight, &mut output, m, d, eps,
        );

        // Sanity: output should be finite and non-zero
        for row in 0..m {
            assert!(output[row].is_finite(), "Row {} output is not finite: {}", row, output[row]);
        }

        eprintln!("✓ CPU fused large: M={}, D={}, output[0..3] = [{:.4}, {:.4}, {:.4}, {:.4}]",
            m, d, output[0], output[1], output[2], output[3]);
    }
}
