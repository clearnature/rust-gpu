//! Fused RMSNorm + GEMM operation — integrates ffn_fused_kernels into the tensor graph.
//!
//! For single-token decode (M=1), replaces the2-kernel sequence:
//!   1. rmsnorm(x, gamma) → normed [1, D]
//!   2. matmul(normed, W) → out [1, N]
//!
//! With a single fused kernel:
//!   fused_rmsnorm_gemm(x, gamma, W) → out [1, N]
//!
//! Savings: eliminates GMEM write + read of normed[D] (D × 4 bytes × 2).
//!
//! For multi-token prefill (M>1), falls back to separate rmsnorm + matmul
//! (the fused kernel works but is not yet optimized for batched GEMV).

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Fused RMSNorm + GEMM/GEMV.
///
/// Computes: `out[j] = Σᵢ (x[i] * rsqrt(mean(x²)+eps) * gamma[i]) * W[i,j]`
///
/// For M=1 (single-token decode), launches a single fused kernel that:
/// 1. Reads x[0..D] once → accumulates sum_sq
/// 2. wg_reduce_sum → inv_rms
/// 3. Reloads x2, loads gamma + weight → normed × weight inline
/// 4. wg_reduce_sum → scalar output per column
///
/// For M>1 (prefill), falls back to separate rmsnorm + matmul.
#[cfg(feature = "rocm")]
pub fn fused_rmsnorm_gemm(
    x: &Tensor,
    gamma: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor, String> {
    let x_shape = x.shape();
    let w_shape = weight.shape();
    assert_eq!(x_shape.len(), 2, "fused_rmsnorm_gemm: x must be 2D");
    assert_eq!(w_shape.len(), 2, "fused_rmsnorm_gemm: weight must be 2D");

    let m = x_shape[0];
    let d = x_shape[1];
    let n = w_shape[1];
    assert_eq!(w_shape[0], d, "weight dim mismatch");
    assert_eq!(gamma.numel(), d, "gamma dim mismatch");

    let runtime = x.runtime().clone();

    if m == 1 {
        // Single-token decode: use fused kernel
        fused_rmsnorm_gemm_decode(&runtime, x, gamma, weight, d, n, eps)
    } else {
        // Multi-token prefill: separate rmsnorm + matmul
        let normed = super::rmsnorm::rmsnorm(x, gamma, &runtime.device)?;
        super::bf16_matmul::matmul(&normed, weight, &runtime.device)
    }
}

/// Fused RMSNorm + GEMV for decode (M=1).
///
/// Uses the ffn_fused_rmsnorm_gemm kernel (128 threads, 4 waves).
/// Launches N workgroups (one per output element).
#[cfg(feature = "rocm")]
fn fused_rmsnorm_gemm_decode(
    runtime: &Arc<GpuRuntime>,
    x: &Tensor,
    gamma: &Tensor,
    weight: &Tensor,
    d: usize,
    n: usize,
    eps: f32,
) -> Result<Tensor, String> {
    use crate::t0::ffn_fused_kernels;
    use crate::t0::ir::Target;

    // Build or reuse the fused kernel
    let kernel = {
        let name = format!("ffn_fused_d{}_e{}", d, (eps * 1e8) as u32);
        let cached = runtime.get_kernel(&name);
        if let Some(k) = cached { k } else {
            let kb = ffn_fused_kernels::build_ffn_fused_rmsnorm_gemm();
            let compiled = kb.compile(Target::detect())?;
            runtime.compile_dsl(compiled)?
        }
    };

    // Allocate output [N] f32
    let out_buf = runtime.alloc(n * 4).map_err(|e| e.to_string())?;
    out_buf.zero();

    // Kernarg layout: [residual:u64, hidden:u64, gamma:u64, weight:u64, output:u64, D:u32, eps:f32]
    // For pure RMSNorm+GEMV (no residual add), residual = x
    let x_addr = x.buffer().gpu_addr();
    let gamma_addr = gamma.buffer().gpu_addr();
    let wt_addr = weight.buffer().gpu_addr();
    let out_addr = out_buf.gpu_addr();

    let d_u32 = d as u32;
    let mut ka = Vec::with_capacity(44);
    ka.extend_from_slice(&x_addr.to_le_bytes());       // residual
    ka.extend_from_slice(&x_addr.to_le_bytes());       // hidden
    ka.extend_from_slice(&gamma_addr.to_le_bytes());   // gamma
    ka.extend_from_slice(&wt_addr.to_le_bytes());      // weight
    ka.extend_from_slice(&out_addr.to_le_bytes());     // output
    ka.extend_from_slice(&d_u32.to_le_bytes());        // D
    ka.extend_from_slice(&eps.to_le_bytes());           // eps

    // Dispatch: N workgroups × 128 threads
    let wg_size = ffn_fused_kernels::ffn_fused_wg_size();
    runtime.dispatch(&kernel, [n as u32 * wg_size, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    Ok(Tensor::from_buffer(out_arc, runtime, &[1, n], DType::F32, "fused_rmsnorm_gemm_out"))
}

/// Fused residual_add + RMSNorm + GEMV (full3-op fusion).
///
/// Computes: `out[j] = Σᵢ ((residual[i] + hidden[i]) * rsqrt(mean(x2²)+eps) * gamma[i]) * W[i,j]`
///
/// Use for FFN sub-layer: residual add + norm + down projection in one kernel.
#[cfg(feature = "rocm")]
pub fn fused_residual_rmsnorm_gemm(
    residual: &Tensor,
    hidden: &Tensor,
    gamma: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor, String> {
    let r_shape = residual.shape();
    assert_eq!(r_shape, hidden.shape(), "residual and hidden must match");
    assert_eq!(r_shape.len(), 2, "must be 2D");

    let m = r_shape[0];
    let runtime = residual.runtime().clone();

    if m == 1 {
        // Decode: use fused kernel with residual+hidden as separate inputs
        // For now: compute x2 = residual + hidden, then fuse rmsnorm+gemv
        // TODO: extend kernel to accept separate residual/hidden pointers
        let x2 = super::add::add(residual, hidden, &runtime.device)?;
        let d = r_shape[1];
        let n = weight.shape()[1];
        fused_rmsnorm_gemm_decode(&runtime, &x2, gamma, weight, d, n, eps)
    } else {
        // Prefill: separate ops
        let x2 = super::add::add(residual, hidden, &runtime.device)?;
        let normed = super::rmsnorm::rmsnorm(&x2, gamma, &runtime.device)?;
        super::bf16_matmul::matmul(&normed, weight, &runtime.device)
    }
}
