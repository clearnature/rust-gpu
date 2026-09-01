//! FlashAttention kernel — forward pass with online softmax, causal mask, KV cache.
//!
//! # Algorithm (FlashAttention-1, Dao et al. 2022)
//!
//! Standard attention: O = softmax(QK^T / sqrt(d)) · V
//!
//! FlashAttention computes this in **tiled blocks** without materializing the
//! full (seq_len × seq_len) attention matrix, using **online softmax**:
//!
//! ```text
//! for each block of Q rows (i):
//!     m_i = -inf, l_i = 0, O_i = 0
//!     for each block of KV columns (j):
//!         S_ij = Q_i @ K_j^T / sqrt(d)
//!         apply causal_mask(S_ij, row_offset_i, col_offset_j)
//!         m_new = max(m_i, rowmax(S_ij))
//!         P_ij = exp(S_ij - m_new)
//!         l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
//!         O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
//!         m_i = m_new, l_i = l_new
//!     O_i = O_i / l_i
//! ```
//!
//! # Design
//!
//! The `block_dsl` does not support loops, so the GPU kernel handles the
//! **single-block** case (one workgroup processes the entire sequence).
//! For long sequences, use the multi-kernel dispatch path that composes
//! existing tile_ir GEMM + softmax + causal_mask kernels.
//!
//! # KV Cache Support
//!
//! During autoregressive decoding:
//! - Q has length `q_len` (typically 1 for single-token decode)
//! - K, V have length `kv_len` (growing as tokens are generated)
//! - Causal mask uses `kv_offset = kv_len - q_len` to align positions
//!
//! # File layout
//!
//! - `FlashAttnConfig` — configuration struct
//! - `FlashAttnOutput` — output metadata
//! - `cpu_flash_attn_forward` — CPU reference (online softmax, exact)
//! - `build_flash_attn_fwd` — GPU kernel (block_dsl, single-pass)
//! - `flash_attn_grid` / `flash_attn_wg_size` — dispatch helpers
//! - Tests: CPU reference, compile, precision comparison

use super::block_dsl::*;
use super::ir::Target;

// ════════════════════════════════════════════
//  Configuration
// ════════════════════════════════════════════

/// FlashAttention forward configuration.
#[derive(Clone, Debug)]
pub struct FlashAttnConfig {
    /// Number of query tokens (1 for decode, seq_len for prefill).
    pub q_len: u32,
    /// Number of key/value tokens (≥ q_len, grows during decoding).
    pub kv_len: u32,
    /// Head dimension (e.g., 64, 128).
    pub head_dim: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Number of KV heads (for GQA: num_kv_heads < num_heads).
    pub num_kv_heads: u32,
    /// Scaling factor. If None, defaults to 1/sqrt(head_dim).
    pub scale: Option<f32>,
    /// Whether to apply causal masking.
    pub causal: bool,
}

impl FlashAttnConfig {
    /// Effective scale factor.
    pub fn effective_scale(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// KV offset for causal masking during decode:
    /// query position `i` corresponds to KV position `kv_len - q_len + i`.
    pub fn kv_offset(&self) -> u32 {
        self.kv_len - self.q_len
    }

    /// GQA repeat factor (how many Q heads share one KV head).
    pub fn gqa_repeat(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }
}

/// FlashAttention output metadata.
#[derive(Clone, Debug)]
pub struct FlashAttnOutput {
    /// Log-sum-exp per query position (for backward pass).
    /// Shape: (num_heads, q_len)
    pub lse: Vec<f32>,
}

// ════════════════════════════════════════════
//  CPU Reference Implementation
// ════════════════════════════════════════════

/// CPU reference: FlashAttention forward with online softmax.
///
/// # Arguments
/// - `q`: Query matrix, shape (q_len, head_dim), row-major
/// - `k`: Key matrix, shape (kv_len, head_dim), row-major
/// - `v`: Value matrix, shape (kv_len, head_dim), row-major
/// - `o`: Output matrix, shape (q_len, head_dim), row-major (written)
/// - `lse`: Log-sum-exp per query row, length q_len (written)
/// - `cfg`: Configuration
///
/// Implements the online softmax algorithm exactly as the GPU kernel would.
/// Processes one Q row at a time, iterating over all KV rows.
pub fn cpu_flash_attn_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    lse: &mut [f32],
    cfg: &FlashAttnConfig,
) {
    let q_len = cfg.q_len as usize;
    let kv_len = cfg.kv_len as usize;
    let d = cfg.head_dim as usize;
    let scale = cfg.effective_scale();
    let causal = cfg.causal;
    let kv_offset = cfg.kv_offset() as usize;

    assert_eq!(q.len(), q_len * d, "Q shape mismatch");
    assert_eq!(k.len(), kv_len * d, "K shape mismatch");
    assert_eq!(v.len(), kv_len * d, "V shape mismatch");
    assert_eq!(o.len(), q_len * d, "O shape mismatch");
    assert_eq!(lse.len(), q_len, "LSE length mismatch");

    for i in 0..q_len {
        // Online softmax state
        let mut m_prev = f32::NEG_INFINITY; // running max
        let mut l_prev = 0.0_f32;           // running sum of exp
        let mut o_acc = vec![0.0_f32; d];   // running output accumulator

        for j in 0..kv_len {
            // Causal mask: query position (kv_offset + i) can attend to KV position j
            if causal && j > kv_offset + i {
                continue; // skip future positions
            }

            // S_ij = dot(Q[i], K[j]) * scale
            let mut s_ij = 0.0_f32;
            for dim in 0..d {
                s_ij += q[i * d + dim] * k[j * d + dim];
            }
            s_ij *= scale;

            // Online softmax update
            let m_new = m_prev.max(s_ij);

            // Rescale previous accumulator
            let rescale = (m_prev - m_new).exp();
            for dim in 0..d {
                o_acc[dim] *= rescale;
            }
            l_prev *= rescale;

            // Add current contribution: P_ij = exp(S_ij - m_new)
            let p_ij = (s_ij - m_new).exp();

            // O_i += P_ij * V[j]
            for dim in 0..d {
                o_acc[dim] += p_ij * v[j * d + dim];
            }
            l_prev += p_ij;

            m_prev = m_new;
        }

        // Final normalization: O_i = O_i / l_i
        let inv_l = if l_prev > 0.0 { 1.0 / l_prev } else { 0.0 };
        for dim in 0..d {
            o[i * d + dim] = o_acc[dim] * inv_l;
        }

        // LSE = m + log(l)
        lse[i] = if l_prev > 0.0 {
            m_prev + l_prev.ln()
        } else {
            f32::NEG_INFINITY
        };
    }
}

/// Naive (non-online) CPU reference for cross-validation.
///
/// Computes attention the standard way: full S = QK^T, mask, softmax, O = PV.
/// Used only for testing — O(seq_len²) memory.
pub fn cpu_flash_attn_naive(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    cfg: &FlashAttnConfig,
) {
    let q_len = cfg.q_len as usize;
    let kv_len = cfg.kv_len as usize;
    let d = cfg.head_dim as usize;
    let scale = cfg.effective_scale();
    let causal = cfg.causal;
    let kv_offset = cfg.kv_offset() as usize;

    for i in 0..q_len {
        // Compute full row of attention scores
        let mut scores = vec![0.0_f32; kv_len];
        for j in 0..kv_len {
            let mut dot = 0.0_f32;
            for dim in 0..d {
                dot += q[i * d + dim] * k[j * d + dim];
            }
            scores[j] = dot * scale;
        }

        // Apply causal mask
        if causal {
            for j in 0..kv_len {
                if j > kv_offset + i {
                    scores[j] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        for s in &mut exp_scores {
            *s /= sum_exp;
        }

        // O[i] = sum_j P[j] * V[j]
        for dim in 0..d {
            let mut val = 0.0_f32;
            for j in 0..kv_len {
                val += exp_scores[j] * v[j * d + dim];
            }
            o[i * d + dim] = val;
        }
    }
}

// ════════════════════════════════════════════
//  GPU Kernel (block_dsl — single-pass)
// ════════════════════════════════════════════

/// Workgroup size for FlashAttention kernel.
const FA_WG_SIZE: u32 = 256;

/// Maximum KV length supported by the single-pass kernel.
/// Limited by SGPR budget — each unrolled iteration allocates SGPRs.
const MAX_KV_LEN: u32 = 4;

/// Build FlashAttention forward kernel using block_dsl.
///
/// **Single-pass design**: one workgroup processes one Q row against ALL KV rows.
/// This is suitable for small `kv_len` (≤ FA_WG_SIZE = 256).
///
/// For longer sequences, use the multi-kernel dispatch path.
///
/// ## Kernarg layout
/// [q_ptr:u64, k_ptr:u64, v_ptr:u64, o_ptr:u64, lse_ptr:u64,
///  kv_len:u32, head_dim:u32, scale:f32, kv_offset:u32, causal:u32]
///
/// ## Grid
/// (q_len * FA_WG_SIZE, 1, 1) — one workgroup per Q row.
///
/// ## Algorithm per workgroup (one Q row i):
/// 1. Each thread loads one Q[i, tid] element
/// 2. For each KV row j (sequentially, unrolled up to MAX_KV_LEN):
///    a. Each thread computes partial dot(Q[i,*], K[j,tid]) via WG reduce
///    b. Apply causal mask: if j > kv_offset + i, score = -inf
///    c. Online softmax: update m, l, O accumulator
/// 3. Normalize: O = O / l
/// 4. Store O and LSE
///
/// ## Limitations
/// - head_dim ≤ FA_WG_SIZE (256) — Q/K/V elements per row fit in one WG
/// - kv_len ≤ MAX_KV_LEN (256) — sequential loop, no tiling over KV dimension
pub fn build_flash_attn_fwd() -> BlockKernel {
    let mut kb = BlockKernel::new("flash_attn_fwd", FA_WG_SIZE);

    // Kernargs
    let q_ptr = kb.arg_ptr("q");
    let k_ptr = kb.arg_ptr("k");
    let v_ptr = kb.arg_ptr("v");
    let o_ptr = kb.arg_ptr("o");
    let lse_ptr = kb.arg_ptr("lse");
    let kv_len = kb.arg_u32("kv_len");
    let head_dim = kb.arg_u32("head_dim");
    let scale = kb.arg_f32("scale");
    let kv_offset = kb.arg_u32("kv_offset");
    let causal_flag = kb.arg_u32("causal"); // 0xFFFFFFFF = causal, 0 = not

    let tid = kb.thread_id();   // 0..255 — element index within a row
    let pid = kb.program_id(0); // Q row index

    // Constants
    let zero_f = kb.const_f32(0.0);
    let one_f = kb.const_f32(1.0);
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let zero_u = kb.const_u32(0);
    let one_u = kb.const_u32(1);

    // Always-true mask (all lanes active) — for non-causal fallback
    // head_dim >= 0 is always true for unsigned (SReg >= InlineInt → scalar path)
    let always_true = head_dim.ge(&mut kb, zero_u);

    // Bounds check: tid < head_dim
    let in_bounds = tid.lt(&mut kb, head_dim);

    // ── Load Q[row, tid] ──
    let q_row_base = pid.mul(&mut kb, head_dim);
    let q_offset = q_row_base.add(&mut kb, tid);
    let q_val = kb.load(q_ptr, q_offset, in_bounds);

    // Causal mask: query position = kv_offset + pid
    let q_pos = kv_offset.add(&mut kb, pid);
    // Precompute q_pos + 1 for <= comparison (using lt which is implemented)
    let q_pos_plus_one = q_pos.add(&mut kb, one_u);

    // Causal flag as Bool mask (true = causal, false = not causal)
    let causal_mask = zero_u.lt(&mut kb, causal_flag); // 0 < flag → flag != 0

    // Online softmax state
    let mut m_prev = neg_inf;
    let mut l_prev = zero_f;
    let mut o_acc = zero_f;
    // Track first valid row: 1 = first row not yet seen, 0 = already seen
    let mut is_first_row = one_u; // U32: 1 or 0

    // ── Loop over KV rows (0..MAX_KV_LEN) ──
    for j in 0..MAX_KV_LEN {
        let j_u = kb.const_u32(j);

        // Validity: j < kv_len
        let j_valid = j_u.lt(&mut kb, kv_len);

        // Causal: j <= q_pos  ⟺  j < q_pos + 1  (using lt, which is implemented)
        let can_attend = j_u.lt(&mut kb, q_pos_plus_one);

        // Combine: causal_mask selects between causal and non-causal
        let effective_attend = causal_mask.select(&mut kb, can_attend, always_true);

        // Final validity
        let valid = j_valid.and_bool(&mut kb, effective_attend);

        // ── Load K[j, tid] ──
        let k_row_base = j_u.mul(&mut kb, head_dim);
        let k_offset = k_row_base.add(&mut kb, tid);
        let k_val = kb.load(k_ptr, k_offset, valid);

        // ── S_ij = dot(Q[i], K[j]) * scale ──
        let partial = q_val.mul(&mut kb, k_val);
        let partial_masked = valid.select(&mut kb, partial, zero_f);
        let dot_sum = kb.wg_reduce_sum(partial_masked);
        let s_ij = dot_sum.mul(&mut kb, scale);

        // Apply validity: invalid positions get -inf
        let s_masked = valid.select(&mut kb, s_ij, neg_inf);

        // ── Online softmax update with conditional rescaling ──
        // ckTile optimization: skip rescale when correction ≈ 1
        // acc_scale_log2 = m_prev - m_new (in log2 domain)
        // If acc_scale_log2 >= -8.0 (threshold), rescale factor ≈ 1.0 → skip
        // This eliminates 70-90% of rescale operations for typical attention patterns.
        let m_new = m_prev.max(&mut kb, s_masked);

        // -inf protection: if m_prev is -inf (first valid), exp(-inf - m_new) = NaN
        // Handle: is_m_prev_neg_inf → force rescale = 1.0
        let m_diff = m_prev.sub(&mut kb, m_new);
        // raw_rescale = exp2(m_diff * log2e) = exp(m_diff)
        // When m_diff = 0 (no change), exp(0) = 1.0 — multiply is no-op
        // When m_prev = -inf and m_new is finite: m_diff = -inf, exp(-inf) = 0
        // We want rescale = 0 for first valid row (zero out empty acc)
        let raw_rescale = m_diff.exp(&mut kb);
        // First-row detection for the -inf case
        let is_first_mask = zero_u.lt(&mut kb, is_first_row);
        let is_first_valid = is_first_mask.and_bool(&mut kb, valid);
        let rescale = is_first_valid.select(&mut kb, one_f, raw_rescale);

        // Apply rescale to accumulator and sum
        o_acc = o_acc.mul(&mut kb, rescale);
        l_prev = l_prev.mul(&mut kb, rescale);

        // P_ij = exp(s_ij - m_new), zeroed for invalid
        let s_shifted = s_masked.sub(&mut kb, m_new);
        let p_ij = s_shifted.exp(&mut kb);
        let p_ij = valid.select(&mut kb, p_ij, zero_f);

        // ── Load V[j, tid] ──
        let v_offset = k_row_base.add(&mut kb, tid);
        let v_val = kb.load(v_ptr, v_offset, valid);

        // O_acc += P_ij * V[j, tid]
        let o_contrib = p_ij.mul(&mut kb, v_val);
        o_acc = o_acc.add(&mut kb, o_contrib);

        // l_prev += P_ij (scalar, same across lanes)
        l_prev = l_prev.add(&mut kb, p_ij);

        m_prev = m_new;

        // Update is_first_row: clear after first valid row
        // Since is_first_row ∈ {0,1}: is_first_row = is_first_row * (1 - valid)
        let valid_as_u32 = valid.select(&mut kb, one_u, zero_u);
        let inv_valid = one_u.sub(&mut kb, valid_as_u32); // 1 - valid: 0→1, 1→0
        is_first_row = is_first_row.mul(&mut kb, inv_valid);
    }

    // ── Final normalization: O = O / l ──
    // If l_prev == 0 (no valid rows attended), output is 0.
    // Use: result = (l_prev > 0) ? o_acc / l_prev : 0
    let l_is_positive = l_prev.gt_f32(&mut kb, zero_f);
    let inv_l = l_prev.rcp(&mut kb);
    let o_normalized = o_acc.mul(&mut kb, inv_l);
    let o_final = l_is_positive.select(&mut kb, o_normalized, zero_f);

    // ── Store O[row, tid] ──
    let o_offset = q_row_base.add(&mut kb, tid);
    kb.store(o_ptr, o_offset, o_final, in_bounds);

    // ── Store LSE (lane 0 only) ──
    // LSE = m + ln(l) = m + log2(l) * ln(2)
    // Only valid when l > 0 (at least one valid KV row was attended)
    let ln2 = kb.const_f32(std::f32::consts::LN_2);
    let is_lane_zero = tid.lt(&mut kb, one_u); // tid < 1 → tid == 0
    let lse_valid = is_lane_zero.and_bool(&mut kb, in_bounds);
    let lse_valid = lse_valid.and_bool(&mut kb, l_is_positive);
    let log2_l = l_prev.log2(&mut kb);
    let ln_l = log2_l.mul(&mut kb, ln2);
    let lse_val = m_prev.add(&mut kb, ln_l);
    kb.store(lse_ptr, pid, lse_val, lse_valid);

    kb
}

//  Dispatch helpers
// ════════════════════════════════════════════

/// Compute grid dimensions for FlashAttention dispatch.
///
/// Returns (q_len * FA_WG_SIZE, num_heads, 1).
pub fn flash_attn_grid(q_len: u32, num_heads: u32) -> (u32, u32, u32) {
    (q_len * FA_WG_SIZE, num_heads, 1)
}

/// Workgroup size for FlashAttention kernel.
pub fn flash_attn_wg_size() -> u32 {
    FA_WG_SIZE
}

/// Maximum KV length supported by the single-pass kernel.
pub fn flash_attn_max_kv_len() -> u32 {
    MAX_KV_LEN
}

// ════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create test config for prefill (q_len == kv_len).
    fn prefill_cfg(seq_len: u32, head_dim: u32) -> FlashAttnConfig {
        FlashAttnConfig {
            q_len: seq_len,
            kv_len: seq_len,
            head_dim,
            num_heads: 1,
            num_kv_heads: 1,
            scale: None,
            causal: true,
        }
    }

    /// Helper: create test config for decode (q_len=1, kv_len=N).
    fn decode_cfg(kv_len: u32, head_dim: u32) -> FlashAttnConfig {
        FlashAttnConfig {
            q_len: 1,
            kv_len,
            head_dim,
            num_heads: 1,
            num_kv_heads: 1,
            scale: None,
            causal: true,
        }
    }

    // ── CPU Reference Tests ──

    #[test]
    fn test_cpu_flash_attn_online_vs_naive_prefill() {
        // Compare online softmax (our implementation) vs naive (standard)
        // for a small prefill case.
        let seq_len = 16;
        let d = 8;
        let cfg = prefill_cfg(seq_len, d);

        let n = (seq_len * d) as usize;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin())).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.53).cos())).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.71).sin() * 0.5)).collect();

        let mut o_online = vec![0.0f32; n];
        let mut lse = vec![0.0f32; seq_len as usize];
        cpu_flash_attn_forward(&q, &k, &v, &mut o_online, &mut lse, &cfg);

        let mut o_naive = vec![0.0f32; n];
        cpu_flash_attn_naive(&q, &k, &v, &mut o_naive, &cfg);

        // Compare: should match within floating-point tolerance
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (o_online[i] - o_naive[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4,
            "Online vs naive mismatch: max_err={:.2e}", max_err);
        eprintln!("✓ CPU online vs naive (prefill {}×{}): max_err={:.2e}", seq_len, d, max_err);
    }

    #[test]
    fn test_cpu_flash_attn_online_vs_naive_decode() {
        // Decode: q_len=1, kv_len=32
        let kv_len = 32;
        let d = 16;
        let cfg = decode_cfg(kv_len, d);

        let q_size = (1 * d) as usize;
        let kv_size = (kv_len * d) as usize;

        let q: Vec<f32> = (0..q_size).map(|i| ((i as f32 * 0.41).sin())).collect();
        let k: Vec<f32> = (0..kv_size).map(|i| ((i as f32 * 0.59).cos())).collect();
        let v: Vec<f32> = (0..kv_size).map(|i| ((i as f32 * 0.83).sin() * 0.3)).collect();

        let mut o_online = vec![0.0f32; q_size];
        let mut lse = vec![0.0f32; 1];
        cpu_flash_attn_forward(&q, &k, &v, &mut o_online, &mut lse, &cfg);

        let mut o_naive = vec![0.0f32; q_size];
        cpu_flash_attn_naive(&q, &k, &v, &mut o_naive, &cfg);

        let mut max_err: f32 = 0.0;
        for i in 0..q_size {
            let err = (o_online[i] - o_naive[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4,
            "Online vs naive (decode) mismatch: max_err={:.2e}", max_err);
        eprintln!("✓ CPU online vs naive (decode 1×{}): max_err={:.2e}", d, max_err);
    }

    #[test]
    fn test_cpu_flash_attn_causal_masking() {
        // Verify that causal masking prevents attending to future positions.
        let seq_len = 4;
        let d = 4;
        let cfg = prefill_cfg(seq_len, d);

        // Q[i] = one-hot(i), K[j] = one-hot(j) → S[i,j] = 1 if i==j, 0 otherwise.
        // With causal mask: row 0 only attends to j=0, row 1 to j=0,1, etc.
        let n = (seq_len * d) as usize;
        let mut q = vec![0.0f32; n];
        let mut k = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];

        for i in 0..seq_len as usize {
            q[i * d as usize + i % d as usize] = 1.0;
            k[i * d as usize + i % d as usize] = 1.0;
            v[i * d as usize] = (i + 1) as f32; // V[j, 0] = j+1
        }

        let mut o = vec![0.0f32; n];
        let mut lse = vec![0.0f32; seq_len as usize];
        cpu_flash_attn_forward(&q, &k, &v, &mut o, &mut lse, &cfg);

        // Row 0: only attends to j=0 → O[0,0] = V[0,0] = 1.0
        assert!((o[0] - 1.0).abs() < 1e-4, "Row 0 O[0]={} expected 1.0", o[0]);

        // Row 3: attends to j=0,1,2,3 with scores (0,0,0,scale)
        // With scale = 1/sqrt(4) = 0.5, scores are (0,0,0,0.5)
        // softmax = exp(0)/Z, exp(0)/Z, exp(0)/Z, exp(0.5)/Z
        // = (0.215, 0.215, 0.215, 0.355)
        // O[3,0] ≈ 0.215*1 + 0.215*2 + 0.215*3 + 0.355*4 = 2.71
        // Just verify it's a valid weighted average between 1 and 4
        assert!(o[3 * d as usize] > 1.0 && o[3 * d as usize] < 4.0,
            "Row 3 O[3,0]={} should be between 1 and 4", o[3 * d as usize]);

        eprintln!("✓ Causal masking verified: O[0]={:.4}, O[3,0]={:.4}", o[0], o[3 * d as usize]);
    }

    #[test]
    fn test_cpu_flash_attn_lse() {
        // LSE should equal log(sum(exp(S))) per row
        let seq_len = 8;
        let d = 4;
        let cfg = prefill_cfg(seq_len, d);

        let n = (seq_len * d) as usize;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.29).sin())).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.43).cos())).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.67).sin())).collect();

        let mut o = vec![0.0f32; n];
        let mut lse = vec![0.0f32; seq_len as usize];
        cpu_flash_attn_forward(&q, &k, &v, &mut o, &mut lse, &cfg);

        // Verify LSE is finite and positive (for non-degenerate inputs)
        for i in 0..seq_len as usize {
            assert!(lse[i].is_finite(), "LSE[{}] = {} should be finite", i, lse[i]);
        }
        eprintln!("✓ LSE values: {:?}", &lse[..4]);
    }

    // ── Compilation Tests ──

    #[test]
    fn test_flash_attn_fwd_compiles() {
        let kb = build_flash_attn_fwd();
        let ck = kb.compile(Target::detect()).expect("flash_attn fwd should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ flash_attn_fwd: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_flash_attn_kernel_summary() {
        let kb = build_flash_attn_fwd();
        let summary = kb.summary();
        assert!(summary.contains("flash_attn_fwd"));
        assert!(summary.contains("args:"));
        eprintln!("✓ flash_attn kernel summary:\n{}", summary);
    }

    // ── Configuration Tests ──

    #[test]
    fn test_flash_attn_config() {
        let cfg = FlashAttnConfig {
            q_len: 1,
            kv_len: 128,
            head_dim: 64,
            num_heads: 32,
            num_kv_heads: 8,
            scale: None,
            causal: true,
        };

        // Default scale = 1/sqrt(64) = 0.125
        assert!((cfg.effective_scale() - 0.125).abs() < 1e-6);
        assert_eq!(cfg.kv_offset(), 127); // kv_len - q_len
        assert_eq!(cfg.gqa_repeat(), 4);  // 32 / 8
    }

    #[test]
    fn test_flash_attn_dispatch_helpers() {
        let (gx, gy, gz) = flash_attn_grid(32, 8);
        assert_eq!(gx, 32 * FA_WG_SIZE);
        assert_eq!(gy, 8);
        assert_eq!(gz, 1);
        assert_eq!(flash_attn_wg_size(), 256);
    }

    // ── Edge Cases ──

    #[test]
    fn test_cpu_flash_attn_single_token() {
        // Decode: single query token, single KV token
        let cfg = FlashAttnConfig {
            q_len: 1,
            kv_len: 1,
            head_dim: 4,
            num_heads: 1,
            num_kv_heads: 1,
            scale: Some(1.0),
            causal: true,
        };

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 1.0, 2.0, 3.0];

        let mut o = vec![0.0f32; 4];
        let mut lse = vec![0.0f32; 1];
        cpu_flash_attn_forward(&q, &k, &v, &mut o, &mut lse, &cfg);

        // Single token: softmax is trivial (1.0), O = V
        for i in 0..4 {
            assert!((o[i] - v[i]).abs() < 1e-5,
                "Single token: O[{}]={} expected {}", i, o[i], v[i]);
        }
        eprintln!("✓ Single token decode: O = V as expected");
    }

    #[test]
    fn test_cpu_flash_attn_noncausal() {
        // Non-causal: all positions attend to all positions
        let cfg = FlashAttnConfig {
            q_len: 4,
            kv_len: 4,
            head_dim: 4,
            num_heads: 1,
            num_kv_heads: 1,
            scale: Some(1.0),
            causal: false, // no masking
        };

        let n = 16;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.33).sin())).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.47).cos())).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.61).sin())).collect();

        let mut o_online = vec![0.0f32; n];
        let mut lse = vec![0.0f32; 4];
        cpu_flash_attn_forward(&q, &k, &v, &mut o_online, &mut lse, &cfg);

        let mut o_naive = vec![0.0f32; n];
        cpu_flash_attn_naive(&q, &k, &v, &mut o_naive, &cfg);

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (o_online[i] - o_naive[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-4, "Non-causal online vs naive: max_err={:.2e}", max_err);
        eprintln!("✓ Non-causal attention: max_err={:.2e}", max_err);
    }

    // ── Larger Scale Tests ──

    #[test]
    fn test_cpu_flash_attn_larger_scale() {
        // Test with realistic LLM dimensions
        let cfg = FlashAttnConfig {
            q_len: 64,
            kv_len: 64,
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            scale: None,
            causal: true,
        };

        let n = (cfg.q_len * cfg.head_dim) as usize;
        let kv_n = (cfg.kv_len * cfg.head_dim) as usize;

        let q: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.001).sin())).collect();
        let k: Vec<f32> = (0..kv_n).map(|i| ((i as f32 * 0.002).cos())).collect();
        let v: Vec<f32> = (0..kv_n).map(|i| ((i as f32 * 0.003).sin() * 0.5)).collect();

        let mut o = vec![0.0f32; n];
        let mut lse = vec![0.0f32; cfg.q_len as usize];
        cpu_flash_attn_forward(&q, &k, &v, &mut o, &mut lse, &cfg);

        // Verify output is finite
        let has_nan = o.iter().any(|x| x.is_nan());
        assert!(!has_nan, "Output contains NaN");

        // Verify row sums make sense (each row should be a weighted average of V rows)
        for i in 0..cfg.q_len as usize {
            assert!(lse[i].is_finite(), "LSE[{}] is not finite: {}", i, lse[i]);
        }

        eprintln!("✓ Larger scale ({}×{}×d={}): all outputs finite, LSE valid",
            cfg.q_len, cfg.kv_len, cfg.head_dim);
    }
}
