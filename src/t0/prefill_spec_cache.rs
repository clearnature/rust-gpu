//! Prefill Spec Cache — process-level cache for GEMM configuration selection.
//!
//! # Problem
//!
//! `auto_select(m, k, n)` → `cost_model::predict_best()` → `auto_schedule_gemm()`
//! runs an exhaustive search over 400+ tile configurations with hardware cost
//! modeling. This takes ~0.5-2 ms per call. During LLM prefill, the same
//! (M, K, N) dimensions repeat across layers (e.g., QKV projection, FFN),
//! making this redundant.
//!
//! # Solution
//!
//! A process-level `HashMap<(u32,u32,u32), GemmConfig>` cache that makes
//! repeated `auto_select()` calls effectively free (~20 ns for HashMap lookup).
//!
//! # Usage
//!
//! ```rust,ignore
//! use t0_gpu::t0::prefill_spec_cache::cached_auto_select;
//!
//! // First call: ~1 ms (cost model search)
//! let cfg1 = cached_auto_select(4096, 4096, 11008);
//!
//! // Subsequent calls with same dims: ~20 ns (cache hit)
//! let cfg2 = cached_auto_select(4096, 4096, 11008);
//! ```
//!
//! # Thread Safety
//!
//! Uses `OnceLock<Mutex<HashMap>>` — safe for multi-threaded access.
//! Lock contention is negligible (hold time ~20 ns for read, ~1 ms for
//! first-compute-then-insert).

use std::collections::HashMap;
use std::sync::Mutex;

use super::gemm_gen::GemmConfig;

/// Global spec cache: (M, K, N) → GemmConfig.
///
/// Uses `OnceLock` for lazy initialization (zero cost if never used).
static SPEC_CACHE: std::sync::OnceLock<Mutex<HashMap<(u32, u32, u32), GemmConfig>>> =
    std::sync::OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<(u32, u32, u32), GemmConfig>> {
    SPEC_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(64)))
}

/// Cached version of `auto_select()`.
///
/// Returns the cached GemmConfig for (M, K, N) if available,
/// otherwise computes via `cost_model::predict_best()` and caches the result.
///
/// This is a drop-in replacement for `super::gemm_gen::auto_select()`.
pub fn cached_auto_select(m: u32, k: u32, n: u32) -> GemmConfig {
    let key = (m, k, n);

    // Fast path: cache hit (lock held ~20 ns)
    {
        let cache = get_cache().lock().unwrap();
        if let Some(cfg) = cache.get(&key) {
            return cfg.clone();
        }
    }

    // Slow path: compute via cost model (~0.5-2 ms)
    let cfg = super::gemm_gen::auto_select(m, k, n);

    // Insert into cache
    {
        let mut cache = get_cache().lock().unwrap();
        cache.insert(key, cfg.clone());
    }

    cfg
}

/// Cached version of `auto_select_backward_data()`.
pub fn cached_auto_select_backward_data(m: u32, n_orig: u32, k_orig: u32) -> GemmConfig {
    // Backward data: dX[M,K] = dY[M,N] @ W[K,N]^T
    // GEMM: M=M, K=N_orig (contraction), N=K_orig (output)
    cached_auto_select(m, n_orig, k_orig)
}

/// Cached version of `auto_select_backward_weight()`.
///
/// Backward weight has its own selection path, so we cache it separately.
pub fn cached_auto_select_backward_weight(m: u32, n_orig: u32, k_orig: u32) -> GemmConfig {
    // Use a distinct key namespace by offsetting: add u32::MAX/2 to distinguish
    // from forward/backward-data keys. This avoids collision when
    // backward_weight(m, n, k) has same dims as forward(m, k, n).
    let key = (m.wrapping_add(0x8000_0000), n_orig, k_orig);

    {
        let cache = get_cache().lock().unwrap();
        if let Some(cfg) = cache.get(&key) {
            return cfg.clone();
        }
    }

    let cfg = super::gemm_gen::auto_select_backward_weight(m, n_orig, k_orig);
    {
        let mut cache = get_cache().lock().unwrap();
        cache.insert(key, cfg.clone());
    }
    cfg
}

/// Pre-warm the cache for known prefill dimensions.
///
/// Call this at startup or model load time to avoid first-call latency
/// during actual inference.
///
/// ```rust,ignore
/// use t0_gpu::t0::prefill_spec_cache::prewarm_cache;
///
/// // Pre-warm for LLaMA-7B prefill
/// prewarm_cache(&[
///     (1, 4096, 4096),    // Q projection (seq=1)
///     (1, 4096, 11008),   // FFN up (seq=1)
///     (512, 4096, 4096),  // Q projection (seq=512)
///     (512, 4096, 11008), // FFN up (seq=512)
/// ]);
/// ```
pub fn prewarm_cache(dims: &[(u32, u32, u32)]) {
    for &(m, k, n) in dims {
        cached_auto_select(m, k, n);
    }
}

/// Clear the spec cache. Useful for testing or after model change.
pub fn clear_cache() {
    let mut cache = get_cache().lock().unwrap();
    cache.clear();
}

/// Number of entries currently cached.
pub fn cache_size() -> usize {
    get_cache().lock().unwrap().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        clear_cache();

        // First call: compute + cache
        let cfg1 = cached_auto_select(64, 256, 64);
        let size_after_first = cache_size();

        // Second call: cache hit (same result)
        let cfg2 = cached_auto_select(64, 256, 64);
        let size_after_second = cache_size();

        // Cache should not grow on second call
        assert_eq!(size_after_first, size_after_second);

        // Configs should be equal (same dimensions → same selection)
        assert_eq!(cfg1.tile_m, cfg2.tile_m);
        assert_eq!(cfg1.tile_n, cfg2.tile_n);
        assert_eq!(cfg1.tile_k, cfg2.tile_k);
    }

    #[test]
    fn test_cache_different_dims() {
        clear_cache();

        cached_auto_select(33, 127, 33);
        let size1 = cache_size();
        cached_auto_select(37, 131, 37);
        let size2 = cache_size();
        cached_auto_select(33, 127, 33); // cache hit
        let size3 = cache_size();

        // First insert grows cache, second insert grows, third does not
        assert_eq!(size1, 1);
        assert_eq!(size2, 2);
        assert_eq!(size3, 2);
    }

    #[test]
    fn test_clear_cache() {
        clear_cache();
        cached_auto_select(64, 256, 64);
        assert!(cache_size() > 0);
        clear_cache();
        assert_eq!(cache_size(), 0);
    }

    #[test]
    fn test_prewarm() {
        clear_cache();
        prewarm_cache(&[
            (41, 137, 41),
            (43, 139, 43),
            (47, 149, 47),
        ]);
        // All 3 should be cached (unique keys)
        assert!(cache_size() >= 3);
    }

    #[test]
    fn test_backward_data_cached() {
        clear_cache();

        let cfg1 = cached_auto_select_backward_data(51, 157, 163);
        let size1 = cache_size();
        let cfg2 = cached_auto_select_backward_data(51, 157, 163);
        let size2 = cache_size();

        assert_eq!(cfg1.tile_m, cfg2.tile_m);
        // Cache hit: no new entry
        assert_eq!(size1, size2);
    }
}
