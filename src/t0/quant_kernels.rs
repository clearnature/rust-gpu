//! INT4 Quantization Kernels — GPTQ-style INT4 packed weights + dequantization
//!
//! # INT4 Packed Format
//!
//! Each byte stores 2 INT4 values (4 bits each):
//! ```text
//! byte = [hi_nibble | lo_nibble]
//!        element[2i+1]  element[2i]
//! ```
//!
//! INT4 range: 0..15 (unsigned), interpret as signed via (val - zero_point)
//!
//! # GPTQ Format
//!
//! Packed weights in INT4 format, with per-group scale and zero-point:
//! ```text
//! weight_packed: &[u8]   — (rows * cols / 2) bytes
//! scale: &[f32]          — (rows * n_groups) scales
//! zero_point: &[i32]     — (rows * n_groups) zero-points
//! ```
//!
//! # Kernels
//!
//! 1. `dequant_int4_f32` — Unpack INT4 → apply (val - zp) * scale → F32
//! 2. `dequant_int4_bf16` — Unpack INT4 → apply (val - zp) * scale → BF16 (via store_bf16)

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

// ============================================================================
// Kernel 1: INT4 Packed → F32 Dequantization
// ============================================================================

/// Build INT4 → F32 dequantization kernel.
///
/// Unpacks packed INT4 weights, applies per-group affine dequantization:
///   f32_val = (int4_val - zero_point) * scale
///
/// # Parameters
/// - `group_size_log2`: log2 of group_size (e.g., 7 for group_size=128).
///   Must be a compile-time constant since the DSL only supports constant shifts.
///
/// # Kernarg layout
/// `[packed_weights:u64, scales:u64, zero_points:u64, output:u64,
///   rows:u32, cols:u32]`
///
/// # Grid
/// `(ceil(total_elements / WG_SIZE) * WG_SIZE, 1, 1)`
///
/// Each thread processes 1 element (one nibble from packed byte array).
pub fn build_dequant_int4_f32(group_size_log2: u8) -> BlockKernel {
    let mut kb = BlockKernel::new("dequant_int4_f32", WG_SIZE);

    let packed_ptr = kb.arg_ptr("packed_weights");
    let scales_ptr = kb.arg_ptr("scales");
    let zp_ptr = kb.arg_ptr("zero_points");
    let out_ptr = kb.arg_ptr("output");
    let rows = kb.arg_u32("rows");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let elem_idx = wg_offset.add(&mut kb, tid);

    // Total elements = rows * cols
    let total = rows.mul(&mut kb, cols);
    let mask = elem_idx.lt(&mut kb, total);

    // byte_idx = elem_idx / 2 (2 elements per byte) — use SHR since no U32 div
    let byte_idx = elem_idx.shr(&mut kb, 1);

    // Load packed byte (as u32, mask to 8 bits)
    let packed_byte_u32 = kb.load_u32(packed_ptr, byte_idx, mask);
    let packed_byte = packed_byte_u32.bitand(&mut kb, 0xFF);

    // Extract nibble: even indices use low nibble, odd use high nibble
    // is_odd = elem_idx & 1 → 0 or 1
    let is_odd = elem_idx.bitand(&mut kb, 1);
    let lo_nibble = packed_byte.bitand(&mut kb, 0x0F);
    let hi_nibble = packed_byte.shr(&mut kb, 4).bitand(&mut kb, 0x0F);

    // Select: is_odd < 1 means is_odd == 0, so use lo_nibble (true_val)
    //         is_odd >= 1 means is_odd == 1, so use hi_nibble (false_val)
    let one = kb.const_u32(1);
    let is_even_mask = is_odd.lt(&mut kb, one);  // true when is_odd == 0
    let int4_val = is_even_mask.select(&mut kb, lo_nibble, hi_nibble);

    // Group index for scale/zp lookup (constant shift for power-of-2 group size)
    let group_idx = elem_idx.shr(&mut kb, group_size_log2);

    // Apply dequantization: (val - zero_point) * scale
    let zero_point = kb.load_u32(zp_ptr, group_idx, mask);
    let signed_val = int4_val.sub(&mut kb, zero_point);
    let signed_f32 = signed_val.to_f32(&mut kb);
    let scale = kb.load(scales_ptr, group_idx, mask);
    let dequant_f32 = signed_f32.mul(&mut kb, scale);

    // Store as F32
    kb.store(out_ptr, elem_idx, dequant_f32, mask);

    kb
}

// ============================================================================
// Kernel 2: INT4 Packed → BF16 Dequantization
// ============================================================================

/// Build INT4 → BF16 dequantization kernel.
///
/// Same dequantization as `build_dequant_int4_f32` but stores result as BF16.
/// Uses `StoreBf16` which truncates F32 → BF16 during the store operation.
///
/// # Parameters
/// - `group_size_log2`: log2 of group_size (e.g., 7 for group_size=128).
///   Must be a compile-time constant since the DSL only supports constant shifts.
///
/// # Kernarg layout
/// `[packed_weights:u64, scales:u64, zero_points:u64, output:u64,
///   rows:u32, cols:u32]`
///
/// # Grid
/// `(ceil(total_elements / WG_SIZE) * WG_SIZE, 1, 1)`
pub fn build_dequant_int4_bf16(group_size_log2: u8) -> BlockKernel {
    let mut kb = BlockKernel::new("dequant_int4_bf16", WG_SIZE);

    let packed_ptr = kb.arg_ptr("packed_weights");
    let scales_ptr = kb.arg_ptr("scales");
    let zp_ptr = kb.arg_ptr("zero_points");
    let out_ptr = kb.arg_ptr("output");
    let rows = kb.arg_u32("rows");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let elem_idx = wg_offset.add(&mut kb, tid);

    let total = rows.mul(&mut kb, cols);
    let mask = elem_idx.lt(&mut kb, total);

    let byte_idx = elem_idx.shr(&mut kb, 1);

    let packed_byte_u32 = kb.load_u32(packed_ptr, byte_idx, mask);
    let packed_byte = packed_byte_u32.bitand(&mut kb, 0xFF);

    let is_odd = elem_idx.bitand(&mut kb, 1);
    let lo_nibble = packed_byte.bitand(&mut kb, 0x0F);
    let hi_nibble = packed_byte.shr(&mut kb, 4).bitand(&mut kb, 0x0F);

    let one = kb.const_u32(1);
    let is_even_mask = is_odd.lt(&mut kb, one);
    let int4_val = is_even_mask.select(&mut kb, lo_nibble, hi_nibble);

    let group_idx = elem_idx.shr(&mut kb, group_size_log2);

    let zero_point = kb.load_u32(zp_ptr, group_idx, mask);
    let signed_val = int4_val.sub(&mut kb, zero_point);
    let signed_f32 = signed_val.to_f32(&mut kb);
    let scale = kb.load(scales_ptr, group_idx, mask);
    let dequant_f32 = signed_f32.mul(&mut kb, scale);

    // Store as BF16 (StoreBf16 truncates F32 → BF16)
    kb.store_bf16(out_ptr, elem_idx, dequant_f32, mask);

    kb
}

// ============================================================================
// GPTQ Weight Loader
// ============================================================================

/// Metadata for a GPTQ-quantized weight tensor.
///
/// GPTQ format stores weights as INT4 packed, with per-group scale and
/// zero-point for affine dequantization:
///
/// ```text
/// dequant(w) = (int4_value - zero_point) * scale
/// ```
///
/// Group size typically 128 or 256 elements.
#[derive(Clone, Debug)]
pub struct GptqWeight {
    /// Rows in the weight matrix (output features).
    pub rows: u32,
    /// Columns in the weight matrix (input features).
    pub cols: u32,
    /// Group size for quantization (e.g., 128).
    pub group_size: u32,
    /// Number of groups per row = cols / group_size.
    pub n_groups: u32,
    /// Packed INT4 data: (rows * cols / 2) bytes.
    /// Each byte: lo_nibble = element[2i], hi_nibble = element[2i+1].
    pub packed_weights: Vec<u8>,
    /// Scale factors: [n_groups × rows] f32 values.
    /// Layout: scale[row_idx * n_groups + group_idx]
    pub scales: Vec<f32>,
    /// Zero points: [n_groups × rows] i32 values.
    /// Layout: zero_point[row_idx * n_groups + group_idx]
    pub zero_points: Vec<i32>,
}

impl GptqWeight {
    /// Create a new GPTQ weight from raw components.
    pub fn new(
        rows: u32,
        cols: u32,
        group_size: u32,
        packed_weights: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    ) -> Self {
        let n_groups = cols / group_size;
        let expected_packed = (rows as usize) * (cols as usize) / 2;
        let expected_params = (n_groups as usize) * (rows as usize);

        assert_eq!(
            packed_weights.len(),
            expected_packed,
            "packed_weights length mismatch: expected {} bytes for {}×{} matrix",
            expected_packed, rows, cols
        );
        assert_eq!(
            scales.len(),
            expected_params,
            "scales length mismatch: expected {} for {} groups × {} rows",
            expected_params, n_groups, rows
        );
        assert_eq!(
            zero_points.len(),
            expected_params,
            "zero_points length mismatch"
        );
        Self { rows, cols, group_size, n_groups, packed_weights, scales, zero_points }
    }

    /// Load GPTQ weights from raw bytes (standard format).
    ///
    /// The `raw` format is:
    /// ```text
    /// [4 bytes: rows (LE)] [4 bytes: cols (LE)] [4 bytes: group_size (LE)]
    /// [rows * cols / 2 bytes: packed INT4 weights]
    /// [n_groups * rows * 4 bytes: scales (f32 LE)]
    /// [n_groups * rows * 4 bytes: zero_points (i32 LE)]
    /// ```
    pub fn from_bytes(raw: &[u8]) -> Self {
        assert!(raw.len() >= 12, "GPTQ data too short (need at least 12 bytes header)");

        let rows = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let cols = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let group_size = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let n_groups = cols / group_size;
        let n_params = (n_groups * rows) as usize;

        let packed_len = (rows as usize) * (cols as usize) / 2;
        let expected_total = 12 + packed_len + n_params * 8;

        assert!(
            raw.len() >= expected_total,
            "GPTQ data truncated: need {} bytes, got {}",
            expected_total, raw.len()
        );

        let packed_weights = raw[12..12 + packed_len].to_vec();

        let scales_start = 12 + packed_len;
        let scales_bytes = &raw[scales_start..scales_start + n_params * 4];
        let scales: Vec<f32> = scales_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let zp_start = scales_start + n_params * 4;
        let zp_bytes = &raw[zp_start..zp_start + n_params * 4];
        let zero_points: Vec<i32> = zp_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Self::new(rows, cols, group_size, packed_weights, scales, zero_points)
    }

    /// Get byte size needed for GPU buffer allocation.
    pub fn gpu_buffer_size(&self) -> usize {
        self.packed_weights.len()
            + self.scales.len() * 4
            + self.zero_points.len() * 4
    }
}

// ============================================================================
// INT4 WMMA Reference (for gemm_gen integration)
// ============================================================================

/// INT4 WMMA tile specification for integration with `gemm_gen`.
///
/// This describes how INT4-packed weights should be loaded and fed
/// into `v_wmma_i32_16x16x16_iu4` instructions. The actual WMMA
/// code generation uses `compile::T0Kernel::wmma_iu4()` directly.
///
/// # INT4 WMMA Register Layout
///
/// For `v_wmma_i32_16x16x16_iu4`:
/// - A: 1 VGPR (16 lanes × 4 bits = 64 bits = 1 VGPR)
/// - B: 1 VGPR (same)
/// - C/D: 8 VGPRs (16×16 i32 accumulator)
///
/// Each VGPR for A/B packs 16 INT4 values (4 bits each) into 64 bits.
/// The hardware reads these directly without unpacking.
#[derive(Clone, Debug)]
pub struct Int4WmmaConfig {
    /// Tile M dimension (must be 16 for IU4 K=16 WMMA).
    pub tile_m: u32,
    /// Tile N dimension (must be 16 for IU4 K=16 WMMA).
    pub tile_n: u32,
    /// Tile K dimension (16 for K=16 variant, 32 for K=32 variant).
    pub tile_k: u32,
}

impl Int4WmmaConfig {
    /// Default INT4 WMMA config: 16×16×16 tiles.
    pub fn k16() -> Self {
        Self { tile_m: 16, tile_n: 16, tile_k: 16 }
    }

    /// INT4 WMMA config with K=32 (uses `v_wmma_i32_16x16x32_iu4`).
    /// Higher throughput but needs 2 VGPRs per A/B operand.
    pub fn k32() -> Self {
        Self { tile_m: 16, tile_n: 16, tile_k: 32 }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_nibbles_basic() {
        // 0xAB → lo=0x0B=11, hi=0x0A=10
        let packed: u8 = 0xAB;
        let lo = packed & 0x0F;
        let hi = (packed >> 4) & 0x0F;
        assert_eq!(lo, 11);
        assert_eq!(hi, 10);
    }

    #[test]
    fn test_int4_signed_conversion() {
        // INT4 unsigned 0..15 → signed -8..7 via (val - 8)
        assert_eq!((0i32).wrapping_sub(8), -8);
        assert_eq!((8i32).wrapping_sub(8), 0);
        assert_eq!((15i32).wrapping_sub(8), 7);
    }

    #[test]
    fn test_dequant_formula() {
        // Verify dequantization formula: (val - zp) * scale
        let val: i32 = 12;
        let zp: i32 = 8;
        let scale: f32 = 0.5;
        let result = (val - zp) as f32 * scale;
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_gptq_weight_packing_roundtrip() {
        // Pack two INT4 values into one byte
        let lo: u8 = 3;
        let hi: u8 = 7;
        let packed = lo | (hi << 4);
        assert_eq!(packed & 0x0F, 3);
        assert_eq!((packed >> 4) & 0x0F, 7);
    }

    #[test]
    fn test_gptq_weight_from_bytes() {
        // Construct minimal GPTQ data: 16×16 matrix, group_size=16
        let rows: u32 = 16;
        let cols: u32 = 16;
        let group_size: u32 = 16;
        let n_groups = cols / group_size; // 1
        let n_params = (n_groups * rows) as usize; // 16

        let mut data = Vec::new();
        data.extend_from_slice(&rows.to_le_bytes());
        data.extend_from_slice(&cols.to_le_bytes());
        data.extend_from_slice(&group_size.to_le_bytes());

        // Packed weights: 16*16/2 = 128 bytes
        let packed_len = (rows as usize) * (cols as usize) / 2;
        data.extend(std::iter::repeat(0xABu8).take(packed_len));

        // Scales: n_params f32 values
        for _ in 0..n_params {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }

        // Zero points: n_params i32 values
        for _ in 0..n_params {
            data.extend_from_slice(&8i32.to_le_bytes());
        }

        let gptq = GptqWeight::from_bytes(&data);
        assert_eq!(gptq.rows, 16);
        assert_eq!(gptq.cols, 16);
        assert_eq!(gptq.group_size, 16);
        assert_eq!(gptq.n_groups, 1);
        assert_eq!(gptq.packed_weights.len(), packed_len);
        assert_eq!(gptq.scales.len(), n_params);
        assert_eq!(gptq.zero_points.len(), n_params);
        assert!(gptq.scales.iter().all(|&s| s == 1.0));
        assert!(gptq.zero_points.iter().all(|&z| z == 8));
    }

    #[test]
    fn test_dequant_int4_f32_kernel_compiles() {
        // group_size=128 → group_size_log2=7
        let kernel = build_dequant_int4_f32(7);
        let result = kernel.compile(Target::GFX1200);
        assert!(result.is_ok(), "dequant_int4_f32 kernel failed to compile: {:?}", result.err());
    }

    #[test]
    fn test_dequant_int4_bf16_kernel_compiles() {
        // group_size=128 → group_size_log2=7
        let kernel = build_dequant_int4_bf16(7);
        let result = kernel.compile(Target::GFX1200);
        assert!(result.is_ok(), "dequant_int4_bf16 kernel failed to compile: {:?}", result.err());
    }

    #[test]
    fn test_nibble_extraction_pattern() {
        // Verify the standard INT4 packing pattern used by GPTQ
        // Byte 0xD7: lo=7, hi=13
        let byte: u8 = 0xD7;
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;
        assert_eq!(lo, 7);
        assert_eq!(hi, 13);

        // Signed conversion
        let lo_signed = lo as i32 - 8;
        let hi_signed = hi as i32 - 8;
        assert_eq!(lo_signed, -1);
        assert_eq!(hi_signed, 5);
    }

    #[test]
    fn test_group_index_calculation() {
        // Element index 200, group_size 128 → group_idx = 1
        let elem_idx: u32 = 200;
        let group_size: u32 = 128;
        let group_idx = elem_idx / group_size;
        assert_eq!(group_idx, 1);

        // Element index 127, group_size 128 → group_idx = 0
        let elem_idx2: u32 = 127;
        let group_idx2 = elem_idx2 / group_size;
        assert_eq!(group_idx2, 0);
    }

    #[test]
    fn test_gptq_gpu_buffer_size() {
        let gptq = GptqWeight::new(
            16, 16, 16,
            vec![0u8; 128],  // 16*16/2
            vec![1.0f32; 16], // 1 group * 16 rows
            vec![8i32; 16],
        );
        // 128 + 16*4 + 16*4 = 128 + 64 + 64 = 256
        assert_eq!(gptq.gpu_buffer_size(), 256);
    }

    #[test]
    fn test_int4_wmma_config() {
        let cfg16 = Int4WmmaConfig::k16();
        assert_eq!(cfg16.tile_k, 16);

        let cfg32 = Int4WmmaConfig::k32();
        assert_eq!(cfg32.tile_k, 32);
    }
}
