use crate::universal::core::{DType, GpuMemory};

pub mod blas;
pub mod prim;
pub mod rng;
pub mod gpu_prim;
pub mod gpu_blas;
pub mod fft;
pub mod sparse;
pub mod swizzle;

pub use blas::T0BlasLib;
pub use prim::T0PrimLib;
pub use rng::T0RngLib;
pub use gpu_prim::GpuPrimLib;
pub use gpu_blas::GpuBlasLib;
pub use fft::{GpuFftLib, FftDirection};
pub use sparse::GpuSparseLib;

// ═══════════════════════════════════════════════════════
// 数学库 trait
// ═══════════════════════════════════════════════════════

pub trait BlasLib: Send + Sync {
    fn gemm(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        m: u32, n: u32, k: u32,
        alpha: f32,
        a: &GpuMemory, lda: u32,
        b: &GpuMemory, ldb: u32,
        beta: f32,
        c: &GpuMemory, ldc: u32,
        dtype: DType,
    ) -> Result<(), String>;

    fn gemv(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        m: u32, n: u32,
        alpha: f32,
        a: &GpuMemory, lda: u32,
        x: &GpuMemory, incx: u32,
        beta: f32,
        y: &GpuMemory, incy: u32,
    ) -> Result<(), String>;
}

pub trait PrimLib: Send + Sync {
    fn scan_exclusive(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        dtype: DType,
    ) -> Result<(), String>;

    fn reduce(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        op: ReduceOp,
        dtype: DType,
    ) -> Result<(), String>;

    fn radix_sort(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        keys: &GpuMemory,
        n: u32,
    ) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug)]
pub enum ReduceOp { Sum, Max, Min, Prod }

pub trait RngLib: Send + Sync {
    fn uniform(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        seed: u64,
    ) -> Result<(), String>;

    fn normal(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        seed: u64,
    ) -> Result<(), String>;

    fn bernoulli(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        p: f32,
        seed: u64,
    ) -> Result<(), String>;
}

/// FFT 库 trait
pub trait FftLib: Send + Sync {
    /// 1D FFT (复数交错格式: [re0, im0, re1, im1, ...])
    fn fft_1d(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        direction: fft::FftDirection,
    ) -> Result<(), String>;
}

/// 稀疏矩阵库 trait
pub trait SparseLib: Send + Sync {
    /// SpMV: y = A @ x (CSR 格式)
    fn spmv_csr(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,        // [rows] f32
        matrix_values: &GpuMemory, // [nnz] f32
        col_indices: &GpuMemory,   // [nnz] u32
        row_offsets: &GpuMemory,   // [rows+1] u32
        input: &GpuMemory,         // [cols] f32
        rows: u32,
        cols: u32,
        nnz: u32,
    ) -> Result<(), String>;

    /// SpMM: C = A @ B (CSR × 稠密)
    fn spmm_csr(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,        // [rows × n_cols_b] f32
        matrix_values: &GpuMemory, // [nnz] f32
        col_indices: &GpuMemory,   // [nnz] u32
        row_offsets: &GpuMemory,   // [rows+1] u32
        input: &GpuMemory,         // [cols × n_cols_b] f32
        rows: u32,
        cols: u32,
        nnz: u32,
        n_cols_b: u32,
    ) -> Result<(), String>;
}
