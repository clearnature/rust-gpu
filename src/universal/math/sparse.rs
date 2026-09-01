use crate::universal::core::{DType, GpuMemory, Kernel, Grid, Block};
use crate::universal::math::SparseLib;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// GPU 稀疏矩阵库
// ═══════════════════════════════════════════════════════

pub struct GpuSparseLib {
    device: Arc<dyn crate::universal::core::GpuDevice>,
}

impl GpuSparseLib {
    pub fn new(device: Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }
}

/// CSR (Compressed Sparse Row) 格式
pub struct CsrMatrix {
    pub values: GpuMemory,      // 非零值 [nnz]
    pub col_indices: GpuMemory, // 列索引 [nnz]
    pub row_offsets: GpuMemory, // 行偏移 [rows + 1]
    pub rows: u32,
    pub cols: u32,
    pub nnz: u32,               // 非零元素数
}

/// COO (Coordinate) 格式
pub struct CooMatrix {
    pub values: GpuMemory,      // 非零值 [nnz]
    pub row_indices: GpuMemory, // 行索引 [nnz]
    pub col_indices: GpuMemory, // 列索引 [nnz]
    pub rows: u32,
    pub cols: u32,
    pub nnz: u32,
}

impl SparseLib for GpuSparseLib {
    fn spmv_csr(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,        // [rows] f32
        matrix_values: &GpuMemory, // [nnz] f32
        col_indices: &GpuMemory,   // [nnz] u32
        row_offsets: &GpuMemory,   // [rows+1] u32
        input: &GpuMemory,         // [cols] f32
        rows: u32,
        _cols: u32,
        _nnz: u32,
    ) -> Result<(), String> {
        // SpMV: y = A @ x (CSR 格式)
        // 算法: 每个线程处理一行
        //   for j in row_offsets[row]..row_offsets[row+1]:
        //     y[row] += values[j] * x[col_indices[j]]
        //
        // CPU fallback (GPU 版需要 LDS + 归约)
        let host_values = matrix_values.host_ptr.ok_or("Values not CPU-mapped")? as *const f32;
        let host_cols = col_indices.host_ptr.ok_or("ColIndices not CPU-mapped")? as *const u32;
        let host_rows = row_offsets.host_ptr.ok_or("RowOffsets not CPU-mapped")? as *const u32;
        let host_x = input.host_ptr.ok_or("Input not CPU-mapped")? as *const f32;
        let host_y = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;

        let vals = unsafe { std::slice::from_raw_parts(host_values, _nnz as usize) };
        let cols = unsafe { std::slice::from_raw_parts(host_cols, _nnz as usize) };
        let offs = unsafe { std::slice::from_raw_parts(host_rows, (rows + 1) as usize) };
        let x = unsafe { std::slice::from_raw_parts(host_x, _cols as usize) };
        let y = unsafe { std::slice::from_raw_parts_mut(host_y, rows as usize) };

        for i in 0..rows as usize {
            let mut sum = 0.0f32;
            let start = offs[i] as usize;
            let end = offs[i + 1] as usize;
            for j in start..end {
                sum += vals[j] * x[cols[j] as usize];
            }
            y[i] = sum;
        }

        Ok(())
    }

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
    ) -> Result<(), String> {
        // SpMM: C = A @ B (CSR × 稠密)
        // 算法: 每个线程处理一行, 内循环遍历 B 的列
        //
        // CPU fallback
        let host_values = matrix_values.host_ptr.ok_or("Values not CPU-mapped")? as *const f32;
        let host_cols_idx = col_indices.host_ptr.ok_or("ColIndices not CPU-mapped")? as *const u32;
        let host_rows_off = row_offsets.host_ptr.ok_or("RowOffsets not CPU-mapped")? as *const u32;
        let host_b = input.host_ptr.ok_or("Input not CPU-mapped")? as *const f32;
        let host_c = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;

        let vals = unsafe { std::slice::from_raw_parts(host_values, nnz as usize) };
        let col_ids = unsafe { std::slice::from_raw_parts(host_cols_idx, nnz as usize) };
        let offs = unsafe { std::slice::from_raw_parts(host_rows_off, (rows + 1) as usize) };
        let b = unsafe { std::slice::from_raw_parts(host_b, (cols as usize * n_cols_b as usize)) };
        let c = unsafe { std::slice::from_raw_parts_mut(host_c, (rows as usize * n_cols_b as usize)) };

        // C[i, k] = sum_j A[i,j] * B[j,k]
        for i in 0..rows as usize {
            let start = offs[i] as usize;
            let end = offs[i + 1] as usize;
            for k in 0..n_cols_b as usize {
                let mut sum = 0.0f32;
                for j in start..end {
                    let col = col_ids[j] as usize;
                    sum += vals[j] * b[col * n_cols_b as usize + k];
                }
                c[i * n_cols_b as usize + k] = sum;
            }
        }

        Ok(())
    }
}
