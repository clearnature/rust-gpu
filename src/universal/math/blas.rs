use crate::universal::core::{DType, GpuMemory};
use crate::universal::math::BlasLib;

// ═══════════════════════════════════════════════════════
// T0BlasLib — BLAS 操作 (使用 t0-gpu JIT GEMM)
// ═══════════════════════════════════════════════════════

pub struct T0BlasLib {
    device: std::sync::Arc<dyn crate::universal::core::GpuDevice>,
}

impl T0BlasLib {
    pub fn new(device: std::sync::Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }
}

impl BlasLib for T0BlasLib {
    fn gemm(
        &self,
        _queue: &mut dyn crate::universal::core::ComputeQueue,
        m: u32, n: u32, k: u32,
        alpha: f32,
        a: &GpuMemory, lda: u32,
        b: &GpuMemory, ldb: u32,
        beta: f32,
        c: &GpuMemory, ldc: u32,
        dtype: DType,
    ) -> Result<(), String> {
        // TODO: 使用 t0-gpu 的 GEMM JIT 编译器
        // 参考: t0/auto_gemm.rs
        //
        // 1. cost_model::best_gemm_config(m, n, k) → 最优 tile 配置
        // 2. gemm_gen::build_kernel(config) → T0Kernel
        // 3. kernel.compile() → ELF
        // 4. device.load_kernel(elf) → GpuKernel
        // 5. queue.submit(kernel, grid, block, kernargs)

        // 暂时用 CPU 实现作为 fallback
        if dtype != DType::F32 {
            return Err("Only F32 GEMM supported in CPU fallback".into());
        }

        let a_ptr = a.host_ptr.ok_or("A not CPU-mapped")? as *const f32;
        let b_ptr = b.host_ptr.ok_or("B not CPU-mapped")? as *const f32;
        let c_ptr = c.host_ptr.ok_or("C not CPU-mapped")? as *mut f32;

        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, (m * k) as usize) };
        let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, (k * n) as usize) };
        let c_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr, (m * n) as usize) };

        // C = alpha * A @ B + beta * C
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;
                for p in 0..k as usize {
                    sum += a_slice[i * lda as usize + p] * b_slice[p * ldb as usize + j];
                }
                c_slice[i * ldc as usize + j] = alpha * sum + beta * c_slice[i * ldc as usize + j];
            }
        }

        Ok(())
    }

    fn gemv(
        &self,
        _queue: &mut dyn crate::universal::core::ComputeQueue,
        m: u32, n: u32,
        alpha: f32,
        a: &GpuMemory, lda: u32,
        x: &GpuMemory, incx: u32,
        beta: f32,
        y: &GpuMemory, incy: u32,
    ) -> Result<(), String> {
        // y = alpha * A @ x + beta * y
        let a_ptr = a.host_ptr.ok_or("A not CPU-mapped")? as *const f32;
        let x_ptr = x.host_ptr.ok_or("x not CPU-mapped")? as *const f32;
        let y_ptr = y.host_ptr.ok_or("y not CPU-mapped")? as *mut f32;

        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, (m * n) as usize) };
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, n as usize) };
        let y_slice = unsafe { std::slice::from_raw_parts_mut(y_ptr, m as usize) };

        for i in 0..m as usize {
            let mut sum = 0.0f32;
            for j in 0..n as usize {
                sum += a_slice[i * lda as usize + j] * x_slice[j * incx as usize];
            }
            y_slice[i * incy as usize] = alpha * sum + beta * y_slice[i * incy as usize];
        }

        Ok(())
    }
}
