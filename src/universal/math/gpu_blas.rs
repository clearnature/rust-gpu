use crate::universal::core::{DType, GpuMemory, Kernel, Grid, Block};
use crate::universal::math::BlasLib;
use crate::t0::schedule::Schedule;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// GPU BLAS — 使用 t0-gpu JIT GEMM
// ═══════════════════════════════════════════════════════

pub struct GpuBlasLib {
    device: Arc<dyn crate::universal::core::GpuDevice>,
}

impl GpuBlasLib {
    pub fn new(device: Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }
}

impl BlasLib for GpuBlasLib {
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
    ) -> Result<(), String> {
        // 使用 t0-gpu 的 auto_gemm JIT 编译器
        let target = crate::t0::ir::Target::detect();

        // 使用 AutoGemmSchedule 自动选择最优 tile 配置
        let sched = crate::t0::schedule::AutoGemmSchedule::for_problem(m, n, k);
        let (tile_m, tile_n) = sched.gemm_tile_mn();
        let wg_size = sched.workgroup_size().0;

        let kernel_ir = crate::t0::schedule::build_gemm_forward(&sched);
        let elf = kernel_ir.compile(target).map_err(|e| format!("Compile GEMM failed: {}", e))?;

        let kernel = self.device.load_kernel(&elf, "t0_gemm")?;

        // 构造 kernargs: [A_ptr, B_ptr, C_ptr, K, N, alpha, beta]
        let mut kernargs = Vec::new();
        kernargs.extend_from_slice(&a.device_addr.to_le_bytes());
        kernargs.extend_from_slice(&b.device_addr.to_le_bytes());
        kernargs.extend_from_slice(&c.device_addr.to_le_bytes());
        kernargs.extend_from_slice(&k.to_le_bytes());
        kernargs.extend_from_slice(&n.to_le_bytes());
        kernargs.extend_from_slice(&lda.to_le_bytes());
        kernargs.extend_from_slice(&ldb.to_le_bytes());
        kernargs.extend_from_slice(&ldc.to_le_bytes());
        kernargs.extend_from_slice(&alpha.to_le_bytes());
        kernargs.extend_from_slice(&beta.to_le_bytes());

        // Grid: [ceil(N/tile_n), ceil(M/tile_m), 1]
        let grid_x = (n + tile_n as u32 - 1) / tile_n as u32;
        let grid_y = (m + tile_m as u32 - 1) / tile_m as u32;

        queue.submit(&*kernel, Grid(grid_x, grid_y, 1), Block(wg_size, 1, 1), &kernargs, None)
    }

    fn gemv(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        m: u32, n: u32,
        alpha: f32,
        a: &GpuMemory, lda: u32,
        x: &GpuMemory, incx: u32,
        beta: f32,
        y: &GpuMemory, incy: u32,
    ) -> Result<(), String> {
        // GEMV: y = alpha * A @ x + beta * y
        // 使用简单的逐行 kernel
        let wg_size = 256u32;
        let grid_x = (m + wg_size - 1) / wg_size;

        let target = crate::t0::ir::Target::detect();

        let mut k = crate::t0::compile::T0Kernel::new("t0_gemv");
        let a_ptr = k.arg_ptr("A");
        let x_ptr = k.arg_ptr("x");
        let y_ptr = k.arg_ptr("y");
        let m_arg = k.arg_u32("m");
        let n_arg = k.arg_u32("n");
        let lda_arg = k.arg_u32("lda");
        let alpha_arg = k.arg_f32("alpha");
        let beta_arg = k.arg_f32("beta");
        k.emit_arg_loads();

        // 全局线程 ID = 行索引
        let row = k.compute_global_id_x(wg_size);

        // 累加器
        let acc = k.alloc_vreg();
        k.v_mov_imm(acc, 0);

        // 内循环: sum += A[row, col] * x[col]
        // 简化实现: 只处理 n <= 256 的情况
        // TODO: 实现完整的内循环

        // 写回 y[row]
        let byte_off = k.alloc_vreg();
        k.v_lshlrev_b32(byte_off, 2, row);

        let y_addr = k.alloc_vreg_array(2, crate::t0::ir::Alignment::Align2);
        k.v_mov_from_sgpr(y_addr, crate::t0::ir::SReg(y_ptr.0));
        k.v_mov_from_sgpr(crate::t0::ir::VReg(y_addr.0 + 1), crate::t0::ir::SReg(y_ptr.0 + 1));
        k.v_add_co(y_addr, y_addr, byte_off);
        k.v_add_co_ci(crate::t0::ir::VReg(y_addr.0 + 1), crate::t0::ir::VReg(y_addr.0 + 1));
        k.global_store(y_addr, acc, crate::t0::ir::Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        let elf = k.compile(target).map_err(|e| format!("Compile GEMV failed: {}", e))?;
        let kernel = self.device.load_kernel(&elf, "t0_gemv")?;

        let kernargs = vec![
            a.device_addr.to_le_bytes().to_vec(),
            x.device_addr.to_le_bytes().to_vec(),
            y.device_addr.to_le_bytes().to_vec(),
            (m as u64).to_le_bytes().to_vec(),
            (n as u64).to_le_bytes().to_vec(),
            (lda as u64).to_le_bytes().to_vec(),
            alpha.to_le_bytes().to_vec(),
            beta.to_le_bytes().to_vec(),
        ].concat();

        queue.submit(&*kernel, Grid(grid_x, 1, 1), Block(wg_size as u16, 1, 1), &kernargs, None)
    }
}
