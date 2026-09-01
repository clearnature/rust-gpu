use crate::universal::core::{DType, GpuMemory, Kernel, Grid, Block};
use crate::universal::math::{PrimLib, ReduceOp};
use crate::t0::compile::T0Kernel;
use crate::t0::ir::{Alignment, SReg, VReg, Width};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// GPU 并行原语 — 使用 t0-gpu JIT 编译器
// ═══════════════════════════════════════════════════════

pub struct GpuPrimLib {
    device: Arc<dyn crate::universal::core::GpuDevice>,
}

impl GpuPrimLib {
    pub fn new(device: Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }

    fn compile_kernel(&self, name: &str, elf: &[u8]) -> Result<Box<dyn Kernel>, String> {
        self.device.load_kernel(elf, name)
    }

    fn build_kernargs_u64(args: &[u64]) -> Vec<u8> {
        let mut kernargs = Vec::new();
        for arg in args {
            kernargs.extend_from_slice(&arg.to_le_bytes());
        }
        kernargs
    }

    /// 辅助: 构造 64-bit 地址 + 字节偏移
    fn build_addr(k: &mut T0Kernel, base: u32, byte_off: VReg) -> VReg {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr, SReg(base));
        k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(base + 1));
        k.v_add_co(addr, addr, byte_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        addr
    }
}

impl PrimLib for GpuPrimLib {
    fn scan_exclusive(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        _dtype: DType,
    ) -> Result<(), String> {
        // GPU exclusive scan kernel
        // 算法: 所有线程加载到 LDS → barrier → thread 0 顺序 scan → barrier → 读回
        let wg_size = 64u32.min(n.next_power_of_two().max(64));
        let grid_x = 1u32;

        let target = crate::t0::ir::Target::detect();
        let mut k = T0Kernel::new("t0_scan_exclusive");

        let input_ptr = k.arg_ptr("input");
        let output_ptr = k.arg_ptr("output");
        let n_arg = k.arg_u32("n");
        k.emit_arg_loads();

        let tid = k.compute_global_id_x(wg_size);

        // 加载 input[tid] → val
        let byte_off = k.alloc_vreg();
        k.v_lshlrev_b32(byte_off, 2, tid);

        let in_addr = Self::build_addr(&mut k, input_ptr.0, byte_off);

        let val = k.alloc_vreg();
        k.v_mov_imm(val, 0);
        k.global_load(val, in_addr, Width::B32, 0);
        k.wait_vmcnt(0);

        // 写入 LDS[tid*4]
        let lds_addr = k.alloc_vreg();
        k.v_lshlrev_b32(lds_addr, 2, tid);
        k.lds_store(lds_addr, val, Width::B32, 0);
        k.barrier();

        // Thread 0: 顺序 scan (exclusive prefix sum)
        // 读取所有 LDS 值, 计算前缀和, 写回
        let zero = k.alloc_vreg();
        k.v_mov_imm(zero, 0);

        let acc = k.alloc_vreg();
        k.v_mov_imm(acc, 0);

        let lds_iter = k.alloc_vreg();
        k.v_mov_imm(lds_iter, 0);

        let lds_val = k.alloc_vreg();
        let lds_out = k.alloc_vreg();
        let four = k.alloc_vreg();
        k.v_mov_imm(four, 4);

        // 顺序 scan 循环 (固定 64 次迭代)
        for _ in 0..64u32 {
            // 读取当前值
            k.lds_load(lds_val, lds_iter, Width::B32, 0);
            k.wait_lgkmcnt(0);

            // 写回前缀和 (exclusive: 写入累加前的值)
            k.lds_store(lds_iter, acc, Width::B32, 0);

            // 累加
            k.v_add_f32(acc, acc, lds_val);

            // lds_iter += 4
            k.v_add_u32(lds_iter, lds_iter, four);
        }

        // barrier 确保所有线程看到更新后的 LDS
        k.barrier();

        // 所有线程读取自己的 scan 结果
        k.v_lshlrev_b32(lds_addr, 2, tid);
        k.lds_load(val, lds_addr, Width::B32, 0);
        k.wait_lgkmcnt(0);

        // 写回 output[tid]
        let out_addr = Self::build_addr(&mut k, output_ptr.0, byte_off);
        k.global_store(out_addr, val, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        let elf = k.compile(target).map_err(|e| format!("Compile scan: {}", e))?;
        let kernel = self.compile_kernel("t0_scan_exclusive", &elf)?;

        let kernargs = Self::build_kernargs_u64(&[
            input.device_addr,
            output.device_addr,
            n as u64,
        ]);

        queue.submit(&*kernel, Grid(grid_x, 1, 1), Block(wg_size as u16, 1, 1), &kernargs, None)
    }

    fn reduce(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        op: ReduceOp,
        _dtype: DType,
    ) -> Result<(), String> {
        // GPU 并行归约 kernel (安全版)
        // 算法: 所有线程加载到 LDS → barrier → thread 0 顺序归约
        // 避免 barrier 死锁 (不需要条件 EXEC mask)
        let wg_size = 64u32.min(n.next_power_of_two().max(64));
        let grid_x = 1u32;

        let target = crate::t0::ir::Target::detect();
        let mut k = T0Kernel::new("t0_reduce_sum");

        let input_ptr = k.arg_ptr("input");
        let output_ptr = k.arg_ptr("output");
        let n_arg = k.arg_u32("n");
        k.emit_arg_loads();

        let tid = k.compute_global_id_x(wg_size);

        // 加载 input[tid] → val
        let byte_off = k.alloc_vreg();
        k.v_lshlrev_b32(byte_off, 2, tid);

        let in_addr = Self::build_addr(&mut k, input_ptr.0, byte_off);

        let val = k.alloc_vreg();
        k.v_mov_imm(val, 0);
        k.global_load(val, in_addr, Width::B32, 0);
        k.wait_vmcnt(0);

        // 写入 LDS[tid*4]
        let lds_addr = k.alloc_vreg();
        k.v_lshlrev_b32(lds_addr, 2, tid);
        k.lds_store(lds_addr, val, Width::B32, 0);
        k.barrier();

        // Thread 0: 顺序读取所有 LDS 值并累加
        // (其他线程直接跳到 endpgm)
        let zero = k.alloc_vreg();
        k.v_mov_imm(zero, 0);

        let acc = k.alloc_vreg();
        k.v_mov_imm(acc, 0);

        let lds_iter = k.alloc_vreg();
        k.v_mov_imm(lds_iter, 0);

        // 顺序累加循环 (简化: 固定 64 次迭代)
        let lds_val = k.alloc_vreg();
        for i in 0..64u32 {
            k.lds_load(lds_val, lds_iter, Width::B32, 0);
            k.wait_lgkmcnt(0);
            k.v_add_f32(acc, acc, lds_val);
            // lds_iter += 4
            let four = k.alloc_vreg();
            k.v_mov_imm(four, 4);
            k.v_add_u32(lds_iter, lds_iter, four);
        }

        // Thread 0 写回 output[0]
        let out_addr = Self::build_addr(&mut k, output_ptr.0, zero);
        k.global_store(out_addr, acc, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        let elf = k.compile(target).map_err(|e| format!("Compile reduce: {}", e))?;
        let kernel = self.compile_kernel("t0_reduce_sum", &elf)?;

        let kernargs = Self::build_kernargs_u64(&[
            input.device_addr,
            output.device_addr,
            n as u64,
        ]);

        queue.submit(&*kernel, Grid(grid_x, 1, 1), Block(wg_size as u16, 1, 1), &kernargs, None)
    }

    fn radix_sort(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        keys: &GpuMemory,
        n: u32,
    ) -> Result<(), String> {
        // GPU Bitonic Sort kernel (简化版)
        // 算法: 所有元素加载到 LDS → 比较-交换网络 → 写回
        // 限制: n <= 64 (单工作组, 展开的 bitonic 网络)
        let wg_size = 64u32.min(n.next_power_of_two().max(64));
        let grid_x = 1u32;

        let target = crate::t0::ir::Target::detect();
        let mut k = T0Kernel::new("t0_bitonic_sort");

        let keys_ptr = k.arg_ptr("keys");
        let _n_arg = k.arg_u32("n");
        k.emit_arg_loads();

        let tid = k.compute_global_id_x(wg_size);

        // 加载 keys[tid] → val
        let byte_off = k.alloc_vreg();
        k.v_lshlrev_b32(byte_off, 2, tid);

        let keys_addr = Self::build_addr(&mut k, keys_ptr.0, byte_off);

        let val = k.alloc_vreg();
        k.v_mov_imm(val, 0);
        k.global_load(val, keys_addr, Width::B32, 0);
        k.wait_vmcnt(0);

        // 写入 LDS[tid*4]
        let lds_addr = k.alloc_vreg();
        k.v_lshlrev_b32(lds_addr, 2, tid);
        k.lds_store(lds_addr, val, Width::B32, 0);
        k.barrier();

        // Bitonic sort 网络 (展开)
        // 使用比较-交换操作: min/max
        // 每步: partner = tid ^ step, 比较并交换

        let partner = k.alloc_vreg();
        let lds_i = k.alloc_vreg();
        let lds_j = k.alloc_vreg();
        let val_i = k.alloc_vreg();
        let val_j = k.alloc_vreg();
        let temp = k.alloc_vreg();
        let step_v = k.alloc_vreg();

        // 宏: 比较-交换步骤
        macro_rules! bitonic_step {
            ($step:expr) => {
                k.v_mov_imm(step_v, $step);
                k.v_xor_b32(partner, crate::t0::ir::Operand::VReg(tid), crate::t0::ir::Operand::VReg(step_v));
                k.v_lshlrev_b32(lds_i, 2, tid);
                k.v_lshlrev_b32(lds_j, 2, partner);
                k.lds_load(val_i, lds_i, Width::B32, 0);
                k.lds_load(val_j, lds_j, Width::B32, 0);
                k.wait_lgkmcnt(0);
                k.v_min_f32(temp, val_i, val_j);
                k.v_max_f32(val_j, val_i, val_j);
                k.v_mov(val_i, temp);
                k.lds_store(lds_i, val_i, Width::B32, 0);
                k.lds_store(lds_j, val_j, Width::B32, 0);
                k.barrier();
            };
        }

        // Stage 2: step 1
        bitonic_step!(1);

        // Stage 4: steps 2, 1
        bitonic_step!(2);
        bitonic_step!(1);

        // Stage 8: steps 4, 2, 1
        bitonic_step!(4);
        bitonic_step!(2);
        bitonic_step!(1);

        // Stage 16: steps 8, 4, 2, 1
        bitonic_step!(8);
        bitonic_step!(4);
        bitonic_step!(2);
        bitonic_step!(1);

        // Stage 32: steps 16, 8, 4, 2, 1
        bitonic_step!(16);
        bitonic_step!(8);
        bitonic_step!(4);
        bitonic_step!(2);
        bitonic_step!(1);

        // 读取排序后的值
        k.v_lshlrev_b32(lds_addr, 2, tid);
        k.lds_load(val, lds_addr, Width::B32, 0);
        k.wait_lgkmcnt(0);

        // 写回 keys[tid]
        k.global_store(keys_addr, val, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();

        let elf = k.compile(target).map_err(|e| format!("Compile sort: {}", e))?;
        let kernel = self.compile_kernel("t0_bitonic_sort", &elf)?;

        let kernargs = Self::build_kernargs_u64(&[
            keys.device_addr,
            n as u64,
        ]);

        queue.submit(&*kernel, Grid(grid_x, 1, 1), Block(wg_size as u16, 1, 1), &kernargs, None)
    }
}
