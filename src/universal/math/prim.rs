use crate::universal::core::{DType, GpuMemory, Kernel};
use crate::universal::math::{PrimLib, ReduceOp};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// T0PrimLib — 使用 t0-gpu JIT 编译器实现并行原语
// ═══════════════════════════════════════════════════════

pub struct T0PrimLib {
    device: Arc<dyn crate::universal::core::GpuDevice>,
}

impl T0PrimLib {
    pub fn new(device: Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }
}

impl PrimLib for T0PrimLib {
    fn scan_exclusive(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        dtype: DType,
    ) -> Result<(), String> {
        // Blelloch exclusive scan 算法
        // 1. 每个工作组处理一个 block
        // 2. 工作组内做 Blelloch up-sweep + down-sweep
        // 3. 跨工作组做 block prefix sum

        // TODO: 编译 scan kernel 并 dispatch
        // 暂时用 CPU 实现作为 fallback
        let size = n as usize * dtype.size_bytes();
        let mut host_input = vec![0u8; size];
        let mut host_output = vec![0u8; size];

        // 读取输入
        let ptr = input.host_ptr.ok_or("Input not CPU-mapped")? as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, host_input.as_mut_ptr(), size); }

        // CPU exclusive scan
        match dtype {
            DType::F32 => {
                let inp: &[f32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const f32, n as usize) };
                let out: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(host_output.as_mut_ptr() as *mut f32, n as usize) };
                let mut acc = 0.0f32;
                for i in 0..n as usize {
                    out[i] = acc;
                    acc += inp[i];
                }
            }
            DType::U32 => {
                let inp: &[u32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const u32, n as usize) };
                let out: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(host_output.as_mut_ptr() as *mut u32, n as usize) };
                let mut acc = 0u32;
                for i in 0..n as usize {
                    out[i] = acc;
                    acc = acc.wrapping_add(inp[i]);
                }
            }
            _ => return Err(format!("Unsupported dtype for scan: {:?}", dtype)),
        }

        // 写回输出
        let out_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(host_output.as_ptr(), out_ptr, size); }

        Ok(())
    }

    fn reduce(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        op: ReduceOp,
        dtype: DType,
    ) -> Result<(), String> {
        // GPU 并行归约
        // 1. 每个工作组做树形归约
        // 2. 跨工作组做最终归约

        // TODO: 编译 reduce kernel 并 dispatch
        // 暂时用 CPU 实现作为 fallback
        let size = n as usize * dtype.size_bytes();
        let mut host_input = vec![0u8; size];

        let ptr = input.host_ptr.ok_or("Input not CPU-mapped")? as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, host_input.as_mut_ptr(), size); }

        match (dtype, op) {
            (DType::F32, ReduceOp::Sum) => {
                let inp: &[f32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const f32, n as usize) };
                let sum: f32 = inp.iter().sum();
                let out_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;
                unsafe { std::ptr::write_volatile(out_ptr, sum); }
            }
            (DType::F32, ReduceOp::Max) => {
                let inp: &[f32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const f32, n as usize) };
                let max = inp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let out_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;
                unsafe { std::ptr::write_volatile(out_ptr, max); }
            }
            (DType::F32, ReduceOp::Min) => {
                let inp: &[f32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const f32, n as usize) };
                let min = inp.iter().cloned().fold(f32::INFINITY, f32::min);
                let out_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;
                unsafe { std::ptr::write_volatile(out_ptr, min); }
            }
            (DType::U32, ReduceOp::Sum) => {
                let inp: &[u32] = unsafe { std::slice::from_raw_parts(host_input.as_ptr() as *const u32, n as usize) };
                let sum: u32 = inp.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
                let out_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut u32;
                unsafe { std::ptr::write_volatile(out_ptr, sum); }
            }
            _ => return Err(format!("Unsupported dtype/op for reduce: {:?}/{:?}", dtype, op)),
        }

        Ok(())
    }

    fn radix_sort(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        keys: &GpuMemory,
        n: u32,
    ) -> Result<(), String> {
        let size = n as usize * 4;
        let host_ptr = keys.host_ptr.ok_or("Keys not CPU-mapped")? as *mut u8;

        // 读取
        let slice = unsafe { std::slice::from_raw_parts_mut(host_ptr as *mut u32, n as usize) };
        slice.sort();

        Ok(())
    }
}
