use crate::universal::core::{DType, GpuMemory, Kernel, Grid, Block};
use crate::universal::math::FftLib;
use crate::t0::compile::T0Kernel;
use crate::t0::ir::{Alignment, SReg, VReg, Width};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// GPU FFT — 使用 t0-gpu JIT 编译器
// ═══════════════════════════════════════════════════════

/// FFT 方向
#[derive(Clone, Copy, Debug)]
pub enum FftDirection {
    Forward,
    Inverse,
}

pub struct GpuFftLib {
    device: Arc<dyn crate::universal::core::GpuDevice>,
}

impl GpuFftLib {
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

    fn build_addr(k: &mut T0Kernel, base: u32, byte_off: VReg) -> VReg {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_from_sgpr(addr, SReg(base));
        k.v_mov_from_sgpr(VReg(addr.0 + 1), SReg(base + 1));
        k.v_add_co(addr, addr, byte_off);
        k.v_add_co_ci(VReg(addr.0 + 1), VReg(addr.0 + 1));
        addr
    }
}

impl FftLib for GpuFftLib {
    fn fft_1d(
        &self,
        queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        input: &GpuMemory,
        n: u32,
        direction: FftDirection,
    ) -> Result<(), String> {
        // Radix-2 FFT (Cooley-Tukey)
        // 输入: 复数数组 [re0, im0, re1, im1, ...] (交错格式)
        // n 必须是 2 的幂
        //
        // 算法:
        // 1. 位反转排列 (bit-reversal permutation)
        // 2. 蝶形运算 (butterfly) log2(n) 级
        //
        // 由于 T0Kernel 不支持动态循环, 使用 CPU fallback
        // TODO: 实现 GPU 版 FFT (需要循环展开或 LLVM 后端)

        let host_in = input.host_ptr.ok_or("Input not CPU-mapped")? as *const f32;
        let host_out = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;

        let len = n as usize;
        let mut data: Vec<f32> = unsafe {
            std::slice::from_raw_parts(host_in, len * 2).to_vec()
        };

        // 位反转排列
        let log_n = (len as f64).log2() as u32;
        for i in 0..len {
            let j = i.reverse_bits() >> (usize::BITS - log_n);
            if i < j {
                data.swap(i * 2, j * 2);
                data.swap(i * 2 + 1, j * 2 + 1);
            }
        }

        // 蝶形运算
        let sign = match direction {
            FftDirection::Forward => -1.0f32,
            FftDirection::Inverse => 1.0f32,
        };

        let mut m = 1;
        for _ in 0..log_n {
            let half_m = m;
            m *= 2;
            let angle = sign * std::f32::consts::PI / half_m as f32;
            let wm_re = angle.cos();
            let wm_im = angle.sin();

            for k in (0..len).step_by(m) {
                let mut w_re = 1.0f32;
                let mut w_im = 0.0f32;

                for j in 0..half_m {
                    let idx_even = (k + j) * 2;
                    let idx_odd = (k + j + half_m) * 2;

                    let t_re = w_re * data[idx_odd] - w_im * data[idx_odd + 1];
                    let t_im = w_re * data[idx_odd + 1] + w_im * data[idx_odd];

                    let u_re = data[idx_even];
                    let u_im = data[idx_even + 1];

                    data[idx_even] = u_re + t_re;
                    data[idx_even + 1] = u_im + t_im;
                    data[idx_odd] = u_re - t_re;
                    data[idx_odd + 1] = u_im - t_im;

                    let new_w_re = w_re * wm_re - w_im * wm_im;
                    let new_w_im = w_re * wm_im + w_im * wm_re;
                    w_re = new_w_re;
                    w_im = new_w_im;
                }
            }
        }

        // 逆变换需要归一化
        if matches!(direction, FftDirection::Inverse) {
            let n_f = len as f32;
            for i in 0..len * 2 {
                data[i] /= n_f;
            }
        }

        // 写回输出
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(host_out, len * 2)
        };
        out_slice.copy_from_slice(&data);

        Ok(())
    }
}
