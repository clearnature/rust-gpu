use crate::universal::core::GpuMemory;
use crate::universal::math::RngLib;

// ═══════════════════════════════════════════════════════
// T0RngLib — PRNG 实现 (Philox 4x32)
// ═══════════════════════════════════════════════════════

pub struct T0RngLib {
    device: std::sync::Arc<dyn crate::universal::core::GpuDevice>,
}

impl T0RngLib {
    pub fn new(device: std::sync::Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self { device }
    }
}

/// Philox 4x32-10 PRNG
/// 参考: Salmon et al., "Parallel random numbers: as easy as 1, 2, 3" (2011)
fn philox_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const PHILOX_M4: u64 = 0xD2511F53;
    const PHILOX_M2: u64 = 0xCD9E8D57;

    let hi0 = (ctr[0] as u64).wrapping_mul(PHILOX_M4);
    let lo0 = (ctr[2] as u64).wrapping_mul(PHILOX_M2);

    [
        (hi0 >> 32) as u32 ^ ctr[1] ^ key[0],
        hi0 as u32,
        (lo0 >> 32) as u32 ^ ctr[3] ^ key[1],
        lo0 as u32,
    ]
}

fn philox_4x32(seed: u64, idx: u64) -> [u32; 4] {
    let mut ctr = [
        idx as u32,
        (idx >> 32) as u32,
        seed as u32,
        (seed >> 32) as u32,
    ];
    let mut key = [0x9E3779B9u32, 0xBB67AE85u32];

    // 10 rounds
    for _ in 0..10 {
        ctr = philox_round(ctr, key);
        key[0] = key[0].wrapping_add(0x9E3779B9);
        key[1] = key[1].wrapping_add(0xBB67AE85);
    }

    ctr
}

/// u32 → f32 uniform [0, 1)
fn u32_to_uniform(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / 16777216.0) // 2^24, avoid precision issues
}

/// Box-Muller 变换: uniform → normal
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

impl RngLib for T0RngLib {
    fn uniform(
        &self,
        _queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        seed: u64,
    ) -> Result<(), String> {
        let host_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;
        let slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, n as usize) };

        for i in 0..n as usize {
            let ctr = philox_4x32(seed, i as u64);
            slice[i] = u32_to_uniform(ctr[0]);
        }

        Ok(())
    }

    fn normal(
        &self,
        _queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        seed: u64,
    ) -> Result<(), String> {
        let host_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut f32;
        let slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, n as usize) };

        let mut i = 0;
        while i + 1 < n as usize {
            let ctr = philox_4x32(seed, (i / 2) as u64);
            let u1 = u32_to_uniform(ctr[0]).max(1e-10); // avoid log(0)
            let u2 = u32_to_uniform(ctr[1]);
            let (n1, n2) = box_muller(u1, u2);
            slice[i] = n1;
            slice[i + 1] = n2;
            i += 2;
        }
        if i < n as usize {
            let ctr = philox_4x32(seed, (i / 2) as u64);
            let u1 = u32_to_uniform(ctr[0]).max(1e-10);
            let u2 = u32_to_uniform(ctr[1]);
            slice[i] = box_muller(u1, u2).0;
        }

        Ok(())
    }

    fn bernoulli(
        &self,
        _queue: &mut dyn crate::universal::core::ComputeQueue,
        output: &GpuMemory,
        n: u32,
        p: f32,
        seed: u64,
    ) -> Result<(), String> {
        let host_ptr = output.host_ptr.ok_or("Output not CPU-mapped")? as *mut u32;
        let slice = unsafe { std::slice::from_raw_parts_mut(host_ptr, n as usize) };

        for i in 0..n as usize {
            let ctr = philox_4x32(seed, i as u64);
            let u = u32_to_uniform(ctr[0]);
            slice[i] = if u < p { 1 } else { 0 };
        }

        Ok(())
    }
}
