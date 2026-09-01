#[cfg(all(test, feature = "rocm"))]
mod fft_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::math::{FftLib, FftDirection, GpuFftLib};
    use std::sync::{Arc, OnceLock};

    struct SyncDev(Arc<dyn GpuDevice>);
    unsafe impl Sync for SyncDev {}
    unsafe impl Send for SyncDev {}
    static DEVICE: OnceLock<SyncDev> = OnceLock::new();

    fn get_device() -> Arc<dyn GpuDevice> {
        let dev = DEVICE.get_or_init(|| {
            let mgr = DeviceManager::discover();
            assert!(!mgr.devices().is_empty());
            let device = mgr.open(mgr.devices()[0].id).unwrap();
            SyncDev(Arc::from(device))
        });
        dev.0.clone()
    }

    // ═══════════════════════════════════════════════════════
    // FFT 基础测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_fft_constant_signal() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 常数信号: [1, 0, 1, 0, 1, 0, 1, 0] (4 个复数, 全部 = 1+0i)
        let n = 4u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n {
            unsafe {
                *ptr.add(i as usize * 2) = 1.0;      // re
                *ptr.add(i as usize * 2 + 1) = 0.0;  // im
            }
        }

        fft.fft_1d(&mut *queue, &output, &input, n, FftDirection::Forward).unwrap();

        let out_ptr = output.host_ptr.unwrap() as *const f32;
        // 常数信号的 FFT: [4, 0, 0, 0, 0, 0, 0, 0]
        let re0 = unsafe { *out_ptr };
        let im0 = unsafe { *out_ptr.add(1) };

        eprintln!("[FFT] Constant signal: X[0] = {} + {}i", re0, im0);
        assert!((re0 - 4.0).abs() < 0.01, "X[0] should be 4, got {}", re0);
        assert!(im0.abs() < 0.01, "X[0] imag should be 0, got {}", im0);
    }

    #[test]
    fn test_fft_impulse() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 脉冲信号: [1, 0, 0, 0, 0, 0, 0, 0] (δ[n])
        let n = 4u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        unsafe {
            *ptr.add(0) = 1.0;  // re0 = 1
            *ptr.add(1) = 0.0;  // im0 = 0
            *ptr.add(2) = 0.0;  // re1 = 0
            *ptr.add(3) = 0.0;  // im1 = 0
            *ptr.add(4) = 0.0;  // re2 = 0
            *ptr.add(5) = 0.0;  // im2 = 0
            *ptr.add(6) = 0.0;  // re3 = 0
            *ptr.add(7) = 0.0;  // im3 = 0
        }

        fft.fft_1d(&mut *queue, &output, &input, n, FftDirection::Forward).unwrap();

        let out_ptr = output.host_ptr.unwrap() as *const f32;
        // 脉冲信号的 FFT: 全部 = 1+0i
        for i in 0..n {
            let re = unsafe { *out_ptr.add(i as usize * 2) };
            let im = unsafe { *out_ptr.add(i as usize * 2 + 1) };
            eprintln!("[FFT] Impulse: X[{}] = {} + {}i", i, re, im);
            assert!((re - 1.0).abs() < 0.01, "X[{}] re should be 1, got {}", i, re);
            assert!(im.abs() < 0.01, "X[{}] im should be 0, got {}", i, im);
        }
    }

    #[test]
    fn test_fft_roundtrip() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 信号: [1, 2, 3, 4] (实数, 虚部=0)
        let n = 4u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let freq = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n {
            unsafe {
                *ptr.add(i as usize * 2) = (i + 1) as f32;  // re = 1,2,3,4
                *ptr.add(i as usize * 2 + 1) = 0.0;         // im = 0
            }
        }

        // Forward FFT
        fft.fft_1d(&mut *queue, &freq, &input, n, FftDirection::Forward).unwrap();

        // Inverse FFT
        fft.fft_1d(&mut *queue, &output, &freq, n, FftDirection::Inverse).unwrap();

        // 验证 roundtrip: output 应该等于 input
        let in_ptr = input.host_ptr.unwrap() as *const f32;
        let out_ptr = output.host_ptr.unwrap() as *const f32;

        for i in 0..n * 2 {
            let expected = unsafe { *in_ptr.add(i as usize) };
            let actual = unsafe { *out_ptr.add(i as usize) };
            eprintln!("[FFT] Roundtrip: [{}] expected={} actual={}", i, expected, actual);
            assert!((actual - expected).abs() < 0.01, "Roundtrip mismatch at [{}]: {} vs {}", i, expected, actual);
        }
    }

    #[test]
    fn test_fft_sine_wave() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 正弦波: sin(2π * k * n / N), k=1
        let n = 8u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
            unsafe {
                *ptr.add(i as usize * 2) = angle.sin();      // re = sin
                *ptr.add(i as usize * 2 + 1) = 0.0;          // im = 0
            }
        }

        fft.fft_1d(&mut *queue, &output, &input, n, FftDirection::Forward).unwrap();

        let out_ptr = output.host_ptr.unwrap() as *const f32;
        eprintln!("[FFT] Sine wave spectrum:");
        for i in 0..n {
            let re = unsafe { *out_ptr.add(i as usize * 2) };
            let im = unsafe { *out_ptr.add(i as usize * 2 + 1) };
            let mag = (re * re + im * im).sqrt();
            eprintln!("  X[{}] = {} + {}i (|X| = {:.3})", i, re, im, mag);
        }

        // 正弦波 sin(2π*n/8) 的 FFT 应该在 k=1 和 k=7 处有峰值
        let re1 = unsafe { *out_ptr.add(2) };  // X[1].re
        let im1 = unsafe { *out_ptr.add(3) };  // X[1].im
        let mag1 = (re1 * re1 + im1 * im1).sqrt();
        eprintln!("[FFT] X[1] magnitude = {:.3} (should be ~4)", mag1);
        assert!(mag1 > 3.0, "X[1] magnitude should be ~4, got {}", mag1);
    }
}
