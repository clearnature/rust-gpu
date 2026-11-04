// AdamW GPU 数值诊断：dump params/m/v 的 GPU 产出 vs CPU 参考的前 N 项
// cargo run --release --features rocm --example adamw_debug
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::ignis::gpu_context::GpuRuntime;
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig};
    use t0_gpu::t0::adamw_kernels::{build_adamw_step, cpu_adamw_step, adamw_grid};

    let rt = GpuRuntime::new().expect("runtime");
    let n: u32 = 256;
    let (lr, beta1, beta2, eps, wd, t) = (1e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32, 1u32);

    let params_init: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 12.8).collect();
    let grads: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin() * 0.5)).collect();
    let m_init = vec![0.0f32; n as usize];
    let v_init = vec![0.0f32; n as usize];

    let mut cpu_params = params_init.clone();
    let mut cpu_m = m_init.clone();
    let mut cpu_v = v_init.clone();
    cpu_adamw_step(&mut cpu_params, &grads, &mut cpu_m, &mut cpu_v, lr, beta1, beta2, eps, wd, t);

    let params_buf = rt.upload_f32(&params_init).unwrap();
    let grads_buf = rt.upload_f32(&grads).unwrap();
    let m_buf = rt.upload_f32(&m_init).unwrap();
    let v_buf = rt.upload_f32(&v_init).unwrap();

    let bc1 = 1.0f32 / (1.0 - beta1.powi(t as i32));
    let bc2 = 1.0f32 / (1.0 - beta2.powi(t as i32));

    let kb = build_adamw_step();
    let ck = kb.compile_via_ssa(t0_gpu::t0::ir::Target::detect()).expect("compile");
    let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
        workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
    }).expect("load");
    println!("[adamw] ck: wg={:?} lds={} ka={} args={:?}",
        ck.workgroup_size, ck.lds_size, ck.kernarg_size,
        ck.args.iter().map(|a| (a.name.clone(), a.offset)).collect::<Vec<_>>());

    let ka = t0_gpu::kernargs![
        params_buf.gpu_addr() => u64,
        grads_buf.gpu_addr() => u64,
        m_buf.gpu_addr() => u64,
        v_buf.gpu_addr() => u64,
        n => u32,
        lr => f32,
        beta1 => f32,
        beta2 => f32,
        eps => f32,
        wd => f32,
        bc1 => f32,
        bc2 => f32
    ];
    println!("[adamw] ka len={} (expect {})", ka.len(), ck.kernarg_size);
    let (grid_x, _) = adamw_grid(n);
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

    let gpu_p = rt.read_f32(&params_buf, n as usize);
    let gpu_m = rt.read_f32(&m_buf, n as usize);
    let gpu_v = rt.read_f32(&v_buf, n as usize);

    println!("{:>4} {:>14} {:>14} | {:>12} {:>12} | {:>12} {:>12}",
        "i", "gpu_p", "cpu_p", "gpu_m", "cpu_m", "gpu_v", "cpu_v");
    for i in 0..16usize {
        println!("{:>4} {:>14.4} {:>14.4} | {:>12.4} {:>12.4} | {:>12.4} {:>12.4}",
            i, gpu_p[i], cpu_params[i], gpu_m[i], cpu_m[i], gpu_v[i], cpu_v[i]);
    }
    let worst = (0..n as usize)
        .max_by(|&a, &b| (gpu_p[a]-cpu_params[a]).abs().partial_cmp(&(gpu_p[b]-cpu_params[b]).abs()).unwrap())
        .unwrap();
    println!("[adamw] worst idx={} gpu={} cpu={} bits(gpu)={:#010x}",
        worst, gpu_p[worst], cpu_params[worst], gpu_p[worst].to_bits());
    let n_bad = (0..n as usize).filter(|&i| (gpu_p[i]-cpu_params[i]).abs() > 1e-4).count();
    println!("[adamw] bad params: {n_bad}/{}", n);
}
