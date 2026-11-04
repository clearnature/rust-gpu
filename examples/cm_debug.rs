#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::t0::cost_model::{auto_schedule_gemm, DataFormat};
    let cands = auto_schedule_gemm(4096, 4096, 4096, DataFormat::BF16);
    for (i, c) in cands.iter().enumerate().take(8) {
        let gc = c.config.to_gemm_config();
        println!("[cm] [{i}] {} wg={} score={:.0}", gc.name(), gc.wg_size, c.score);
    }
    println!("[cm] total {}", cands.len());
}
#[cfg(not(feature = "rocm"))]
fn main() {}
