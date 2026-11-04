// 列出 cost_model 4096³ 候选配置名
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::t0::cost_model::{auto_schedule_gemm, DataFormat};
    for (i, c) in auto_schedule_gemm(4096, 4096, 4096, DataFormat::BF16).iter().take(10).enumerate() {
        println!("[{i}] {} (score {:.0})", c.config.name(), c.score);
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
