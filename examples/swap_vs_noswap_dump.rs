// swap vs non-swap 的 IR LDS 流量对比（前 40 条 ds 指令）
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::t0::tile_ir::{lower_gemm, TileGemm};
    let mode = std::env::var("MODE").unwrap_or_else(|_| "swap".into());
    let mut spec = TileGemm::tile_64x64_k16();
    spec.acc_swap = mode == "swap";
    let kernel = lower_gemm(&spec);
    println!("=== mode={mode} ===");
    let mut n = 0;
    for (i, op) in kernel.ops().iter().enumerate() {
        let s = format!("{op:?}");
        if s.contains("DsLoadB128") || s.contains("DsStoreB128") || s.contains("Wmma") {
            println!("Op[{i}] {s}");
            n += 1;
            if n >= 40 { break; }
        }
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
