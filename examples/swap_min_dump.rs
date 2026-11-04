// 最小 swap 序列 ASM 人工走查用
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::t0::tile_ir::{lower_gemm, TileGemm};
    let mut spec = TileGemm::tile_64x64_k16();
    spec.acc_swap = true;
    let kernel = lower_gemm(&spec);
    // 打印 acc_swap 相关片段：找 ds_store 到 swap 区与 ds_load 回来的指令
    for (i, op) in kernel.ops().iter().enumerate() {
        let s = format!("{op:?}");
        if s.contains("DsStoreB128") || s.contains("DsLoadB128") {
            println!("Op[{i}] {s}");
        }
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
