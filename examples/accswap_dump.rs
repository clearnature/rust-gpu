// dump 64x64 acc_swap 内核 IR 中写 VReg(0) 的 op
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::t0::tile_ir::{lower_gemm, TileGemm};
    let mut spec = TileGemm::tile_64x64_k16();
    spec.acc_swap = true;
    let kernel = lower_gemm(&spec);
    for (i, op) in kernel.ops().iter().enumerate() {
        if op.vreg_defs().iter().any(|d| d.0 == 0) {
            println!("Op[{i}] writes VReg(0): {op:?}");
        }
    }
    println!("total ops: {}", kernel.ops().len());
}
#[cfg(not(feature = "rocm"))]
fn main() {}
