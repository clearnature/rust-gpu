#[cfg(test)]
mod check {
    #[test]
    fn check_v_add_co_encoding() {
        use crate::rdna3_asm::gfx11;
        
        // v_add_co_u32 v4, vcc_lo, v4, v3
        let [w0, w1] = gfx11::v_add_co_u32(4, 0x6A, 4, 3);
        eprintln!("Our encoder: word0=0x{:08X} word1=0x{:08X}", w0, w1);
        eprintln!("llvm-mc:     word0=0xD7006A04 word1=0x02020704");
        
        let src0 = w1 & 0x1FF;
        let src1 = (w1 >> 9) & 0x1FF;
        let src2 = (w1 >> 18) & 0x1FF;
        eprintln!("Our word1: src0={} src1={} src2={}", src0, src1, src2);
        
        let w1_ref: u32 = 0x02020704;
        let src0_r = w1_ref & 0x1FF;
        let src1_r = (w1_ref >> 9) & 0x1FF;
        let src2_r = (w1_ref >> 18) & 0x1FF;
        eprintln!("LLVM  word1: src0={} src1={} src2={}", src0_r, src1_r, src2_r);
        
        // Check difference
        let diff = w1 ^ w1_ref;
        eprintln!("XOR diff: 0x{:08X} = bit {}", diff, diff.trailing_zeros());
        
        // v_add_co_ci_u32 v5, vcc_lo, v5, 0, vcc_lo
        let [w0, w1] = gfx11::v_add_co_ci_u32(5, 0x6A, 5, 0, 0x6A);
        eprintln!("\nv_add_co_ci_u32: word0=0x{:08X} word1=0x{:08X}", w0, w1);
        eprintln!("llvm-mc:         word0=0xD5206A05 word1=0x01A90105");
        
        let src0 = w1 & 0x1FF;
        let src1 = (w1 >> 9) & 0x1FF;
        let src2 = (w1 >> 18) & 0x1FF;
        eprintln!("Our word1: src0={} src1={} src2={}", src0, src1, src2);
        
        let w1_ref: u32 = 0x01A90105;
        let src0_r = w1_ref & 0x1FF;
        let src1_r = (w1_ref >> 9) & 0x1FF;
        let src2_r = (w1_ref >> 18) & 0x1FF;
        eprintln!("LLVM  word1: src0={} src1={} src2={}", src0_r, src1_r, src2_r);
    }
}
