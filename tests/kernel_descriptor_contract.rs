//! KernelDescriptor 字节级契约断言（TDD 回归锁，纯 CPU）
//!
//! 出处：[ROCm] amd_hsa_kernel_code.h KD 布局 + [GPU] KFD 加载器（kernel.rs）
//! 四个硬编码读取点：kernarg_size@0x8、entry_offset@0x10、rsrc1@0x30、rsrc2@0x34。
//! KD 由 for_gfx11 构建（GFX1100/GFX1200 共用，内部有 gfx12 分支）。
//!
//! 用法：cargo test --release --features rocm --test kernel_descriptor_contract

use t0_gpu::rdna3_code_object::{KernelConfig, KernelDescriptor};

fn test_config(target: &str) -> KernelConfig {
    KernelConfig {
        name: "kd_contract".into(),
        lds_size: 2048,
        kernarg_size: 40,
        vgpr_count: 24,
        sgpr_count: 16,
        workgroup_size_x: 128,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        scratch_size: 0,
        target_gfx: target.into(),
    }
}

/// [ROCm] KD 总长 64B（amd_hsa_kernel_code.h；KFD 按此尺寸映射）
#[test]
fn kernel_descriptor_is_64_bytes() {
    let kd = KernelDescriptor::for_gfx11(&test_config("gfx1200"), 4096);
    assert_eq!(kd.as_bytes().len(), 64, "KD 必须 64B");
}

/// [ROCm+GPU] 关键字段偏移（KFD 加载器 kernel.rs 硬编码读取点一一对应）：
///   group_segment_fixed_size@0x0、kernarg_size@0x8（loader 读 0x8）、
///   kernel_code_entry_byte_offset@0x10（loader 读 0x10，KFD patch 0→0x10xx）、
///   compute_pgm_rsrc1@0x30（loader 读 0x30 并 patch PRIV bit20）
#[test]
fn kd_field_offsets_match_kfd_loader_read_points() {
    let kd = KernelDescriptor::for_gfx11(&test_config("gfx1200"), 4096);
    let b = kd.as_bytes();

    assert_eq!(u32::from_le_bytes(b[0x00..0x04].try_into().unwrap()), 2048,
        "group_segment_fixed_size@0x0");
    assert_eq!(u32::from_le_bytes(b[0x08..0x0C].try_into().unwrap()), 40,
        "kernarg_size@0x8（KFD loader 读点）");
    assert_eq!(i64::from_le_bytes(b[0x10..0x18].try_into().unwrap()), 64,
        "kernel_code_entry_byte_offset@0x10 = 64（代码紧跟描述符；KFD patch 源）");

    let rsrc1 = u32::from_le_bytes(b[0x30..0x34].try_into().unwrap());
    // 真实编码（for_gfx11 实现）：
    //   VGPR[7:0] = (count+7)/8 - 1（Wave32 粒度 8 的数量-1 编码）
    //   SGPR[13:6] = 0（GFX10+ 必须为 0，SGPR 数量经其它机制管理）
    //   FLOAT_MODE[19:12] = 0xCF（FP32_ROUND[15:12]=0xF + FP16_ROUND[19:16]=0xC）
    assert_eq!(rsrc1 & 0xFF, (24 + 7) / 8 - 1, "rsrc1 VGPR 数量-1 编码（粒度 8）");
    assert_eq!((rsrc1 >> 6) & 0x3F, 0, "rsrc1 SGPR[13:6] GFX10+ 必须为 0");
    assert_eq!((rsrc1 >> 12) & 0xFF, 0xCF, "rsrc1 FLOAT_MODE[19:12] = 0xCF");
    assert_eq!((rsrc1 >> 12) & 0xF, 0xF, "FP32_ROUND[15:12] = Round Nearest Even");
    // GFX1200 分支：MEM_ORDERED bit30 + FORWARD_PROGRESS bit31
    assert_eq!((rsrc1 >> 30) & 0x1, 1, "bit30 ENABLE_MEM_ORDERED（gfx12 分支）");
    assert_eq!((rsrc1 >> 31) & 0x1, 1, "bit31 FORWARD_PROGRESS（gfx12 分支）");
}

/// [ROCm] GFX1100 与 GFX1200 的 rsrc1 差异恰为 bit30/31（其余位相同）
#[test]
fn rsrc1_gfx12_differs_only_in_bits_30_31() {
    let k = u32::from_le_bytes(
        KernelDescriptor::for_gfx11(&test_config("gfx1100"), 4096).as_bytes()[0x30..0x34]
            .try_into()
            .unwrap(),
    );
    let g = u32::from_le_bytes(
        KernelDescriptor::for_gfx11(&test_config("gfx1200"), 4096).as_bytes()[0x30..0x34]
            .try_into()
            .unwrap(),
    );
    assert_eq!((g >> 30) & 0x1, 1, "gfx1200 MEM_ORDERED");
    assert_eq!((k >> 30) & 0x1, 0, "gfx1100 无 MEM_ORDERED 强制位");
    let mask = (1u32 << 30) | (1u32 << 31);
    assert_eq!(g & !mask, k & !mask, "除 bit30/31 外两架构 rsrc1 应逐位相同");
}

/// [ROCm] kernel_code_properties：bit3=kernarg_segment_ptr、bit7=workitem_id_x、
/// bit10=wavefront_size32（wave32）；rsrc2 user_sgpr=2（bit1:2 = kernarg ptr 双字）
#[test]
fn kd_properties_and_rsrc2() {
    let kd = KernelDescriptor::for_gfx11(&test_config("gfx1200"), 4096);
    let b = kd.as_bytes();
    let props = u16::from_le_bytes(b[0x38..0x3A].try_into().unwrap());
    assert_eq!(props & (1 << 3), 1 << 3, "ENABLE_KERNARG_SEGMENT_PTR（bit3）");
    assert_eq!(props & (1 << 7), 1 << 7, "ENABLE_WORKITEM_ID_X（bit7）");
    assert_eq!(props & (1 << 10), 1 << 10, "WAVEFRONT_SIZE32（bit10，wave32）");

    let rsrc2 = u32::from_le_bytes(b[0x34..0x38].try_into().unwrap());
    assert_eq!((rsrc2 >> 1) & 0x3, 2, "USER_SGPR=2（kernarg 指针双字）");
    assert_eq!(rsrc2 & 1, 0, "无 private segment（scratch_size=0）");
}

/// [ROCm] GFX1100 与 GFX1200 的 ELF e_flags（ELF64 header offset 48）：
/// 0x048 / 0x041——rdna3_code_object e_flags 分支回归锁
#[test]
fn code_object_elf_e_flags_per_arch() {
    for (target, expect) in [("gfx1200", 0x048u32), ("gfx1100", 0x041u32)] {
        use t0_gpu::rdna3_asm::Rdna3Assembler;
        use t0_gpu::rdna3_code_object::AmdGpuCodeObject;
        let mut asm = Rdna3Assembler::new();
        asm.set_target(target);
        asm.emit(t0_gpu::rdna3_asm::gfx11::S_ENDPGM);
        let co = AmdGpuCodeObject::from_assembler(&asm, test_config(target));
        let elf = co.to_bytes_v2();
        let e_flags = u32::from_le_bytes(elf[48..52].try_into().unwrap());
        assert_eq!(e_flags, expect, "{target} e_flags（offset 48）");
    }
}
