//! PM4 命令流契约断言（TDD 回归锁，纯 CPU）
//!
//! 出处标注：
//!   [PM4]  AMD PM4 规格：PACKET3 header = type3(3<<30) | (count-1)<<16 | opcode<<8
//!   [UAPI/内核驱动] opcode 值（SET_SH_REG=0x76 等，amdgpu pm4_packages.h 同值）
//!   [GPU]  PM4-in-AQL 路径实测（compute_barrier + release_mem 替代 wait_idle）
//!
//! 用法：cargo test --release --features rocm --test pm4_contract

use t0_gpu::kfd::pm4::Pm4CmdBuilder;
use t0_gpu::kfd::{PM4_ACQUIRE_MEM, PM4_DISPATCH_DIRECT, PM4_SET_SH_REG};

/// [PM4] PACKET3 header 编码：type3(3<<30) | (body-1)<<16 | opcode<<8
///（count 字段 = body dwords - 1，PM4 规格定义）
#[test]
fn pkt3_header_encoding() {
    let mut b = Pm4CmdBuilder::new();
    b.set_sh_reg(0x2C04, &[0x11, 0x22, 0x33]); // body: reg_offset + 3 values = 4 dwords
    let cmds = b.finish();
    assert_eq!(cmds.len(), 5, "header + 4 body dwords");
    let header = cmds[0];
    assert_eq!((header >> 30) & 0x3, 3, "packet type 必须 3（PACKET3）");
    assert_eq!((header >> 16) & 0x3FFF, 3, "count 字段 = body-1 = 3");
    assert_eq!((header >> 8) & 0xFF, PM4_SET_SH_REG, "opcode = SET_SH_REG(0x76)");
    // body: reg_offset（(reg_addr-SH_REG_BASE)>>2）+ 值序列
    assert_eq!(cmds[1], (0x2A04 - 0x2A00) >> 2, "reg_offset = (addr-base)>>2");
    assert_eq!(&cmds[2..5], &[0x11, 0x22, 0x33]);
}

/// [GPU/事故回归] DISPATCH_DIRECT 命令流：opcode 0x15 + grid 三维 + initiator
///（grid 单位 work-item 与 HSA 一致——PM4 路径与 AQL 路径同语义）
#[test]
fn dispatch_direct_command_stream() {
    let mut b = Pm4CmdBuilder::new();
    b.dispatch_direct([512, 2, 1], 0x1);
    let cmds = b.finish();
    assert_eq!(cmds.len(), 5);
    let header = cmds[0];
    assert_eq!((header >> 30) & 0x3, 3);
    assert_eq!((header >> 8) & 0xFF, PM4_DISPATCH_DIRECT, "opcode = DISPATCH_DIRECT(0x15)");
    assert_eq!((header >> 16) & 0x3FFF, 3, "body = grid×3 + initiator = 4 → count 3");
    assert_eq!(&cmds[1..4], &[512, 2, 1], "grid 三维（work-item 单位）");
}

/// [GPU] ACQUIRE_MEM：GCR 缓存失效位全集（GL2_INV|GL2_WB|GL1_INV|…）——
/// 裸金属自管一致性的核心命令，位缺失 = 内核间脏缓存
#[test]
fn acquire_mem_invalidates_all_cache_levels() {
    let mut b = Pm4CmdBuilder::new();
    b.acquire_mem_gfx10();
    let cmds = b.finish();
    // header + 7 body dwords
    let header = cmds[0];
    assert_eq!((header >> 8) & 0xFF, PM4_ACQUIRE_MEM, "opcode = ACQUIRE_MEM(0x58)");
    assert_eq!((header >> 16) & 0x3FFF, 6, "count 字段 = body-1 = 6（body 7 dwords）");
    assert_eq!(cmds.len(), 8, "header + 7 body");
    let gcr = cmds[7];
    // 全缓存级失效位（GLI/GLM/GLM_WB/GLK/GLK_WB/GLV/GL1/GL2_INV + GL2_WB）
    let expected = 0x1FF;
    assert_eq!(gcr, expected, "GCR_CNTL 位集（bit0-8 全失效+写回）: {gcr:#x}");
}

/// [PM4] 多命令流拼接：命令按调用序连续排列（PM4-in-AQL indirect buffer 的结构前提）
#[test]
fn commands_concatenate_in_order() {
    let mut b = Pm4CmdBuilder::new();
    b.dispatch_direct([256, 1, 1], 0);
    b.acquire_mem_gfx10();
    let cmds = b.finish();
    // dispatch: 1+4；acquire: 1+7 → 总 13
    assert_eq!(cmds.len(), 13);
    assert_eq!((cmds[0] >> 8) & 0xFF, PM4_DISPATCH_DIRECT);
    assert_eq!((cmds[5] >> 8) & 0xFF, PM4_ACQUIRE_MEM, "第二段必须是 acquire（按调用序）");
}
