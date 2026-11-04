//! AQL/KFD 契约断言（TDD 回归锁）
//!
//! 出处标注：
//!   [HSA]  HSA 规格包布局（rocr-runtime hsa.h: hsa_kernel_dispatch_packet_s）
//!   [UAPI] 内核 kfd_ioctl.h（ioctl 编码规则：dir/size/type=0x4B/nr）
//!   [事故] 2026-08-28 AQL header 顺序事故（一次 memcpy 写 → 硬件读未就绪字段）
//!   [GPU]  本机 RX 9060 XT 实测
//!
//! 用法：cargo test --release --features rocm --test aql_contract

use std::mem::offset_of;
use t0_gpu::ignis::gpu_context::GpuRuntime;
use t0_gpu::kfd::{AqlDispatchPacket, DispatchPool, KfdDevice};

// ═══════════════════════════════════════════════════════════════
// 组 1：包布局 [HSA] —— hsa_kernel_dispatch_packet_s 契约
//═══════════════════════════════════════════════════════════════

/// [HSA] 64B 包 + 逐字段偏移（HSA 规格布局；KFD/CP 按硬编码偏移读取，
/// 任何字段重排都是生成级破坏）
#[test]
fn aql_packet_layout_matches_hsa_spec() {
    assert_eq!(std::mem::size_of::<AqlDispatchPacket>(), 64, "AQL 包必须 64B");
    assert_eq!(offset_of!(AqlDispatchPacket, workgroup_size_x), 0x04);
    assert_eq!(offset_of!(AqlDispatchPacket, workgroup_size_y), 0x06);
    assert_eq!(offset_of!(AqlDispatchPacket, workgroup_size_z), 0x08);
    assert_eq!(offset_of!(AqlDispatchPacket, reserved0), 0x0A);
    assert_eq!(offset_of!(AqlDispatchPacket, grid_size_x), 0x0C);
    assert_eq!(offset_of!(AqlDispatchPacket, grid_size_y), 0x10);
    assert_eq!(offset_of!(AqlDispatchPacket, grid_size_z), 0x14);
    assert_eq!(offset_of!(AqlDispatchPacket, private_segment_size), 0x18);
    assert_eq!(offset_of!(AqlDispatchPacket, group_segment_size), 0x1C);
    assert_eq!(offset_of!(AqlDispatchPacket, kernel_object), 0x20);
    assert_eq!(offset_of!(AqlDispatchPacket, kernarg_address), 0x28);
    assert_eq!(offset_of!(AqlDispatchPacket, reserved2), 0x30);
    assert_eq!(offset_of!(AqlDispatchPacket, completion_signal), 0x38);
}

/// [HSA+事故回归] header 位域分量断言：
///   type=2（bits0:7）、barrier=1（bit8，派发间数据依赖顺序）、
///   acquire/release fence = SYSTEM=2（bits9:10 / 11:12）→ 组合 0x1502。
///   （对照：ROCm 无 barrier 示例 = 0x1402——t0 需要 barrier 保证多内核
///   数据依赖，2026-08-28 抓包协议；分量断言便于定位哪个位域漂移）
#[test]
fn aql_header_bit_fields_compose_0x1502() {
    let t = t0_gpu::kfd::HSA_PACKET_TYPE_KERNEL_DISPATCH as u16;      // 2
    let f = t0_gpu::kfd::HSA_FENCE_SCOPE_SYSTEM as u16;               // 2
    assert_eq!(t, 2, "packet type 必须 = kernel dispatch(2)");
    assert_eq!(f, 2, "fence scope 必须 = SYSTEM(2)——裸金属自管一致性");

    let header: u16 = t | (1u16 << 8) | (f << 9) | (f << 11);
    assert_eq!(header & 0xFF, 2, "bits0:7 = type");
    assert_eq!((header >> 8) & 1, 1, "bit8 = barrier（派发间依赖）");
    assert_eq!((header >> 9) & 0x3, 2, "bits9:10 = acquire fence SYSTEM");
    assert_eq!((header >> 11) & 0x3, 2, "bits11:12 = release fence SYSTEM");
    assert_eq!(header, 0x1502, "组合值（含 barrier bit）");
}

/// [UAPI] ioctl 编码对照内核 kfd_ioctl.h
///（编码规则：_IOWR/_IOW/_IOR 的 dir/size/type=0x4B/nr；tinygrad 同值交叉验证）
#[test]
fn ioctl_encodings_match_kernel_uapi() {
    use t0_gpu::kfd::{AMDKFD_IOC_ACQUIRE_VM, AMDKFD_IOC_ALLOC_MEMORY, AMDKFD_IOC_CREATE_QUEUE, AMDKFD_IOC_GET_VERSION};
    assert_eq!(AMDKFD_IOC_GET_VERSION, 0x80084B01);
    assert_eq!(AMDKFD_IOC_CREATE_QUEUE, 0xC0604B02);
    assert_eq!(AMDKFD_IOC_ACQUIRE_VM, 0x40084B15);
    assert_eq!(AMDKFD_IOC_ALLOC_MEMORY, 0xC0284B16);
    // type 字节必须是 0x4B（'K'=KFD magic）
    for (name, v) in [
        ("GET_VERSION", AMDKFD_IOC_GET_VERSION),
        ("CREATE_QUEUE", AMDKFD_IOC_CREATE_QUEUE),
        ("ACQUIRE_VM", AMDKFD_IOC_ACQUIRE_VM),
        ("ALLOC_MEMORY", AMDKFD_IOC_ALLOC_MEMORY),
    ] {
        assert_eq!((v >> 8) & 0xFF, 0x4B, "{name} 的 ioctl type 字节非 0x4B");
    }
}

// ═══════════════════════════════════════════════════════════════
// 组 2：运行时行为 [GPU]（每次 init 一次 GPU，控制派发次数）
// ═══════════════════════════════════════════════════════════════

/// [事故回归] write_kernargs 必须先 wipe 整个 slot 再写入——
/// 超出本次 data 的尾部绝不能残留上次派发的旧指针
///（Direction 2 修复：stale VA → 页错误 / KD stall）
#[test]
fn dispatch_pool_kernarg_slot_is_wiped_on_rewrite() {
    let dev = KfdDevice::open().expect("kfd");
    let pool = DispatchPool::new(&dev, 2).expect("pool");

    // 第一次：40B 全 0x11（污染 slot）
    pool.write_kernargs(0, &vec![0x11u8; 40]);
    let mid = pool.get_kernargs(0).read_bytes(0, 40);
    assert!(mid.iter().all(|&b| b == 0x11), "写入未生效（测试前置失败）");

    // 第二次：8B 短写入 → 尾部 32B 必须被 wipe 清零
    pool.write_kernargs(0, &vec![0x22u8; 8]);
    let after = pool.get_kernargs(0).read_bytes(0, 40);
    assert_eq!(&after[..8], &[0x22u8; 8], "新写入的前 8B");
    assert!(after[8..].iter().all(|&b| b == 0),
        "slot 尾部残留旧 kernarg（wipe 语义破坏）→ stale VA 页错误风险");
}

/// [事故回归] queue poison 后必须拒绝派发——GPU 挂起恢复后防止系统级联挂死
///（2026-11-04 探针实证：poisoned 队列继续派发 → 系统挂）
#[test]
fn poisoned_runtime_refuses_dispatch() {
    let rt = GpuRuntime::new().expect("runtime");
    rt.mark_poisoned();
    assert!(rt.is_poisoned());

    // 构造最小合法派发参数（kernel 可为空壳——poison 检查在派发之前）
    let gk = t0_gpu::kfd::GpuKernel {
        code_buffer: {
            // 最小 code buffer：256B VRAM
            rt.device.alloc_code(256).expect("alloc_code")
        },
        descriptor_va: 0,
        code_entry_va: 0,
        rsrc1: 0,
        rsrc2: 0,
        lds_size: 0,
        workgroup_size: [64, 1, 1],
        kernarg_size: 0,
    };
    let ka = vec![0u8; 64];
    let err = rt.dispatch(&gk, [64, 1, 1], &ka).expect_err("poisoned runtime 必须拒绝派发");
    assert!(err.contains("poisoned"), "拒绝信息应含 poisoned: {err}");
}
