//! ISA/ROCm 事实断言（TDD 回归锁）
//!
//! 每条断言的出处三类（2026-11-08 会话确立，详见 PROJECT_MEMORY）：
//!   [ISA]  RDNA4 Instruction Set Architecture 手册
//!          （/home/yanli/work/9060xt/rdna4-instruction-set-architecture.pdf）
//!   [LLVM] ROCm llvm-project AMDGPU 后端 + 本机 llvm-mc 金标准编码
//!          （/data/ROCm/llvm-project；llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1200）
//!   [GPU]  本机 RX 9060 XT 实测（gfxhub 页错误/ttmp9 探针/fullcheck）
//!
//! 用法：cargo test --release --features rocm --test isa_facts
#![cfg(feature = "rocm")]

use t0_gpu::kfd::{GpuMemset, KfdDevice};
use t0_gpu::t0::gemm_gen::{self, GemmConfig};
use t0_gpu::t0::ir::{Op, Target, VReg, Width};
use t0_gpu::t0::tile_ir::{self, TileGemm};
use t0_gpu::t0::compile::T0Kernel;

// ═══════════════════════════════════════════════════════════════
// 组 1：寄存器事实 [ISA 3.5.3.4 Table 12 + §3.5.3 TTMP 规则]
//   TGID.x 在 ttmp9（编码 117；TTMP0-15 占 108-123）；
//   TTMP 用户态只读（PRIV=0 读合法，写被忽略）。
// ═══════════════════════════════════════════════════════════════

/// [ISA] ttmp9 编码 = 117；v_mov_b32 v1, ttmp9 的 llvm-mc 金标准编码
/// （llvm-mc -mcpu=gfx1200: v_mov_b32_e32 v1, ttmp9 → 0x7E020275）
#[test]
fn ttmp9_encoding_matches_llvm_mc_golden() {
    use t0_gpu::rdna3_asm::gfx11;
    let words = gfx11::v_mov_b32_from_sgpr(1, 117); // 117 = ttmp9
    assert_eq!(words, 0x7E020275, "v_mov v1,ttmp9 编码必须等于 llvm-mc 金标准 0x7E020275");
}

/// [ISA+GPU] GFX1200 上读 TGID.x 的两个内核构建器（aql gpu_zero / pool GpuMemset）
/// 必须都从 ttmp9 取值——s2 是 user SGPR，读到垃圾 0xFFFFFFFF（ttmp9 探针实证）。
/// 本测试跑 GpuMemset 2-WG 清零（gpu_zero 已有 test_llvm_simple_store 覆盖）。
#[test]
fn gpumemset_multi_wg_zeros_fully_on_gfx1200() {
    let dev = KfdDevice::open().expect("kfd");
    if dev.gfx_target_version < 120000 {
        return; // 事实仅对 GFX1200 断言
    }
    let queue = dev.create_queue().expect("queue");
    let ms = GpuMemset::new(&dev).expect("memset kernel");

    let buf = dev.alloc_vram(8192).expect("alloc"); // 8KB = 2 WG × 256 thr × 16B
    buf.write(&vec![0xAAu8; 8192]);
    ms.zero_async(&queue, &buf, 8192);
    let _ = queue.synchronize();

    let out = buf.read_bytes(0, 8192);
    let nonzero = out.iter().filter(|&&b| b != 0).count();
    assert_eq!(nonzero, 0,
        "GpuMemset 多 WG 清零不完整（{nonzero}/8192 非零）——TGID.x 必须读 ttmp9（ISA Table 12），\
         s2 在 RDNA4 是 user SGPR（读到垃圾 → 越界 gid 被 bounds 掩码吞掉 → 区域未清）");
}

/// [ISA §3.5.3] TTMP 用户态只读——发射的 ASM 中 ttmp 只允许出现在源操作数位置，
/// 绝不允许作为 SALU 目的寄存器（写被硬件忽略，写了说明逻辑误解）。
#[test]
fn gfx1200_asm_never_writes_ttmp() {
    for asm in [probe_kernel_asm(), gemm_asm_tile_128x64_k16()] {
        let bad: Vec<&str> = asm.lines()
            .filter(|l| {
                // 写形式：助记符后第一个操作数是 ttmpN（读形式是 sN, ttmpM）
                let l = l.trim_start();
                l.starts_with("s_") && {
                    let rest = l.splitn(2, char::is_whitespace).nth(1).unwrap_or("");
                    rest.starts_with("ttmp")
                }
            })
            .collect();
        assert!(bad.is_empty(), "ASM 出现 TTMP 写（硬件忽略=逻辑错误）：{bad:?}");
    }
}

// ═══════════════════════════════════════════════════════════════
// 组 2：指令形式事实 [LLVM 金标准 + GPU 实测]
//   GFX1200 global 地址空间用 global_load/store（opcode 0xee）；
//   原子返回用 th:TH_ATOMIC_RETURN（非 RDNA3 的 glc）；
//   s_delay_alu 在 GFX1200 禁用（VCC carry-chain 风险）。
// ═══════════════════════════════════════════════════════════════

/// [LLVM] clang -mcpu=gfx1200 对 global 地址空间一律发 global_load/store
///（flat_load 切换是 8/30 捆绑提交的误诊 workaround，已回退——回归锁）
#[test]
fn gfx1200_vmem_form_is_global_not_flat() {
    let asm = probe_kernel_asm();
    assert!(asm.contains("global_load_b32"), "GFX1200 必须发 global_load（LLVM 行为）:\n{asm}");
    assert!(!asm.contains("flat_load"), "GFX1200 不应发 flat_load（误诊 workaround 已回退）:\n{asm}");
}

/// [LLVM+GPU] 原子返回：GFX1200 用 th 字段（th:TH_ATOMIC_RETURN），
/// glc 是 RDNA3 形式（clang gfx1200 实测输出）
#[test]
fn gfx1200_atomic_rtn_uses_th_field() {
    let mut k = T0Kernel::new("isa_atomic_rtn");
    // [A 类家族] 地址对登记 count≥2 并定义两半
    let addr = k.alloc_vreg_array(2, t0_gpu::t0::ir::Alignment::Align2);
    k.v_mov_imm(addr, 0);
    k.v_mov_imm(VReg(addr.0 + 1), 0);
    k.push(Op::GlobalAtomicAddU32Rtn { dst: VReg(4), addr, src: VReg(2) });
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    assert!(asm.contains("th:TH_ATOMIC_RETURN"), "原子返回必须带 th 字段:\n{asm}");
    assert!(!asm.contains(" glc"), "GFX1200 不应使用 RDNA3 的 glc 形式:\n{asm}");
}

/// [GPU 实测] s_delay_alu 在 GFX1200 禁用（提示非语义指令，但本项目发射器
/// 不跟踪 VCC 依赖，发射会致错——caps.delay_alu_ok=false 的输出层回归锁）
#[test]
fn gfx1200_asm_never_emits_s_delay_alu() {
    for asm in [probe_kernel_asm(), gemm_asm_tile_128x64_k16()] {
        assert!(!asm.contains("s_delay_alu"), "GFX1200 ASM 出现 s_delay_alu:\n{}",
            asm.lines().filter(|l| l.contains("s_delay_alu")).take(3).collect::<Vec<_>>().join("\n"));
    }
}

// ═══════════════════════════════════════════════════════════════
// 组 3：等待语义事实 [RDNA4 拆分等待 + F1 elision 修复]
//   GFX1200 拆分形式：s_wait_loadcnt/kmcnt/dscnt/storecnt；
//   n==0 永不 elide（shadow 幻影清零防护，2026-11-08 F1）。
// ═══════════════════════════════════════════════════════════════

/// [F1 回归锁] wait_vmcnt(0) 即使 outstanding==0 也必须发射——
/// 旧 elision 门在 shadow 被条件分支幻影清零时会吞掉真等待（bad=60013 实证）
#[test]
fn waitcnt_zero_never_elided() {
    let mut k = T0Kernel::new("isa_wait_first");
    k.wait_vmcnt(0); // 首条、无在飞 load
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    assert!(asm.contains("s_wait_loadcnt 0"),
        "wait_vmcnt(0) 被吞（F1 回归！）：\n{asm}");
}

/// [RDNA4] WaitLgkmcnt 发射 kmcnt+dscnt 对（dscnt 需 kmcnt 激活；顺序关键）
#[test]
fn gfx1200_wait_lgkmcnt_emits_kmcnt_then_dscnt() {
    let mut k = T0Kernel::new("isa_wait_lgkm");
    let addr = k.alloc_vreg();
    k.lds_store(VReg(0), addr, Width::B32, 0);
    k.wait_lgkmcnt(0);
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    let km = asm.find("s_wait_kmcnt");
    let ds = asm.find("s_wait_dscnt");
    assert!(km.is_some() && ds.is_some(), "LDS 等待必须 kmcnt+dscnt 对:\n{asm}");
    assert!(km.unwrap() < ds.unwrap(), "顺序关键：kmcnt 必须在 dscnt 之前（否则死锁）:\n{asm}");
}

// ═══════════════════════════════════════════════════════════════
// 组 4：LDS 容量契约 [ISA §1.2.2.1：单 work-group LDS 上限 64KB]
//   WGP 128KB = 两个 WG 各 64KB，不是单 WG 128KB（实测越限派发即挂死）。
// ═══════════════════════════════════════════════════════════════

/// [ISA] 合规 preset（≤64KB）必须能 lower；超限 preset 必须被编译期拒绝
///（fail-fast——放行会挂死 GPU，128x128 swap 家族 120s 实证）
#[test]
fn lds_64kb_contract_enforced_at_compile_time() {
    // 合规家族（GPU fullcheck 全绿）
    for spec in [TileGemm::tile_128x64_k16(), TileGemm::tile_256x64_k32_wgp(), TileGemm::tile_64x64_k32()] {
        assert!(spec.lds_total() <= 65536, "{} LDS={} > 64KB", spec.name(), spec.lds_total());
        tile_ir::lower_gemm(&spec); // 不 panic 即通过
    }
    // 超限家族：必须 fail-fast（catch_unwind），绝不生成可派发的坏内核
    for spec in [TileGemm::tile_128x128_k16_swap(), TileGemm::tile_128x128_k64()] {
        if spec.lds_total() > 65536 {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tile_ir::lower_gemm(&spec)));
            assert!(r.is_err(),
                "{} LDS={} > 64KB 却未被拒绝——放行会派发即挂死", spec.name(), spec.lds_total());
        }
    }
}

/// [回归锁 2026-11-08] auto_select 产出的配置必须全部可行
///（每线程 GMEM load ≥1——wg=256×128x64 组合曾生成"WT 从未加载"的坏内核）
#[test]
fn auto_select_outputs_are_feasible() {
    for (m, k, n) in [(256u32, 256u32, 256u32), (1024, 1024, 1024), (4096, 4096, 4096), (128, 4096, 4096)] {
        let cfg = gemm_gen::auto_select(m, k, n);
        assert!(cfg.is_feasible(), "auto_select({m},{k},{n}) 产出不可行配置 {}", cfg.name());
    }
}

// ═══════════════════════════════════════════════════════════════
// 组 5：AQL/HSA 事实 [HSA 规格 grid 单位=work-item + 41.1TF 伪影回归锁]
// ═══════════════════════════════════════════════════════════════

/// [HSA+GPU] grid 按 kernel 真实覆盖（tile_n）分配，与 n_col_passes 无关——
/// 旧版按 effective_n 分配 → 2p 配置一半输出未写（sentinel 实证 50%）
#[test]
fn compute_grid_uses_tile_n_not_effective_n() {
    let mut cfg1 = GemmConfig::tile_128x64_k16(); cfg1.n_col_passes = 1;
    let mut cfg2 = GemmConfig::tile_128x64_k16(); cfg2.n_col_passes = 2;
    // 内核体不读 n_col_passes → 两种 cfg 的覆盖与 grid 必须完全一致
    let (g1x, g1y) = gemm_gen::compute_grid_auto(&cfg1, 256, 256);
    let (g2x, g2y) = gemm_gen::compute_grid_auto(&cfg2, 256, 256);
    assert_eq!((g1x, g1y), (g2x, g2y),
        "grid 必须只依赖 tile_n（kernel 实际覆盖），不得随 n_col_passes 虚标");
    // HSA：grid 单位是 work-item → grid_x 必须是 wg_size 整数倍且覆盖 N
    assert_eq!(g1x % cfg1.wg_size, 0, "grid_x 必须是 wg_size 整数倍（work-item 单位）");
    assert!(g1x / cfg1.wg_size * cfg1.tile_n >= 256, "grid 必须覆盖 N=256");
}

/// [LLVM getKernArgSegmentSize 类比] kernarg builder 字节数 == kernel 声明数——
/// builder↔声明漂移是本会话两类 kernarg 断言失败的根因类（回归锁）
#[test]
fn kernarg_builder_matches_kernel_declaration() {
    // unpad：5 参（ptr+ptr+u32+u32+u32）= 28B
    let unpad = t0_gpu::t0::math::t0_unpad_2d();
    assert_eq!(unpad.kernarg_size(), 28, "t0_unpad_2d 声明应 28B");
    // gemm_gen：7 参 = 40B；builder 同步
    let g = gemm_gen::generate(&GemmConfig::tile_128x64_k16());
    assert_eq!(g.kernarg_size(), 40, "gemm_gen 128x64 声明应 40B");
    assert_eq!(gemm_gen::build_kernargs(0, 0, 0, 64, 64, 64, &GemmConfig::tile_128x64_k16()).len(),
        g.kernarg_size() as usize, "gemm_gen builder↔声明漂移");
    // tile_ir：8 参（含 M）= 44B；builder 同步
    let spec = TileGemm::tile_64x64_k16();
    let t = tile_ir::lower_gemm(&spec);
    assert_eq!(t.kernarg_size(), 44, "tile_ir lower_gemm 声明应 44B（含 M）");
    assert_eq!(tile_ir::build_kernargs_m(0, 0, 0, 64, 64, 64, &spec).len(),
        t.kernarg_size() as usize, "tile_ir builder↔声明漂移");
}

// ═══════════════════════════════════════════════════════════════
// 组 5b：厂家规格核算锁 [RX 9060 XT 官方规格 @3130MHz]
//   WMMA 吞吐 32 cycles/条（FP16 f32_16x16x16 = 8192 FLOP）→ 103.6 TF
//   数据宽度减半吞吐翻倍：FP8/INT8=2×FP16=205，INT4/稀疏=4×=410/821
#[test]
fn vendor_spec_tflops_self_consistent() {
    let simd: u64 = 128;                 // 32 CU × 4
    let clk: u64 = 3130 * 1_000_000;     // boost MHz
    let wmma_flop: u64 = 16 * 16 * 16 * 2; // 8192
    // 厂家标称反推吞吐周期（103 为取整标称：精确值 31.9 ≈ 32）
    let fp16_cyc = simd * wmma_flop * clk / (103 * 1_000_000_000_000);
    assert!((31..=32).contains(&fp16_cyc),
        "FP16 矩阵标称 ↔ ~32 cycles/WMMA @3130MHz，实测 {fp16_cyc}");
    // 宽度减半吞吐翻倍规律：FP8/INT8 标称 205 TF → cycles = FP16 的一半（±2）
    let fp8_cyc = simd * wmma_flop * clk / (205 * 1_000_000_000_000);
    let int4_cyc = simd * wmma_flop * clk / (410 * 1_000_000_000_000);
    assert!((14..=18).contains(&fp8_cyc), "FP8 cycles 应为 FP16 的一半（≈16），实测 {fp8_cyc}");
    assert!((7..=9).contains(&int4_cyc), "INT4 cycles 应为 FP16 的 1/4（≈8），实测 {int4_cyc}");
    assert!((fp8_cyc * 2 - 1..=fp8_cyc * 2 + 1).contains(&fp16_cyc),
        "FP8 = 2×FP16 吞吐（fp16={fp16_cyc} fp8={fp8_cyc}，取整边界）");
    assert!((int4_cyc * 2 - 1..=int4_cyc * 2 + 1).contains(&fp8_cyc),
        "INT4 = 2×FP8（fp8={fp8_cyc} int4={int4_cyc}）");
    // 正向核算：32 cyc 吞吐 → 103.6 TF（厂家标称 103 取整）
    let tf_at_32cyc = simd * wmma_flop * clk / 32 / 1_000_000_000_000;
    assert!((102..=104).contains(&tf_at_32cyc), "32cyc 吞吐核算 = {tf_at_32cyc} TF（标称 103 取整）");
}

// ═══════════════════════════════════════════════════════════════
// 组 6：ELF/CodeObject 事实 [ROCm 工具链兼容]
//   HSACO 必须含 NT_AMD note 的 target 描述符（.amdgpu_metadata 生成）——
//   KFD 裸加载不读 note，但 ROCr/rocprof 等标准工具链需要。
// ═══════════════════════════════════════════════════════════════

#[test]
fn gfx1200_elf_contains_target_note() {
    let mut k = T0Kernel::new("isa_elf_note");
    k.endpgm();
    let elf = k.compile(Target::GFX1200).expect("compile");
    let needle = b"amdgcn-amd-amdhsa--gfx1200";
    assert!(elf.windows(needle.len()).any(|w| w == needle),
        "ELF 缺 target 描述符 note（.amdgpu_metadata 未生成？）");
}

// ═══════════════════════════════════════════════════════════════
// 组 7：GFX1100 (RDNA3) 对称锁 [LLVM 对照 + 双架构回归]
//   同一 IR 在两代架构的指令形式差异（caps 表的输出层回归锁）
// ═══════════════════════════════════════════════════════════════

/// [LLVM] GFX1100：等待用统一 s_waitcnt vmcnt（无拆分形式）、vmem 也是 global
///（RDNA3 无 GFX1200 的拆分 waitcnt 指令）
#[test]
fn gfx1100_uses_unified_waitcnt_and_global_vmem() {
    let asm = probe_kernel_asm_on(Target::GFX1100);
    assert!(asm.contains("global_load_b32"), "GFX1100 vmem 同样是 global_load:\n{asm}");
    assert!(asm.contains("s_waitcnt vmcnt(0)"), "GFX1100 等待必须统一 s_waitcnt vmcnt:\n{asm}");
    assert!(!asm.contains("s_wait_loadcnt"), "s_wait_loadcnt 是 RDNA4 拆分形式，GFX1100 不应出现:\n{asm}");
}

/// [LLVM] GFX1100：原子返回用 glc（RDNA3 形式；GFX1200 是 th 字段——互为镜像）
#[test]
fn gfx1100_atomic_rtn_uses_glc() {
    let mut k = T0Kernel::new("isa_atomic_glc");
    // [A 类家族] 地址对登记 count≥2 并定义两半
    let addr = k.alloc_vreg_array(2, t0_gpu::t0::ir::Alignment::Align2);
    k.v_mov_imm(addr, 0);
    k.v_mov_imm(VReg(addr.0 + 1), 0);
    k.push(Op::GlobalAtomicAddU32Rtn { dst: VReg(4), addr, src: VReg(2) });
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1100).expect("asm");
    assert!(asm.contains(" glc"), "GFX1100 原子返回必须用 glc:\n{asm}");
    assert!(!asm.contains("th:TH_ATOMIC_RETURN"), "GFX1100 无 th 字段:\n{asm}");
}

/// [ROCm ELF 契约] e_flags：GFX1200=0x048 / GFX1100=0x041
///（ELF64 header 偏移 48；rdna3_code_object e_flags 分支回归锁）
#[test]
fn elf_e_flags_per_arch() {
    let read_eflags = |t: Target| {
        let mut k = T0Kernel::new("isa_eflags");
        k.endpgm();
        let elf = k.compile(t).expect("compile");
        u32::from_le_bytes(elf[48..52].try_into().expect("e_flags"))
    };
    assert_eq!(read_eflags(Target::GFX1200), 0x048, "GFX1200 e_flags");
    assert_eq!(read_eflags(Target::GFX1100), 0x041, "GFX1100 e_flags");
}

// ── 共享构建器 ──

/// 最小探针内核 ASM（global load/store + wait + gid：覆盖主要发射路径）
fn probe_kernel_asm() -> String {
    probe_kernel_asm_on(Target::GFX1200)
}

fn probe_kernel_asm_on(target: Target) -> String {
    let mut k = T0Kernel::new("isa_probe");
    k.set_wg_size(256);
    let out = k.arg_ptr("out");
    let n = k.arg_u32("n");
    k.emit_arg_loads();
    let gid = k.compute_global_id_x(256);
    let nv = k.alloc_vreg();
    k.v_mov_from_sgpr(nv, n);
    let saved = k.bounds_check_begin(gid, nv);
    let off = k.alloc_vreg();
    k.v_lshlrev_b32(off, 2, gid);
    let (a, ah) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(a, t0_gpu::t0::ir::SReg(out.0));
    k.v_mov_from_sgpr(ah, t0_gpu::t0::ir::SReg(out.0 + 1));
    k.addr64_add(a, ah, off);
    let v = k.alloc_vreg();
    k.global_load(v, a, Width::B32, 0);
    k.wait_vmcnt(0);
    k.global_store(a, v, Width::B32, 0);
    k.wait_vscnt(0);
    k.bounds_check_end(saved);
    k.endpgm();
    k.to_assembly(target).expect("probe asm")
}

/// GEMM 内核 ASM（bulk 128x64：已验证路径的发射扫描面）
fn gemm_asm_tile_128x64_k16() -> String {
    let spec = TileGemm::tile_128x64_k16();
    let kernel = tile_ir::lower_gemm(&spec);
    kernel.to_assembly(Target::GFX1200).expect("gemm asm")
}
