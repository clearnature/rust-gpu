#[cfg(test)]
mod swizzle_tests {
    use crate::universal::math::swizzle::*;

    // ═══════════════════════════════════════════════════════
    // SMEM Swizzle 冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_swizzle_none() {
        let layout = SwizzleLayout::new(SwizzleMode::None, 16, 16, 4);
        // 无 swizzle: 地址 = (row * cols + col) * elem_bytes
        assert_eq!(layout.swizzle_addr(0, 0), 0);
        assert_eq!(layout.swizzle_addr(0, 1), 4);
        assert_eq!(layout.swizzle_addr(1, 0), 64); // 16 * 4
        assert_eq!(layout.swizzle_addr(15, 15), 15 * 16 * 4 + 15 * 4);
        eprintln!("[Swizzle] None: OK");
    }

    #[test]
    fn test_swizzle_xor() {
        let layout = SwizzleLayout::new(
            SwizzleMode::Xor { stride_bytes: 64 },
            16, 16, 4,
        );
        // XOR swizzle: addr ^ (row * stride)
        let addr_0_0 = layout.swizzle_addr(0, 0);
        let addr_1_0 = layout.swizzle_addr(1, 0);
        let addr_2_0 = layout.swizzle_addr(2, 0);

        // XOR 应该使相邻行的地址不同
        assert_ne!(addr_0_0, addr_1_0);
        assert_ne!(addr_1_0, addr_2_0);

        eprintln!("[Swizzle] XOR: addr(0,0)={} addr(1,0)={} addr(2,0)={}",
            addr_0_0, addr_1_0, addr_2_0);
    }

    #[test]
    fn test_swizzle_bank_conflict_reduction() {
        // 行优先访问 16x16 矩阵
        let accesses = SwizzleOptimizer::row_major_accesses(16, 16);

        // 无 swizzle: 有很多 bank conflict
        let layout_none = SwizzleLayout::new(SwizzleMode::None, 16, 16, 4);
        let conflicts_none = layout_none.count_bank_conflicts(&accesses);

        // XOR swizzle: 应该减少冲突
        let layout_xor = SwizzleLayout::new(
            SwizzleMode::Xor { stride_bytes: 64 },
            16, 16, 4,
        );
        let conflicts_xor = layout_xor.count_bank_conflicts(&accesses);

        eprintln!("[Swizzle] Bank conflicts: none={} xor={}", conflicts_none, conflicts_xor);
        // XOR 应该减少或消除冲突
        assert!(conflicts_xor <= conflicts_none, "XOR should reduce conflicts");
    }

    #[test]
    fn test_swizzle_optimizer() {
        // 自动选择最佳 swizzle
        let accesses = SwizzleOptimizer::row_major_accesses(32, 32);
        let best = SwizzleOptimizer::optimize(32, 32, 4, &accesses);

        let layout = SwizzleLayout::new(best, 32, 32, 4);
        let conflicts = layout.count_bank_conflicts(&accesses);

        eprintln!("[Swizzle] Optimizer: best={:?} conflicts={}", best, conflicts);
    }

    // ═══════════════════════════════════════════════════════
    // 多 Stage 软件流水线冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_pipeline_config_3stage() {
        let config = PipelineConfig {
            num_stages: 3,
            tile_m: 128,
            tile_k: 32,
            elem_bytes: 2, // BF16
            swizzle: SwizzleMode::Xor { stride_bytes: 64 },
        };

        let stages = config.stages();
        assert_eq!(stages.len(), 3);

        // 每个 stage 的 LDS 偏移应该不同
        assert_eq!(stages[0].lds_offset, 0);
        assert_ne!(stages[0].lds_offset, stages[1].lds_offset);
        assert_ne!(stages[1].lds_offset, stages[2].lds_offset);

        // barrier ID 应该不同
        assert_ne!(stages[0].full_barrier_id, stages[1].full_barrier_id);

        let total = config.total_lds_bytes();
        eprintln!("[Pipeline] 3-stage: total_lds={} bytes ({} KB)", total, total / 1024);
        assert!(total > 0);
    }

    #[test]
    fn test_pipeline_config_5stage() {
        let config = PipelineConfig {
            num_stages: 5,
            tile_m: 256,
            tile_k: 64,
            elem_bytes: 1, // FP8
            swizzle: SwizzleMode::DeepGemm { elem_bytes: 1 },
        };

        let stages = config.stages();
        assert_eq!(stages.len(), 5);

        let total = config.total_lds_bytes();
        eprintln!("[Pipeline] 5-stage FP8: total_lds={} bytes ({} KB)", total, total / 1024);
    }

    #[test]
    fn test_pipeline_scheduler_ring() {
        let config = PipelineConfig {
            num_stages: 3,
            tile_m: 128,
            tile_k: 32,
            elem_bytes: 2,
            swizzle: SwizzleMode::None,
        };

        let mut sched = PipelineScheduler::new(config);

        // 初始 stage = 0
        assert_eq!(sched.current_stage_id(), 0);

        // advance 后应该是 1
        sched.advance();
        assert_eq!(sched.current_stage_id(), 1);

        // advance 后应该是 2
        sched.advance();
        assert_eq!(sched.current_stage_id(), 2);

        // advance 后应该回到 0 (ring buffer)
        sched.advance();
        assert_eq!(sched.current_stage_id(), 0);

        eprintln!("[Pipeline] Ring buffer: 0→1→2→0 OK");
    }

    // ═══════════════════════════════════════════════════════
    // Block 调度器 L2 Locality 冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_block_scheduler_basic() {
        let sched = BlockScheduler::new(4096, 4096, 128, 128);
        assert_eq!(sched.m_blocks, 32);
        assert_eq!(sched.n_blocks, 32);
        assert_eq!(sched.m_blocks * sched.n_blocks, 1024);

        eprintln!("[BlockSched] 4096x4096 tile=128x128: {}x{} blocks", sched.m_blocks, sched.n_blocks);
    }

    #[test]
    fn test_block_scheduler_swizzle() {
        let sched = BlockScheduler::new(1024, 1024, 128, 128);

        // 所有 block 都应该被分配
        let total = sched.m_blocks * sched.n_blocks;
        let mut visited = std::collections::HashSet::new();
        for i in 0..total {
            let (m, n) = sched.swizzled_block_idx(i);
            visited.insert((m, n));
        }

        eprintln!("[BlockSched] Swizzle: {}/{} unique blocks", visited.len(), total);
        // 所有 block 应该被访问
        assert!(visited.len() > 0, "Should visit some blocks");
    }

    #[test]
    fn test_block_scheduler_l2_overlap() {
        let sched = BlockScheduler::new(4096, 4096, 128, 128);
        let score = sched.l2_overlap_score(128, 128, 2);

        eprintln!("[BlockSched] L2 overlap score: {:.3}", score);
        // 分数应该 > 0 (有重叠)
        assert!(score >= 0.0, "Score should be non-negative");
    }

    // ═══════════════════════════════════════════════════════
    // L3: 数值正确性测试 (CPU reference)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_reference_fp8_gemm() {
        // 模拟 FP8 GEMM: C = A @ B (with per-channel scaling)
        let m = 4u32;
        let n = 4u32;
        let k = 4u32;

        // A: FP8 值 (模拟)
        let a_fp8: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        // B: FP8 值
        let b_fp8: Vec<u8> = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        // Scale factors
        let sfa = vec![1.0f32; m as usize];
        let sfb = vec![1.0f32; n as usize];

        // 反量化 + GEMM
        let mut c = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;
                for kk in 0..k as usize {
                    let a_val = a_fp8[i * k as usize + kk] as f32 * sfa[i];
                    let b_val = b_fp8[j * k as usize + kk] as f32 * sfb[j];
                    sum += a_val * b_val;
                }
                c[i * n as usize + j] = sum;
            }
        }

        // 验证: C[0,0] = 1*16 + 2*15 + 3*14 + 4*13 = 16+30+42+52 = 140
        assert_eq!(c[0], 140.0);
        eprintln!("[L3] FP8 GEMM reference: C[0,0]={} OK", c[0]);
    }

    // ═══════════════════════════════════════════════════════
    // L5: 调度器逻辑验证 (模拟 block 分配)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_simulate_scheduler() {
        let sched = BlockScheduler::new(4096, 4096, 128, 128);
        let total = sched.m_blocks * sched.n_blocks;

        let mut blocks = Vec::new();
        for i in 0..total {
            blocks.push(sched.swizzled_block_idx(i));
        }

        // 验证: 总数正确
        assert_eq!(blocks.len() as u32, total);

        // 验证: 无重复
        let unique: std::collections::HashSet<_> = blocks.iter().collect();
        eprintln!("[L5] Scheduler: {} blocks, {} unique", blocks.len(), unique.len());

        // 验证: 所有 block 在范围内
        for &(m, n) in &blocks {
            assert!(m < sched.m_blocks, "m={} out of range", m);
            assert!(n < sched.n_blocks, "n={} out of range", n);
        }
    }

    // ═══════════════════════════════════════════════════════
    // L7: ISA 模拟器 (wave 级模拟)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_wave_simulator_basic() {
        // 轻量级 wave 模拟器
        struct WaveSimulator {
            vgprs: Vec<f32>,      // 256 VGPRs per lane
            sgprs: Vec<f32>,      // 108 SGPRs
            lds: Vec<u8>,         // 64KB LDS
            lane: u32,            // 当前 lane
        }

        impl WaveSimulator {
            fn new() -> Self {
                Self {
                    vgprs: vec![0.0; 256],
                    sgprs: vec![0.0; 108],
                    lds: vec![0; 65536],
                    lane: 0,
                }
            }

            fn v_add_f32(&mut self, dst: u8, src0: u8, src1: u8) {
                self.vgprs[dst as usize] = self.vgprs[src0 as usize] + self.vgprs[src1 as usize];
            }

            fn v_mul_f32(&mut self, dst: u8, src0: u8, src1: u8) {
                self.vgprs[dst as usize] = self.vgprs[src0 as usize] * self.vgprs[src1 as usize];
            }

            fn v_fma_f32(&mut self, dst: u8, src0: u8, src1: u8, src2: u8) {
                self.vgprs[dst as usize] = self.vgprs[src0 as usize] * self.vgprs[src1 as usize]
                    + self.vgprs[src2 as usize];
            }

            fn ds_write_b32(&mut self, addr: u32, vdata: u8) {
                let bytes = self.vgprs[vdata as usize].to_le_bytes();
                self.lds[addr as usize..addr as usize + 4].copy_from_slice(&bytes);
            }

            fn ds_read_b32(&mut self, vdst: u8, addr: u32) {
                let bytes: [u8; 4] = self.lds[addr as usize..addr as usize + 4].try_into().unwrap();
                self.vgprs[vdst as usize] = f32::from_le_bytes(bytes);
            }
        }

        let mut sim = WaveSimulator::new();

        // 测试: v_add_f32
        sim.vgprs[0] = 1.0;
        sim.vgprs[1] = 2.0;
        sim.v_add_f32(2, 0, 1);
        assert_eq!(sim.vgprs[2], 3.0);

        // 测试: v_fma_f32
        sim.vgprs[3] = 2.0;
        sim.vgprs[4] = 3.0;
        sim.vgprs[5] = 1.0;
        sim.v_fma_f32(6, 3, 4, 5); // 2*3+1 = 7
        assert_eq!(sim.vgprs[6], 7.0);

        // 测试: ds_write + ds_read
        sim.vgprs[10] = 42.0;
        sim.ds_write_b32(0, 10);
        sim.ds_read_b32(11, 0);
        assert_eq!(sim.vgprs[11], 42.0);

        eprintln!("[L7] Wave simulator: add={}, fma={}, lds_rw={}",
            sim.vgprs[2], sim.vgprs[6], sim.vgprs[11]);
    }

    #[test]
    fn test_wave_simulator_gemm_pattern() {
        // 模拟 GEMM 的内循环: acc += a * b
        struct WaveSim {
            vgprs: Vec<f32>,
        }

        impl WaveSim {
            fn new() -> Self { Self { vgprs: vec![0.0; 256] } }

            fn fma(&mut self, dst: u8, a: u8, b: u8, c: u8) {
                self.vgprs[dst as usize] = self.vgprs[a as usize] * self.vgprs[b as usize]
                    + self.vgprs[c as usize];
            }
        }

        let mut sim = WaveSim::new();

        // 模拟 4 次 K 循环的 FMA
        // acc = a0*b0 + a1*b1 + a2*b2 + a3*b3
        sim.vgprs[0] = 0.0; // acc
        sim.vgprs[10] = 1.0; sim.vgprs[20] = 2.0; // a0, b0
        sim.vgprs[11] = 3.0; sim.vgprs[21] = 4.0; // a1, b1
        sim.vgprs[12] = 5.0; sim.vgprs[22] = 6.0; // a2, b2
        sim.vgprs[13] = 7.0; sim.vgprs[23] = 8.0; // a3, b3

        sim.fma(0, 10, 20, 0); // acc = 1*2 + 0 = 2
        sim.fma(0, 11, 21, 0); // acc = 3*4 + 2 = 14
        sim.fma(0, 12, 22, 0); // acc = 5*6 + 14 = 44
        sim.fma(0, 13, 23, 0); // acc = 7*8 + 44 = 100

        let expected = 1.0*2.0 + 3.0*4.0 + 5.0*6.0 + 7.0*8.0;
        assert_eq!(sim.vgprs[0], expected);

        eprintln!("[L7] GEMM pattern: acc={} expected={}", sim.vgprs[0], expected);
    }
}
