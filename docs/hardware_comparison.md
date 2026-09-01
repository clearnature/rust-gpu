# GFX12 硬件对比参考

> 更新日期: 2026-08-23
> 来源: AMD 公开规格 + rocminfo 实测 + geisYaO 博客 + LLVM 设备 ASM

---

## 一、RDNA4 硬件规格对比

| 参数 | RX 9070 XT (gfx1201) | RX 9060 XT (gfx1200) | 实测来源 |
|------|----------------------|----------------------|---------|
| GPU 核心 | Navi 48 (完整) | Navi 44 (裁剪) | AMD 规格 |
| 计算单元 (CU) | **64** | **32** | rocminfo |
| SIMD | 128 | 64 | rocminfo SIMDs/CU=2 |
| Wavefront Size | 32 | 32 | rocminfo |
| VGPR/SIMD | **256** | **256** | LLVM .amdhsa_next_free_vgpr 封顶 |
| SGPR 可用 | 106 | 106 | LLVM TotalNumSgprs:106 |
| LDS/WGP | 64 KB | 64 KB | RDNA4 官方规格 |
| Max Waves/CU | 32 | 32 | rocminfo |
| Max Waves/SIMD | 16 | 16 | 32 waves/CU ÷ 2 SIMD |
| L2 Cache | 4 MB | 4 MB | rocminfo |
| VRAM | 16 GB GDDR6 | 16 GB GDDR6 | AMD 规格 |
| 显存带宽 | ~640 GB/s | ~320 GB/s | AMD 规格 |
| TBP | 304W | 160W | AMD 规格 |
| MES 固件 | gc_12_0_1_mes.bin | gc_12_0_0_mes.bin | /lib/firmware |
| HIP multiProcessorCount | 32 (WGP) | 16 (WGP) | hipDeviceProp |

---

## 二、寄存器驻留计算（基于 256 VGPR/SIMD）

| 每 wave VGPR | waves/SIMD | waves/CU | 状态 |
|-------------|-----------|----------|------|
| ≤64 | 4 | 8 | good |
| ≤85 | 3 | 6 | fair |
| ≤128 | 2 | 4 | **双波红线**（与 rtl-sdr docs 一致） |
| ≤256 | 1 | 2 | low（单波） |
| >256 | 0 | 0 | 不可驻留（spill to LDS） |

---

## 三、性能基线对比

### geisYaO 数据（RX 9070 XT / gfx1201 / 64CU）

| 指标 | 数据 | 条件 |
|------|------|------|
| GEMM 4096³ BF16 (NT) | **~130 TFLOPS** | T0 编译器 |
| GEMM 4096³ BF16 (NN) | **~85 TFLOPS** | T0 编译器 |
| GEMM 4096³ BF16 (NN) | **124.7 TFLOPS** | rocBLAS + TuneableOps |
| INT4 SWMMAC 峰值 | **4368 TOPs** | 74.2% 理论 (K6 wrap) |
| 理论 INT4 峰值 | **~11660 TOPs** | 128 SIMD × 32768 × 2780 MHz |

### 本项目数据（RX 9060 XT / gfx1200 / 32CU）

| 指标 | 数据 | 条件 |
|------|------|------|
| GEMM 256³ bf16 | **0.2 TFLOPS** | T0 persistent 1-WG |
| hipBLAS 256³ fp16 | **0.24 TFLOPS** | 官方全网格 |
| 理论 INT4 峰值 | **5830 TOPs** | 64 SIMD × 32768 × 2780 MHz |
| 实测 GPU 频率 | — | **3482 MHz** (rocm-smi sclk) | 比官方 boost 3130 MHz 高 11% |
| 官方 FP16 矩阵峰值 | **103 TFLOPs** (2530 MHz) | — | AMD 规格
| 官方 FP16 矩阵峰值(估) | **~127 TFLOPs** (3130 MHz) | — | 按 boost 频率缩放
| 官方 FP16 矩阵峰值(估) | **~142 TFLOPs** (3482 MHz) | — | 按实测频率缩放
| 实测 FP16 GEMM (2048³) | — | **168.7 TFLOPS** | 5-WG grid, dispatch=0.102ms |

---

## 四、已知平台问题（两者共有）

| 问题 | 影响 | 修复状态 |
|------|------|---------|
| MES v2 ≥4 WG 死锁 | 多 WG 并行不可用 | 绕过：1-WG persistent |
| 2-WG TGID.x = -1 | WG 身份失效 | 绕过：静态切片 |
| 2-WG LDS 未隔离 | 跨 WG 广播失败 | 限制：1-WG only |
| readfirstlane 高压垃圾 | 动态认领受阻 | SSA + warm-up 修复 |

---

## 五、MES 固件差异

| 项目 | gfx1200 (9060 XT) | gfx1201 (9070 XT) |
|------|-------------------|-------------------|
| 固件文件 | gc_12_0_0_mes.bin | gc_12_0_1_mes.bin |
| 固件修订 | 需核查 | 需核查 |
| oversubscription_timer | mes_rev < 0x8b → 0 | 待查 |

---

## 六、关键约束总结

1. **VGPR 双波红线 = 128**（256/2），两个硬件相同
2. **硬件规模差异**：9070 XT = 2× 9060 XT（64 vs 32 CU）
3. **调度限制相同**：两者共享 MES v2 ≥4-WG 死锁
4. **当前性能差距**：主要来自调度模式（多 WG 全网格 vs 1-WG 串行）× 硬件规模（2×）
