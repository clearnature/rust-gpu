# geisYaO 全部博文完整存档（19篇）

> 来源: https://www.zhihu.com/people/geisyao/posts
> 存档日期: 2026-08-22
> 硬件: AMD RX 9070 XT (GFX1201, RDNA4) / RX 7900 XTX (GFX1100, RDNA3)
> 项目: t0-gpu — 零依赖裸金属 GPU 编译器 + Ignis 推理引擎

---

## 第一阶段：RDNA3 起步（2026年3月）

### 1. 600 行 Rust 击败 AMD 官方 GEMM 库
**日期**: 2026-03-23 | **赞同**: 126 | **收藏**: 230

600 行 Rust 参数化 GEMM 内核生成器，在 3 个矩阵尺寸上超越 rocBLAS，最高领先 42%。
通过 KFD 驱动直接通信，零外部依赖。

| 矩阵尺寸 | T0 (TFLOPS) | rocBLAS | 比例 |
|----------|-------------|---------|------|
| 1024×1024×1024 | 45.40 | 27.89 | 163% |
| 1024×1024×4096 | 58.84 | 29.94 | 197% |
| 4096×4096×4096 | 67.30 | 58.72 | 115% |

---

### 2. 从零到 79 TFLOPS：一周密集开发实录
**日期**: 2026-03-29 | **赞同**: 126 | **收藏**: 230

从 GPU 硬挂 20+ 次到稳定 79.2 TFLOPS 的完整历程。

**13 条铁律**（每条都是 GPU 硬挂换来的）:
1. LICM 必须 insert(len-1) — hoisted 指令放在 terminator 前
2. CSE key 必须含 opcode + MVal + inline 常量
3. CSE 必须在 barrier 处清空 seen table
4. DCE 必须处理 loop-carried deps
5. Scheduling 必须在 regalloc 之后
6. max_vgprs 不要人为限制 — GEMM 需要 200+
7. raw_asm 是绕过 regalloc 的定时炸弹
8. KFD VA 复用必须用 BufferPool
9. tile_ir 内核必须 skip_optimize
10. coop load chunks_per_row 必须是 2^n
11. CWSR 与 WGP mode 不兼容
12. VGPR 上限 254 — 255/256 触发 CWSR hang
13. SIGPIPE 必须 ignore — 防管道杀进程泄漏队列

---

## 第二阶段：GPU MODE Hackathon（2026年4月）

### 3. 一个AMD粉丝的 MXFP4 GEMM 优化之旅
**日期**: 2026-04-06 | **赞同**: 97 | **收藏**: 113

MI355X (CDNA4) 上手写 FP4 融合 GEMM 内核，10.8μs。17 条死路验证。
关键发现: MFMA 16×16×128 (65536 FLOPs/insn) vs WMMA 16×16×16 (8192 FLOPs/insn)。

### 4. GPU MODE Hackathon 比赛最后一天
**日期**: 2026-04-07 | **赞同**: 11

顿悟: 用 T0 裸金属方式重做。4 小时搭出 GFX950 完整工具链（ISA 编码器 + ELF 生成器 + HIP 加载）。
12 个 ISA bug，每个平均 15 分钟解决。

### 5. 修复 AITER ↔ Triton MXFP4 集成链路
**日期**: 2026-04-13 | **赞同**: 13

三层接口断裂: (1) 打包维度不匹配 (2) Scale 形状变更 (3) float8_e8m0fnu 被拒绝。
修复仅 10 行代码。上游 PR: AITER #2704, Triton #10009。

---

## 第三阶段：GFX1200 RDNA4 适配（2026年4月）

### 6. 48 小时，80+ 次重启，从零写一个 GPU 驱动
**日期**: 2026-04-12 | **赞同**: 139 | **收藏**: 137

RX 9070 XT 到货后 48 小时内从零实现 AM 用户态驱动。

**13 个 Bug 清单**:
| # | Bug | 调试时间 |
|---|-----|---------|
| 1 | KFD MES v2 不支持 >2 WG | 12h |
| 2 | PSP_RING_TYPE=1 (应为2) | 6h |
| 3 | 12 个 GFX_FW_TYPE 常量错误 | 2h |
| 4 | 3 个 GFX_CMD_ID 常量错误 | 1h |
| 5 | PspGfxCmdResp struct 错误 | 1h |
| 6 | NBIF v6.3.1 未识别 | 0.5h |
| 7 | MMIO_REG_HOLE_OFFSET 变更 | 0.5h |
| 8 | HDP REMAP 未初始化 | 1h |
| 9 | Discovery 表重启丢失 | 0.5h |
| 10 | 冷启动缺 SOC/MMHUB/IH | 2h |
| 11 | Reset 逻辑误判 | 1h |
| 12 | CP_MEC_RS64_CNTL 位错误 | 1h |
| 13 | GRBM_GFX_CNTL BASE_IDX=0 (应为1) | 4h |

### 7. 从零开始的 AMD GPU 上裸金属编程
**日期**: 2026-04-09 | **赞同**: 94 | **收藏**: 164

完整教程: /dev/kfd → ISA 编码 → ELF 构建 → AQL 队列 → GPU 执行。
**s_waitcnt 三类策略**: 全等(~20TF) → 批量等待(~60TF) → Let Data Fly(79TF)。

### 8. CWSR Trap Handler Bug 发现与修复
**日期**: 2026-04-15 | **赞同**: 13

VCC 写入导致 GPU hang。根因: CWSR trap handler .asm/.h 不同步。
AMD drm-next 已修复 (commit 911e2c05) 但未 backport 到 ROCm 7.2.1。
手动 cherry-pick + DKMS 重编译修复。

### 9. SP3→GAS 转换器修复 CWSR 打包问题
**日期**: 2026-04-18 | **赞同**: 7

1300 行 Rust SP3→GAS 转换器。8 轮迭代: ~450 错误 → 0。
SP3 宏语言: 变量 + 函数 + for循环 + 条件。Pass 3 做 11 类 GAS 语法转换。

### 10. SFENCE 救不了你 — WC Store Buffer 陷阱
**日期**: 2026-04-19 | **赞同**: 12 | **收藏**: 29

**问题一**: SWMMAC K=32 实际 dense K=16（2:4 稀疏语义）
**问题二**: WC store buffer 间歇性数据损坏

**五阶段模型**: CPU core → WC combine → Root Complex → PCIe fabric → GPU VRAM
MFENCE 只到阶段③，GPU 需要到阶段⑤。

**三层修复**: write() + readback, zero() volatile, BufferPool drain

### 11. Linux 内核不 Flush GPU TLB
**日期**: 2026-04-21 | **赞同**: 11 | **收藏**: 22

**问题一**: kfd_flush_tlb_after_unmap() 只覆盖 CDNA，RDNA 全系不 flush GPU TLB
**问题二**: s2 vs ttmp9 — AMDHSA ABI 规范确认 s2/s3/s4 对所有架构通用
**问题三**: cargo test 并行 → 多 AQL queue 竞争 → 硬 hang

**铁律**: BufferPool 复用 / s2=workgroup_id / GPU 测试串行

---

## 第四阶段：推理引擎（2026年4-5月）

### 12. 朋友，你也许会需要一点"量化"
**日期**: 2026-04-27 | **赞同**: 3

Prefill 赢 (125 tok/s vs 82), decode 输 (9 tok/s vs 86)。
结论: decode 是 memory-bound, 量化(INT4)在带宽节省上全面优于 2:4 稀疏。

### 13. 29 行文本，65 TFLOPS
**日期**: 2026-04-28 | **赞同**: 21 | **收藏**: 47

AI (Claude) 从 parser 源码推断 T0 IR 语法规则，一次性生成 29 行 .t0 内核。
46ms 编译，64.7 TFLOPS。零外部依赖。

---

## 第五阶段：性能优化（2026年5月）

### 14. RX 9070 XT rocBLAS BF16 GEMM 性能基线
**日期**: 2026-05-03 | **赞同**: 7

| 尺寸 | TFLOPS | 配置 |
|------|--------|------|
| 1024³ | 88.4 | ROCBLAS_USE_HIPBLASLT=0 + TuneableOp |
| 2048³ | 112.1 | 同上 |
| 4096³ | 124.7 | 同上 |
| 1024×4096² | 118.0 | 同上 |

hipBLASLt 在 GFX1201 上比纯 rocBLAS 慢 28%。tuning 时触发 GPU reset。

### 15. 从 GEMM 起步到 llama.cpp decode 持平
**日期**: 2026-05-13 | **赞同**: 9 | **收藏**: 21

**8 个性能杠杆**:
1. 多波 kv-split GEMV (mw4_unroll8): decode +3%
2. residual_add + rmsnorm + GEMV 融合: decode 88.6→102.7 (+15.9%)
3. ELF 磁盘缓存: prefill 590→213ms (-64%)
4. 消除冗余 VRAM zero: prefill -60ms
5. 测量公式 fix
6. prefill 队列化 dispatch: pp9 274→123ms (-55%)
7. select_gemm_spec 缓存: pp9 123→34ms (-72%)
8. 公平对照 vs llama.cpp -fa 1: decode 持平

**公平对照**: pp9+tg100 反超 -2.4%, pp57+tg100 +6.0%, pp193+tg100 +38.1%

### 16. NT vs NN 布局对标乌龙
**日期**: 2026-05-30 | **赞同**: 5

| 布局 | T0 | rocBLAS |
|------|-----|---------|
| NT (推理 Linear) | ~130 TF | ~112-115 TF |
| NN (经典 BLAS) | ~85 TF | ~124 TF |

NT vs NT: 领先 13-16%。追了半个月的"7%差距"从来没存在过。

### 17. SWMMAC 指令探索
**日期**: 2026-05-01 | **赞同**: 12 | **收藏**: 16

rocBLAS 在 RDNA4 上 100% 使用 WMMA，零使用 SWMMAC。
SWMMAC 对 LLM 推理实用价值有限: PPL +35 (Wanda), INT4 量化全面优于 2:4 稀疏。

---

## 第六阶段：多租户（2026年5月）

### 18. 把一张 16 GB RX 9070 XT 切成 4 个 LLM serving 容器
**日期**: 2026-05-25 | **赞同**: 10 | **收藏**: 12

4 容器 Qwen3-1.7B BF16: 每 tenant 452 MB (从 7184 MB 降至 -93.7%)。
dma-buf 跨进程零拷贝共享权重。内核 patch: VRAM 硬配额 + 协作式自评 queue。
~150 行 C 内核 patch。

---

## 第七阶段：硬件问题（2026年8月）

### 19. ASUS ProArt 声卡失声
**日期**: 2026-05-30 | **赞同**: 2

AMD Adrenalin 驱动捆绑的 SoundWire 音频驱动与 OEM 驱动抢同一 ACP 硬件。
不影响 GPU 编程，纯硬件兼容性记录。

---

## 技术铁律汇总

| # | 铁律 | 来源 |
|---|------|------|
| 1 | SWMMAC ≠ Dense WMMA, wmma_k=16 | SFENCE篇 |
| 2 | SFENCE 不保证 PCIe 可见性 | SFENCE篇 |
| 3 | 不要用 CPU WC 写零 GPU 输出缓冲区 | SFENCE篇 |
| 4 | 缓冲区复用需要 WC drain | SFENCE篇 |
| 5 | KFD UNMAP 不 flush GPU TLB (RDNA全系) | TLB篇 |
| 6 | Workgroup ID 寄存器对所有架构通用 (s2) | TLB篇 |
| 7 | GPU 测试必须串行执行 | TLB篇 |
| 8-13 | LICM/CSE/DCE/scheduling/raw_asm/CWSR | 79TF篇 |
| 14 | tile_ir 内核必须 skip_optimize | 79TF篇 |
| 15 | coop load chunks_per_row 必须是 2^n | 79TF篇 |

## 性能里程碑

| 日期 | 里程碑 | 硬件 |
|------|--------|------|
| 2026-03-23 | 67.3 TF GEMM, 7/9 超越 rocBLAS | RX 7900 XTX |
| 2026-03-29 | 79.2 TF, 匹敌 Triton | RX 7900 XTX |
| 2026-04-12 | 首次 RDNA4 PM4 dispatch | RX 9070 XT |
| 2026-04-28 | 65 TF, 29行AI生成内核 | RX 9070 XT |
| 2026-05-01 | rocBLAS 基线 124.7 TF | RX 9070 XT |
| 2026-05-13 | decode 持平 llama.cpp -fa 1 | RX 9070 XT |
| 2026-05-25 | 4容器多租户 LLM serving | RX 9070 XT |
| 2026-05-30 | NT 130 TF 超越 rocBLAS 13% | RX 9070 XT |
