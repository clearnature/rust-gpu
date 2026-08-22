# geisYaO 博文存档 — t0-gpu / Ignis 项目

> 来源: https://www.zhihu.com/people/geisyao/posts
> 存档日期: 2026-08-22
> 硬件: AMD RX 9070 XT (GFX1201) / RX 9060 XT (GFX1200)

---

## 1. NT vs NN 布局对标乌龙

**标题**: 我们以为差 rocBLAS 7%，其实反超了 13%，那代价是什么呢？

**核心结论**:
- 我的编译器做的是 NT（y = x·Wᵀ，推理 Linear 层），一直拿 rocBLAS 的 NN（A·B）当标尺
- rocBLAS 自己的 NN 比 NT 快约 12%，NN 是它的"满血形态"
- 正确对比 NT vs NT: 我 130 TF vs rocBLAS-NT 112-115 TF → 领先 13-16%

**关键数据 (4096³ BF16 GEMM)**:

| 布局 | T0 编译器 | rocBLAS + TunableOps |
|------|----------|---------------------|
| NT (推理 Linear) | ~130 TF | ~112-115 TF |
| NN (经典 BLAS) | ~85 TF | ~124 TF |

**时钟发现**: RX 9070 XT 满载撞 340W 功耗墙，核心温度才 ~59°C。不同 kernel 在同一功耗墙下维持的时钟不同。

---

## 2. 4 容器虚拟化

**标题**: 把一张 16 GB 的 RX 9070 XT 切成 4 个 LLM serving 容器

**核心成果**:
- 4 个独立 Qwen3-1.7B BF16 推理容器同时运行
- 每 tenant 只占 452 MB 显存（原来 7,184 MB → -93.7%）
- dma-buf 跨进程零拷贝共享权重
- 内核 patch ~150 行 C 加 per-process VRAM 硬配额

**三个内核 patch**:
1. `AMDKFD_IOC_SET_VRAM_CAP` (0x27) — per-process VRAM 硬配额
2. `AMDKFD_IOC_EVICT_PROCESS_QUEUES` (0x28) — 协作式自评 queue
3. sysfs `vram_cap_<gpu_id>` — 对称读

**已知硬限**:
- L2 cache 无法分区（硅级）
- 内存带宽无法分区（硅级）
- MES page fault 后不可恢复（固件级）

---

## 3. GEMM→GEMV 优化路线

**标题**: 从 GEMM 起步到 llama.cpp decode 持平: 一段裸金属推理框架的优化路线

**5 个优化 Lever**:

| # | 优化 | 效果 |
|---|------|------|
| 1 | 多波 kv-split GEMV (mw4_unroll8) | o_proj -15%, decode +3% |
| 2 | residual_add + rmsnorm + GEMV 融合 | 88.6→102.7 tok/s (+15.9%) |
| 3 | ELF 磁盘缓存 | prefill 590→213 ms (-64%) |
| 4 | 消除冗余 VRAM zero | prefill -60 ms |
| 5 | 测量公式 bug fix | N=5 cold-start 修正 |

**公平对比 (llama.cpp -fa 1)**:

| 场景 | T0 | llama.cpp -fa 1 | Δ |
|------|-----|-----------------|---|
| pp9 + tg100 | 898 ms | 920 ms | **-2.4% 反超** |
| pp57 + tg100 | 976 ms | 921 ms | +6.0% |
| pp193 + tg100 | 1233 ms | 893 ms | +38.1% |

**关键发现**: decode 打平 llama.cpp -fa 1，prefill 短 prompt 已反超。

---

## 4. rocBLAS 性能基线

**标题**: RX 9070 XT (GFX1201) rocBLAS BF16 GEMM 性能基线

**4096³ BF16 NN 对比**:

| 配置 | TFLOPS | 状态 |
|------|--------|------|
| hipBLASLt 默认 | 92.8 | ✅ 但慢 |
| rocBLAS fallback 默认 | 118.9 | ✅ |
| **rocBLAS + TuneableOp tuning** | **124.7** | ✅ **推荐** |
| hipBLASLt + TuneableOp | — | ❌ GPU reset |

**核心结论**: ROCm 7.2 的 hipBLASLt 在 GFX1201 上既不稳定（tuning crash）也不快（比纯 Tensile 慢 28%）。纯 rocBLAS/Tensile 路径最优。

---

## 5. SWMMAC 指令探索

**标题**: RDNA4 SWMMAC 指令探索：消费级 GPU 上的结构化稀疏推理实验

**核心结论**: SWMMAC 在消费级 RDNA4 上对 LLM 推理实用价值有限。

| 方案 | 权重带宽节省 | PPL 代价 | 部署复杂度 |
|------|------------|---------|-----------|
| SWMMAC 2:4 稀疏 | 50% | +35 (Wanda) | 剪枝+微调+vidx |
| **INT4 量化 (GPTQ)** | **75%** | **+1** | **一步量化** |

**关键发现**:
- rocBLAS 在 RDNA4 上 100% 使用 WMMA，零使用 SWMMAC
- hipSPARSELt 不支持 RDNA4 消费级
- SiLU 激活天然稀疏性 < 1.2%，无法有效利用 2:4 结构
- FSR 4 是 SWMMAC 在消费级 RDNA4 上的唯一落地场景

---

## 6. 零依赖 GPU 编译器

**标题**: 29 行文本，65 TFLOPS：一个零依赖 GPU 编译器的故事

**核心成果**:
- 29 行 .t0 文本 → 46ms 编译 → 64.7 TFLOPS
- 零外部依赖（不依赖 LLVM/HIP/ROCm 运行时）
- AI 一次性生成内核代码，无需人工修改

**对比**:

| | Triton (AMD) | rocBLAS | T0 |
|--|-------------|---------|-----|
| LLVM 依赖 | ✅ | N/A | ❌ 零 |
| 编译时间 | JIT 多阶段 | N/A | 46ms |
| 安装体积 | 数 GB | 数 GB | 单个二进制 |
| AI 可生成 | 部分 | ❌ | ✅ 设计目标 |

**编译管线**: .t0 → Parser(2837行) → T0 IR → SSA提升 → 13个优化Pass → 寄存器分配 → 二进制发射 → ELF Code Object → .hsaco

---

## 作者信息

- **Zhihu**: https://www.zhihu.com/people/geisyao
- **项目**: t0-gpu (Rust 裸金属 KFD GPU 编译器) + Ignis (推理引擎)
- **硬件**: AMD RX 9070 XT (GFX1201) + RX 9060 XT (GFX1200)
- **技术栈**: 纯 Rust，零外部依赖，直接走 Linux KFD 调度
