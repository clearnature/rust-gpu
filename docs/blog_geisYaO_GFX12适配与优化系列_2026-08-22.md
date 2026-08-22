# geisYaO GFX12 适配与优化系列 — 完整技术存档

> 来源: https://www.zhihu.com/people/geisyao/posts
> 存档日期: 2026-08-22
> 覆盖范围: GFX1200/RDNA4 裸金属适配的 4 篇系列文章 + 性能优化 + rocBLAS 基线 + AM 驱动

---

## 一、GFX12 适配系列（4 篇）

### 第一篇：48 小时，80+ 次重启，从零写一个 GPU 驱动

**核心事件**: 2026-04-10~12，RX 9070 XT 到货后 48 小时内从零实现 AM 用户态驱动。

**13 个 Bug 清单**:

| # | Bug | 影响 | 调试时间 |
|---|-----|------|---------|
| 1 | KFD MES v2 不支持 >2 WG dispatch | KFD 路径完全不可用 | 12h |
| 2 | PSP_RING_TYPE = 1 (应为 2) | PSP ring 不响应 | 6h |
| 3 | 12 个 GFX_FW_TYPE 常量值错误 | 固件加载失败 | 2h |
| 4 | 3 个 GFX_CMD_ID 常量值错误 | RLC autoload 失败 | 1h |
| 5 | PspGfxCmdResp struct layout 错误 | 命令格式错 | 1h |
| 6 | NBIF v6.3.1 (hw_id=108) 未识别 | NBIO 基址全零 | 0.5h |
| 7 | MMIO_REG_HOLE_OFFSET GFX11→GFX12 | HDP flush 失效 | 0.5h |
| 8 | HDP REMAP 未初始化 | PSP DMA 看不到数据 | 1h |
| 9 | Discovery 表重启后丢失 | 二次启动失败 | 0.5h |
| 10 | 冷启动缺 SOC/MMHUB/IH | BL not ready | 2h |
| 11 | Reset 逻辑对冷启动误判 | BL hang | 1h |
| 12 | CP_MEC_RS64_CNTL 位位置错误 | MEC 命令无效 | 1h |
| 13 | GRBM_GFX_CNTL BASE_IDX=0 (应为 1) | HQD 全部失效 | 4h |

**关键结论**: KFD MES v2 在 ≥4 WG 时必 hang。tinygrad AM driver 是唯一公开的用户态 GPU 驱动实现。

---

### 第二篇：续篇 — SP3→GAS 转换器修复 CWSR

**核心问题**: 所有写 VCC 的指令都会让 GPU 挂死。
**根因**: AMD 驱动的 CWSR trap handler 的 .asm 和 .hex 文件不同步。
**修复**: 用 Rust 写了 SP3→GAS 转换器 (1300 行)，DKMS rebuild。

---

### 第三篇：再续篇 — SFENCE 救不了你

**问题一: SWMMAC K 值语义误解**

`v_swmmac_f32_16x16x32_bf16` 中的 32 是**稀疏 K 维度**（2:4 结构化稀疏），dense K = 16。
A 操作数只有 4 VGPRs = 16 bf16 → 不可能一次处理 32 个 dense K 元素。

**铁律**: GFX12 做 dense BF16 GEMM，`wmma_k` 必须设为 16。每 K=32 块需要 2× SWMMAC。

**问题二: WC Store Buffer 导致间歇性数据损坏**

五阶段模型: CPU core → WC combine buffer → Root Complex → PCIe fabric → GPU VRAM

`SFENCE/MFENCE` 只保证到 Root Complex（阶段③），不保证 GPU 已收到（阶段⑤）。

**修复三层**:
1. `GpuBuffer.write()` — 写入后 `read_volatile(host_ptr)` drain
2. `GpuBuffer.zero()` — volatile 写零 + readback
3. `BufferPool.alloc()` — 复用时 readback drain

**铁律**: SFENCE 不保证 PCIe 可见性。必须用 non-posted read 作为屏障。

---

### 第四篇：又续篇 — Linux 内核不 Flush 你的 GPU TLB

**问题一: KFD UNMAP 不 Flush GPU TLB**

`kfd_flush_tlb_after_unmap()` 只覆盖 CDNA (GFX9.4.x)。RDNA 全系（GFX10/11/12）不匹配。

故障链: alloc → dispatch → drop → UNMAP (TLB 不 flush) → VA 复用 → GPU TLB hit 旧映射 → page fault → 硬 hang

**修复**: BufferPool 复用 buffer（不频繁 UNMAP）。内核 patch: 增加 `IP_VERSION(12, 0, 0)` 条件。

**问题二: s2 vs ttmp9**

AMDHSA ABI 规范: 所有架构通用，workgroup ID 固定在 s2/s3/s4。ttmp 是 trap handler 临时寄存器。

**问题三: cargo test 并行 → 多 KFD Queue 竞争 → 硬 Hang**

**修复**: 全局 GPU 测试互斥锁。`cargo test --test-threads=1`。

**铁律**:
- KFD UNMAP 不 flush GPU TLB（RDNA 全系）
- Workgroup ID 寄存器对所有架构通用 (s2)
- GPU 测试必须串行执行

---

## 二、SWMMAC 指令探索

**核心结论**: rocBLAS 在 RDNA4 上 100% 使用 WMMA，零使用 SWMMAC。

**SWMMAC 对 LLM 推理的局限**:
- 权重后剪枝精度损失过大（PPL +35）
- SiLU/GeLU 激活天然零值不足 1%
- INT4 量化在带宽节省和精度保持上全面优于 2:4 稀疏
- hipSPARSELt 不支持 RDNA4

**SWMMAC 本质**: 带宽加速（少读 50% 权重），不是算力加速。

**对编译器的指导**: WMMA 是 dense GEMM 正确选择。SWMMAC 仅用于 FSR 4。

---

## 三、性能优化路线

### GEMM 起点: 85.47 TF (RX 9070 XT, 4096³ NT BF16)

| Backend | TFLOPS | autotune? |
|---------|--------|-----------|
| **T0 静态 selector** | **85.47** | 否 |
| hipBLASLt heuristic | 77.54 | 否 |
| Triton fixed config | 63.73 | 否 |
| **rocBLAS + TuneableOps** | **113.19** | 是 |

### 8 个性能杠杆

| # | Lever | 效果 |
|---|-------|------|
| 1 | 多波 kv-split GEMV (mw4_unroll8) | o_proj -15%, decode +3% |
| 2 | residual_add + rmsnorm + GEMV 融合 | decode 88.6→102.7 tok/s (+15.9%) |
| 3 | ELF 磁盘缓存 | prefill 590→213 ms (-64%) |
| 4 | 消除冗余 VRAM zero | prefill -60 ms |
| 5 | 测量公式 bug fix | cold-start 数据可信 |
| 6 | prefill 队列化 dispatch | pp9 274→123 ms (-55%) |
| 7 | select_gemm_spec 缓存 | pp9 123→34 ms (-72%) |
| 8 | 公平对照 vs llama.cpp -fa 1 | decode 持平 |

### 公平对照结果 (llama.cpp -fa 1)

| Scenario | Ours | llama -fa 1 | Δ |
|----------|------|-------------|---|
| pp9 + tg100 | 898 ms | 920 ms | **-2.4% 反超** |
| pp57 + tg100 | 976 ms | 921 ms | +6.0% |
| pp193 + tg100 | 1233 ms | 893 ms | +38.1% |

**结论**: decode 持平，short prompt 反超，long prompt 差距在 kernel 层面。

### 方法论
- 测量先于优化
- 小步快验（token bit-exact 对照）
- 用对的指标看对的层
- 冗余 work 是最容易找的 lever
- **对照基线不对 = 把指南针校错了方向**

---

## 四、rocBLAS 性能基线

**RX 9070 XT, ROCm 7.2.1, 纯 rocBLAS (ROCBLAS_USE_HIPBLASLT=0) + TuneableOp**:

| 尺寸 | TFLOPS | ms/iter |
|------|--------|---------|
| 1024³ | 88.4 | 0.024 |
| 2048³ | 112.1 | 0.153 |
| **4096³** | **124.7** | 1.102 |
| 1024×4096² | 118.0 | 0.291 |

**关键发现**: hipBLASLt 在 GFX1201 上比纯 rocBLAS 慢 28%。tuning 时触发 GPU reset。

**环境变量推荐**:
```
ROCBLAS_USE_HIPBLASLT=0
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=1
```

---

## 五、对 t0-gpu 项目的指导意义

| 文章内容 | 对应代码 | 行动项 |
|----------|---------|--------|
| WC readback drain | `kfd/mod.rs` write()/zero() | ✅ 已修复 |
| wmma_k=16 | `tile_ir.rs` k_sub_steps() | ✅ 已正确 (tile_k/16) |
| BufferPool 复用 | `kfd/mod.rs` BufferPool | ✅ 已实现 |
| GPU 测试互斥 | `tile_ir.rs` test modules | ⚠️ 需检查 |
| KFD TLB 不 flush | 高频 alloc/free 场景 | ✅ BufferPool 避免 |
| rocBLAS 124.7 TF | 4096³ 目标上限 | 🎯 K6 persistent 冲刺目标 |
| WMMA 不是 SWMMAC | BF16 GEMM 指令选择 | ✅ 已正确 |
