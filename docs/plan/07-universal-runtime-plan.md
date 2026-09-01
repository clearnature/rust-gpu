# 通用 GPU 运行时综合路线图

> 生成日期: 2026-08-24

## 当前资产盘点

### t0-gpu 已有
| 组件 | 文件 | 行数 | 状态 |
|------|------|------|------|
| KFD 运行时 | src/kfd/mod.rs | 3235 | ✅ VRAM/AQL/doorbell |
| GFX1100 ISA 编码器 | src/rdna3_asm.rs | ~2000 | ✅ 完整 RDNA3 |
| GFX1200 ISA 扩展 | rdna3_asm.rs 内 | ~500 | ✅ RDNA4 移植完成 |
| HSA ELF 生成 | src/rdna3_code_object.rs | ~1500 | ✅ Code object |
| T0 编译器 | src/t0/*.rs | ~5000 | ✅ DSL→SSA→ISA |
| GEMM JIT | src/t0/auto_gemm.rs | ~1000 | ✅ 超越 rocBLAS |
| ML Kernel 集合 | src/t0/*_kernels.rs | ~3000 | ✅ Attention/Softmax/RMSNorm |
| ignis 框架 | src/ignis/ | ~3000 | ✅ Tensor/Autograd/Optimizer |
| wmma_db | src/wmma_db.rs | ~500 | ✅ WMMA 指令数据库 |

### sass-assembler 可复用
| 组件 | 状态 | 可用性 |
|------|------|--------|
| Pascal SASS 编码器 | ✅ 22 opcode 家族, cuobjdump 验证 | 直接用 |
| Volta SASS 编码器 | ✅ 128-bit, 65 条指令 | 可用 |
| Ampere SASS 编码器 | ✅ FP16/Barriers | 可用 |
| Hopper/Blackwell | ⚠️ 骨架 | 需补 |
| ILP 硬件模型 | ✅ 7 架构延迟表 | 可用 |
| IDeviceBackend 抽象 | ✅ 多态设计 | 可复用 |
| ILP 调度算法 | ⚠️ 基础 RAW 检测 | 需改进 |

### tinygrad 可借鉴
| 组件 | 说明 |
|------|------|
| BEAM search | 高层优化参数搜索 |
| HCQ 抽象 | Hardware Command Queue trait |
| AM 直接 PCI | 绕过 KFD 的更快路径 |
| Memory coalescing | 自动向量化 load/store |
| UOp IR | 更高层的计算图表示 |
| 多后端架构 | 10+ 设备后端的模式 |

## 分阶段路线图

### Phase 0: 准备 (1 周)

- [ ] 从 kfd/mod.rs 提取 `trait GpuDriver` / `trait GpuDevice`
- [ ] 现有 KFD 代码封装为 `AmdDriver` / `AmdDevice`
- [ ] 所有现有测试通过 (验证抽象不破坏现有功能)
- [ ] 建立 docs/plan/ 目录的架构文档 (已完成)

### Phase 1: 数学原语 (4-6 周)

- [ ] `t0-math` crate: 从 ignis 分离数学库
- [ ] half/BF16 host 端类型 (1-2 天)
- [ ] Prefix scan (Blelloch + work-efficient) (1 周)
- [ ] Radix sort (1 周)
- [ ] PRNG (Philox/Xoroshiro + distributions) (1 周)
- [ ] GEMV (矩阵×向量) (3-5 天)
- [ ] Batched GEMM (1 周)

### Phase 2: NVIDIA 后端 (4-6 周)

- [ ] 研究 nvidia.ko ioctl 接口
- [ ] `NvidiaDriver` / `NvidiaDevice` 实现
- [ ] 整合 sass-assembler 的 Pascal/Volta/Ampere 编码器
- [ ] .cubin ELF 生成器
- [ ] 简单 kernel 在 NVIDIA GPU 上跑通

### Phase 3: ILP 调度增强 (3-4 周)

- [ ] 端口冲突检测 (使用已有的 port_mask)
- [ ] WAR/WAW 依赖处理
- [ ] 调度优先级 (关键路径距离)
- [ ] Shared memory bank conflict 感知
- [ ] 寄存器压力感知调度

### Phase 4: BEAM 搜索集成 (2-3 周)

- [ ] 定义优化动作空间 (tiling/unrolling/local/group/TC)
- [ ] 实现 BEAM search 框架 (Rust)
- [ ] 实测性能反馈循环
- [ ] 缓存搜索结果 (disk cache)

### Phase 5: 共享 GPU 调度器 (3-4 周)

- [ ] 时间片调度
- [ ] VRAM 配额管理
- [ ] CU/SM 分区
- [ ] 故障隔离
- [ ] 多进程测试

### Phase 6: 高级 ILP (4-6 周)

- [ ] Software pipelining (循环模调度)
- [ ] Trace scheduling (热路径优先)
- [ ] Instruction clustering
- [ ] Prefetch 插入

### Phase 7: 国产 GPU (按需)

- [ ] 华为昇腾后端
- [ ] 摩尔线程后端
- [ ] 壁仞后端
- [ ] 每个后端 4-8 周

## 关键架构决策

| 决策 | 建议 | 理由 |
|------|------|------|
| ISA 编码 | 双后端 (手写 + LLVM) | 手写快, LLVM 覆盖广 |
| 共享调度 | 用户态 Rust | 零内核修改, 可移植 |
| 多厂商发现 | 运行时 dlopen | 一个 binary 跑所有卡 |
| 内存模型 | 先分离后加 UVA | UVA 跨厂商太难 |
| 数学库 | 独立 crate | 与 ignis 解耦, 跨厂商共享 |
| ILP 调度 | 保留硬件模型, 重写调度器 | 模型数据有价值, 算法需改进 |
| BEAM 搜索 | Rust 重写 | 与现有编译器集成 |

## 优先级排序

```
P0 (立刻):  抽象层提取 → 数学原语 (scan/scan/rand)
P1 (近期):  NVIDIA 后端 → ILP 调度增强
P2 (中期):  BEAM 搜索 → 共享调度器
P3 (远期):  高级 ILP → 国产 GPU
```

## 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| NVIDIA ioctl 文档不足 | 无法实现 NvidiaDriver | 参考 nouveau 驱动源码 |
| sass-assembler 编码器不够完整 | NVIDIA 新架构支持不全 | LLVM fallback |
| BEAM 搜索编译慢 | 用户体验差 | 磁盘缓存 + 预编译 |
| 国产 GPU 驱动不稳定 | 后端质量不可控 | 社区反馈循环 |
| 跨厂商内存模型复杂 | Phase 5 延期 | 先做分离地址空间 |
