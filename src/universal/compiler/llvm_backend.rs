use crate::universal::core::Arch;
use crate::universal::compiler::CompiledKernel;
use crate::universal::compiler::{CompilerBackend, IsaEncoder};
use std::process::Command;

// ═══════════════════════════════════════════════════════
// LLVM 后端 — 通过外部 llc 编译
// ═══════════════════════════════════════════════════════
//
// 路径: T0 IR → LLVM IR 文本 → llc → 目标汇编 → as → ELF
//
// 优势: 零 LLVM 库依赖, 只需要系统安装的 llc
// 劣势: 编译延迟 ~50ms (进程启动开销)

pub struct LlvmBackend {
    llc_path: String,
    as_path: String,
}

impl LlvmBackend {
    pub fn new() -> Self {
        // 优先用 ROCm 的 LLVM
        let llc_path = if std::path::Path::new("/opt/rocm/llvm/bin/llc").exists() {
            "/opt/rocm/llvm/bin/llc".to_string()
        } else {
            "llc".to_string() // fallback 到系统 llc
        };

        let as_path = if std::path::Path::new("/opt/rocm/llvm/bin/llvm-mc").exists() {
            "/opt/rocm/llvm/bin/llvm-mc".to_string()
        } else {
            "llvm-mc".to_string()
        };

        Self { llc_path, as_path }
    }

    /// 检查 llc 是否可用
    pub fn is_available(&self) -> bool {
        Command::new(&self.llc_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// 目标三元组
    fn target_triple(arch: Arch) -> &'static str {
        match arch {
            Arch::Gfx1100 => "amdgcn-amd-amdhsa",
            Arch::Gfx1200 | Arch::Gfx1201 => "amdgcn-amd-amdhsa",
            Arch::Gfx942 | Arch::Gfx950 => "amdgcn-amd-amdhsa",
            Arch::Sm80 | Arch::Sm86 | Arch::Sm89 => "nvptx64-nvidia-cuda",
            Arch::Sm90 | Arch::Sm100 => "nvptx64-nvidia-cuda",
            _ => "amdgcn-amd-amdhsa",
        }
    }

    /// CPU 目标
    fn cpu_features(arch: Arch) -> &'static str {
        match arch {
            Arch::Gfx1100 => "gfx1100",
            Arch::Gfx1200 => "gfx1200",
            Arch::Gfx1201 => "gfx1201",
            Arch::Gfx942 => "gfx942",
            Arch::Gfx950 => "gfx950",
            Arch::Sm80 => "sm_80",
            Arch::Sm86 => "sm_86",
            Arch::Sm89 => "sm_89",
            Arch::Sm90 => "sm_90",
            Arch::Sm100 => "sm_100",
            _ => "gfx1200",
        }
    }

    /// 生成 LLVM IR 文本
    fn generate_llvm_ir(&self, kernel: &super::KernelIr) -> String {
        let mut ir = String::new();

        // 模块头
        ir.push_str("; ModuleID = 't0_kernel'\n");
        ir.push_str("target datalayout = \"e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8\"\n");
        ir.push_str("target triple = \"amdgcn-amd-amdhsa\"\n\n");

        // kernel 函数
        ir.push_str(&format!("define amdgpu_kernel void @{}(", kernel.name));

        // 参数
        for (i, arg) in kernel.args.iter().enumerate() {
            if i > 0 { ir.push_str(", "); }
            let ty = if arg.is_ptr { "ptr addrspace(1)" } else { "i32" };
            ir.push_str(&format!("{} %arg_{}", ty, i));
        }
        ir.push_str(") {\nentry:\n");

        // 操作
        for (i, op) in kernel.ops.iter().enumerate() {
            match op {
                super::IrOp::Add { dst, src0, src1 } => {
                    ir.push_str(&format!("  %v{} = fadd float %v{}, %v{}\n", dst, src0, src1));
                }
                super::IrOp::Mul { dst, src0, src1 } => {
                    ir.push_str(&format!("  %v{} = fmul float %v{}, %v{}\n", dst, src0, src1));
                }
                super::IrOp::Fma { dst, src0, src1, src2 } => {
                    ir.push_str(&format!("  %v{}_mul = fmul float %v{}, %v{}\n", dst, src0, src1));
                    ir.push_str(&format!("  %v{} = fadd float %v{}_mul, %v{}\n", dst, dst, src2));
                }
                super::IrOp::Load { dst, base, offset } => {
                    ir.push_str(&format!("  %v{} = load float, ptr addrspace(1) %v{}\n", dst, base));
                }
                super::IrOp::Store { base, offset, src } => {
                    ir.push_str(&format!("  store float %v{}, ptr addrspace(1) %v{}\n", src, base));
                }
                super::IrOp::Barrier => {
                    ir.push_str("  call void @llvm.amdgcn.s.barrier()\n");
                }
                super::IrOp::EndPgm => {
                    // AMDGPU: kernel 结尾不需要特殊指令
                }
            }
        }

        ir.push_str("  ret void\n");
        ir.push_str("}\n\n");

        // 外部声明
        ir.push_str("declare void @llvm.amdgcn.s.barrier() #0\n");
        ir.push_str("attributes #0 = { convergent nounwind }\n");

        ir
    }

    /// 编译 LLVM IR → 目标汇编
    fn compile_ir_to_asm(&self, ir_text: &str, arch: Arch) -> Result<String, String> {
        let triple = Self::target_triple(arch);
        let cpu = Self::cpu_features(arch);

        // 写 IR 到临时文件
        let ir_path = "/tmp/t0_kernel.ll";
        std::fs::write(ir_path, ir_text)
            .map_err(|e| format!("Write IR failed: {}", e))?;

        // llc 编译
        let output = Command::new(&self.llc_path)
            .args(&[
                "-mtriple", triple,
                "-mcpu", cpu,
                "-filetype=asm",
                "-o", "/tmp/t0_kernel.s",
                ir_path,
            ])
            .output()
            .map_err(|e| format!("llc failed: {}", e))?;

        if !output.status.success() {
            return Err(format!("llc error: {}", String::from_utf8_lossy(&output.stderr)));
        }

        std::fs::read_to_string("/tmp/t0_kernel.s")
            .map_err(|e| format!("Read asm failed: {}", e))
    }

    /// 编译汇编 → ELF
    fn compile_asm_to_elf(&self, arch: Arch) -> Result<Vec<u8>, String> {
        let triple = Self::target_triple(arch);
        let cpu = Self::cpu_features(arch);

        let output = Command::new(&self.as_path)
            .args(&[
                "-triple", triple,
                "-mcpu", cpu,
                "-filetype=obj",
                "-o", "/tmp/t0_kernel.o",
                "/tmp/t0_kernel.s",
            ])
            .output()
            .map_err(|e| format!("llvm-mc failed: {}", e))?;

        if !output.status.success() {
            return Err(format!("llvm-mc error: {}", String::from_utf8_lossy(&output.stderr)));
        }

        std::fs::read("/tmp/t0_kernel.o")
            .map_err(|e| format!("Read ELF failed: {}", e))
    }
}

impl CompilerBackend for LlvmBackend {
    fn compile(&self, ir: &super::KernelIr, target: Arch) -> Result<CompiledKernel, String> {
        // 1. 生成 LLVM IR
        let llvm_ir = self.generate_llvm_ir(ir);

        // 2. llc 编译
        let _asm_text = self.compile_ir_to_asm(&llvm_ir, target)?;

        // 3. llvm-mc 汇编 → ELF
        let elf_bytes = self.compile_asm_to_elf(target)?;

        Ok(CompiledKernel {
            elf_bytes,
            name: ir.name.clone(),
            vgpr_count: 0,  // TODO: 从 ELF 解析
            sgpr_count: 0,
            lds_size: 0,
            scratch_size: 0,
            kernarg_size: ir.args.iter().map(|a| if a.is_ptr { 8 } else { 4 }).sum(),
            workgroup_size: ir.workgroup_size,
            target,
        })
    }

    fn supports(&self, target: Arch) -> bool {
        match target {
            Arch::Gfx1100 | Arch::Gfx1200 | Arch::Gfx1201 |
            Arch::Gfx942 | Arch::Gfx950 => true, // AMDGPU
            Arch::Sm80 | Arch::Sm86 | Arch::Sm89 |
            Arch::Sm90 | Arch::Sm100 => true,    // NVPTX
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "LLVM (external llc)"
    }
}
