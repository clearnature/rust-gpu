use crate::universal::core::Arch;

pub mod llvm_backend;

pub use llvm_backend::LlvmBackend;

// ═══════════════════════════════════════════════════════
// 编译器 trait
// ═══════════════════════════════════════════════════════

/// 已编译的 kernel
pub struct CompiledKernel {
    pub elf_bytes: Vec<u8>,
    pub name: String,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub lds_size: u32,
    pub scratch_size: u32,
    pub kernarg_size: usize,
    pub workgroup_size: (u16, u16, u16),
    pub target: Arch,
}

/// 编译器后端 trait
pub trait CompilerBackend: Send + Sync {
    fn compile(&self, ir: &KernelIr, target: Arch) -> Result<CompiledKernel, String>;
    fn supports(&self, target: Arch) -> bool;
    fn name(&self) -> &str;
}

/// ISA 编码器 trait
pub trait IsaEncoder: Send + Sync {
    fn encode_function(&self, func: &SsaFunc) -> Result<Vec<u8>, String>;
    fn target(&self) -> Arch;
}

// ═══════════════════════════════════════════════════════
// Kernel IR (简化版, 后续扩展)
// ═══════════════════════════════════════════════════════

pub struct KernelIr {
    pub name: String,
    pub ops: Vec<IrOp>,
    pub args: Vec<IrArg>,
    pub workgroup_size: (u16, u16, u16),
    pub lds_size: u32,
}

pub enum IrOp {
    Add { dst: u32, src0: u32, src1: u32 },
    Mul { dst: u32, src0: u32, src1: u32 },
    Fma { dst: u32, src0: u32, src1: u32, src2: u32 },
    Load { dst: u32, base: u32, offset: u32 },
    Store { base: u32, offset: u32, src: u32 },
    Barrier,
    EndPgm,
}

pub struct IrArg {
    pub name: String,
    pub dtype: crate::universal::core::DType,
    pub is_ptr: bool,
}

pub struct SsaFunc {
    pub name: Vec<u8>,
    pub ops: Vec<IrOp>,
}
