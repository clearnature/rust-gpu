// ═══════════════════════════════════════════════════════
// 架构与厂商定义
// ═══════════════════════════════════════════════════════

/// GPU 厂商
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vendor {
    AMD,
    NVIDIA,
    Huawei,
    MooreThreads,
    Biren,
    Intel,
    Unknown,
}

/// GPU 架构
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    // AMD RDNA (消费级)
    Gfx1100, // RDNA3 (RX 7900 XTX)
    Gfx1200, // RDNA4 (RX 9060 XT)
    Gfx1201, // RDNA4 (RX 9700)
    // AMD CDNA (数据中心)
    Gfx942,  // CDNA3 (MI300X)
    Gfx950,  // CDNA4 (MI350)
    // NVIDIA
    Sm80,    // Ampere (A100)
    Sm86,    // Ampere (RTX 3090)
    Sm89,    // Ada (RTX 4090)
    Sm90,    // Hopper (H100)
    Sm100,   // Blackwell (B200)
    // 华为
    AscendC64,
    AscendC68,
    // 通用
    Unknown,
}

impl Arch {
    /// 是否为 AMD 架构
    pub fn is_amd(&self) -> bool {
        matches!(self, Self::Gfx1100 | Self::Gfx1200 | Self::Gfx1201 | Self::Gfx942 | Self::Gfx950)
    }

    /// 是否为 NVIDIA 架构
    pub fn is_nvidia(&self) -> bool {
        matches!(self, Self::Sm80 | Self::Sm86 | Self::Sm89 | Self::Sm90 | Self::Sm100)
    }

    /// Wave/Warp 大小
    pub fn wave_size(&self) -> u32 {
        match self {
            Self::Gfx1100 | Self::Gfx1200 | Self::Gfx1201 => 32, // Wave32
            Self::Gfx942 | Self::Gfx950 => 64,                    // Wave64
            Self::Sm80 | Self::Sm86 | Self::Sm89 | Self::Sm90 | Self::Sm100 => 32, // Warp32
            _ => 32,
        }
    }

    /// 对应的厂商
    pub fn vendor(&self) -> Vendor {
        if self.is_amd() { Vendor::AMD }
        else if self.is_nvidia() { Vendor::NVIDIA }
        else { Vendor::Unknown }
    }
}

/// 数据类型
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    FP4E2M1,
    U32,
    U16,
    U8,
    I32,
    I16,
    I8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::U16 | Self::I16 => 2,
            Self::FP8E4M3 | Self::FP8E5M2 | Self::U8 | Self::I8 => 1,
            Self::FP4E2M1 => 1, // packed 2 per byte
        }
    }
}
