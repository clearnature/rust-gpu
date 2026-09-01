//! GPU Hardware Topology Probe
//!
//! Reads AMD KFD sysfs topology to discover GPU hardware parameters.
//! Source: `/sys/class/kfd/kfd/topology/nodes/N/properties`
//!
//! # Example output (RX 9060 XT / gfx1200):
//! ```text
//! GpuProbe {
//!   gfx_target_version: 120000,
//!   simd_count: 64,
//!   simd_per_cu: 2,          → cu_count = 32
//!   lds_size_in_kb: 64,
//!   max_waves_per_simd: 16,
//!   wave_front_size: 32,
//!   max_engine_clk_mhz: 2780,
//!   num_xcc: 1,
//! }
//! ```
//!
//! # Usage
//! ```ignore
//! let probe = GpuProbe::detect()?;
//! println!("{} CUs @ {} MHz", probe.cu_count(), probe.max_engine_clk_mhz);
//! ```

use std::path::Path;

// ============================================================================
// GPU Topology Properties
// ============================================================================

/// Complete GPU hardware topology from KFD sysfs.
///
/// All values are read once at construction time from
/// `/sys/class/kfd/kfd/topology/nodes/N/properties`.
#[derive(Clone, Debug)]
pub struct GpuProbe {
    /// KFD topology node index (1-based; node 0 is CPU)
    pub node: u32,
    /// KFD gpu_id (matches the id used in KFD ioctl)
    pub gpu_id: u32,
    /// GFX target version in decimal (120000 = gfx1200, 110000 = gfx1100)
    pub gfx_target_version: u32,
    /// Total SIMD count across all CUs
    pub simd_count: u32,
    /// SIMD processors per CU (2 for RDNA, 4 for CDNA)
    pub simd_per_cu: u32,
    /// LDS size per CU in KB (64 for RDNA4)
    pub lds_size_in_kb: u32,
    /// GDS size in KB (typically 0)
    pub gds_size_in_kb: u32,
    /// Max waves per SIMD (hardware occupancy limit)
    pub max_waves_per_simd: u32,
    /// Wavefront size (32 for RDNA, 64 for CDNA)
    pub wave_front_size: u32,
    /// Number of SIMD arrays
    pub array_count: u32,
    /// SIMD arrays per engine
    pub simd_arrays_per_engine: u32,
    /// CUs per SIMD array
    pub cu_per_simd_array: u32,
    /// Max scratch slots per CU
    pub max_slots_scratch_cu: u32,
    /// Number of XCC (extreme compute complex) — 1 for single-die GPUs
    pub num_xcc: u32,
    /// Vendor ID (4098 = AMD)
    pub vendor_id: u32,
    /// PCI device ID
    pub device_id: u32,
    /// PCI location ID
    pub location_id: u32,
    /// PCI domain
    pub domain: u32,
    /// DRM render minor (e.g. 128 → /dev/dri/renderD128)
    pub drm_render_minor: u32,
    /// Number of SDMA engines
    pub num_sdma_engines: u32,
    /// Number of SDMA queues per engine
    pub num_sdma_queues_per_engine: u32,
    /// Number of compute queues
    pub num_cp_queues: u32,
    /// CWSR (context save/restore) size in bytes
    pub cwsr_size: u32,
    /// Control stack size in bytes
    pub ctl_stack_size: u32,
    /// Max engine clock for compute (MHz) — from ccompute field
    pub max_engine_clk_ccompute_mhz: u32,
    /// Max engine clock for FP compute (MHz) — from fcompute field
    pub max_engine_clk_fcompute_mhz: u32,
    /// Number of global wave syncs
    pub num_gws: u32,
    /// Firmware version
    pub fw_version: u32,
    /// Capability flags
    pub capability: u32,
}

impl GpuProbe {
    /// Detect the first GPU from KFD topology sysfs.
    ///
    /// Scans nodes 1..=16 (node 0 is always CPU) and returns the first
    /// GPU node found (where gfx_target_version > 0).
    ///
    /// # Errors
    /// Returns Err if no GPU node is found or sysfs is unreadable.
    pub fn detect() -> Result<Self, String> {
        Self::detect_with_range(1..=16)
    }

    /// Detect GPU by specific KFD gpu_id.
    ///
    /// Useful when multiple GPUs are present.
    pub fn detect_by_gpu_id(gpu_id: u32) -> Result<Self, String> {
        for node in 1..=16 {
            let gpu_id_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            let id: u32 = std::fs::read_to_string(&gpu_id_path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            if id == gpu_id {
                return Self::from_node(node).map_err(|e|
                    format!("Failed to read node {}: {}", node, e));
            }
        }
        Err(format!("GPU with gpu_id={} not found in KFD topology", gpu_id))
    }

    /// Detect GPU by PCI location (domain:bus:device.function).
    pub fn detect_by_pci_location(location_id: u32) -> Result<Self, String> {
        for node in 1..=16 {
            if let Ok(probe) = Self::from_node(node) {
                if probe.location_id == location_id {
                    return Ok(probe);
                }
            }
        }
        Err(format!("GPU at PCI location 0x{:08x} not found", location_id))
    }

    /// Scan a range of topology nodes and return the first GPU found.
    fn detect_with_range(range: std::ops::RangeInclusive<u32>) -> Result<Self, String> {
        for node in range {
            let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
            if Path::new(&prop_path).exists() {
                // Check if this is a GPU node (gfx_target_version > 0)
                if let Ok(props) = std::fs::read_to_string(&prop_path) {
                    let gfxv = parse_u32_prop(&props, "gfx_target_version").unwrap_or(0);
                    if gfxv > 0 {
                        return Self::from_node(node).map_err(|e|
                            format!("Failed to parse node {}: {}", node, e));
                    }
                }
            }
        }
        Err("No GPU found in KFD topology".to_string())
    }

    /// Build GpuProbe from a specific topology node.
    pub fn from_node(node: u32) -> Result<Self, String> {
        let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
        let props = std::fs::read_to_string(&prop_path)
            .map_err(|e| format!("Cannot read {}: {}", prop_path, e))?;

        let p = |name: &str| -> u32 {
            parse_u32_prop(&props, name).unwrap_or(0)
        };

        Ok(GpuProbe {
            node,
            gpu_id: {
                let gpu_id_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
                std::fs::read_to_string(&gpu_id_path)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0)
            },
            gfx_target_version: p("gfx_target_version"),
            simd_count: p("simd_count"),
            simd_per_cu: p("simd_per_cu"),
            lds_size_in_kb: p("lds_size_in_kb"),
            gds_size_in_kb: p("gds_size_in_kb"),
            max_waves_per_simd: p("max_waves_per_simd"),
            wave_front_size: p("wave_front_size"),
            array_count: p("array_count"),
            simd_arrays_per_engine: p("simd_arrays_per_engine"),
            cu_per_simd_array: p("cu_per_simd_array"),
            max_slots_scratch_cu: p("max_slots_scratch_cu"),
            num_xcc: p("num_xcc").max(1), // at least 1
            vendor_id: p("vendor_id"),
            device_id: p("device_id"),
            location_id: p("location_id"),
            domain: p("domain"),
            drm_render_minor: p("drm_render_minor"),
            num_sdma_engines: p("num_sdma_engines"),
            num_sdma_queues_per_engine: p("num_sdma_queues_per_engine"),
            num_cp_queues: p("num_cp_queues"),
            cwsr_size: p("cwsr_size"),
            ctl_stack_size: p("ctl_stack_size"),
            max_engine_clk_ccompute_mhz: p("max_engine_clk_ccompute"),
            max_engine_clk_fcompute_mhz: p("max_engine_clk_fcompute"),
            num_gws: p("num_gws"),
            fw_version: p("fw_version"),
            capability: p("capability"),
        })
    }

    // ── Derived Properties ──

    /// Number of Compute Units.
    /// Formula: simd_count / (simd_per_cu * num_xcc)
    pub fn cu_count(&self) -> u32 {
        let denom = self.simd_per_cu.max(1) * self.num_xcc.max(1);
        self.simd_count / denom
    }

    /// GFX target as a human-readable string.
    /// e.g. "gfx1200", "gfx1100"
    pub fn gfx_str(&self) -> String {
        format!("gfx{}", self.gfx_target_version / 100)
    }

    /// Is this an RDNA (wave32) GPU?
    pub fn is_rdna(&self) -> bool {
        self.wave_front_size == 32
    }

    /// Is this a CDNA (wave64) GPU?
    pub fn is_cdna(&self) -> bool {
        self.wave_front_size == 64
    }

    /// Is this GFX1200 (RDNA4)?
    pub fn is_gfx1200(&self) -> bool {
        self.gfx_target_version >= 120000 && self.gfx_target_version < 130000
    }

    /// Is this GFX1100 (RDNA3)?
    pub fn is_gfx1100(&self) -> bool {
        self.gfx_target_version >= 110000 && self.gfx_target_version < 120000
    }

    /// Max waves per CU (hardware occupancy limit).
    /// = max_waves_per_simd * simd_per_cu
    pub fn max_waves_per_cu(&self) -> u32 {
        self.max_waves_per_simd * self.simd_per_cu
    }

    /// Max waves per entire GPU.
    pub fn max_waves_total(&self) -> u32 {
        self.max_waves_per_cu() * self.cu_count()
    }

    /// LDS total per CU in bytes.
    pub fn lds_bytes_per_cu(&self) -> u32 {
        self.lds_size_in_kb * 1024
    }

    /// Scratch backing memory size per CU in bytes.
    /// = max_slots_scratch_cu * wave_front_size * 4 (32-bit per lane)
    pub fn scratch_bytes_per_cu(&self) -> u32 {
        self.max_slots_scratch_cu * self.wave_front_size * 4
    }

    /// DRM render device path (e.g. "/dev/dri/renderD128").
    pub fn drm_render_path(&self) -> String {
        format!("/dev/dri/renderD{}", self.drm_render_minor)
    }

    /// GPU clock in GHz (fcompute field, more representative for shader workloads).
    ///
    /// RX 9060 XT nominal: 2.78 GHz (fcompute), 3.0 GHz (ccompute).
    /// For N14-calibrated peak: 3.15 GHz (boost clock with thermal headroom).
    pub fn gpu_clock_ghz(&self) -> f64 {
        // Use the higher of the two clock fields for peak estimate
        let clk = self.max_engine_clk_fcompute_mhz
            .max(self.max_engine_clk_ccompute_mhz);
        clk as f64 / 1000.0
    }

    /// Peak FP32 FLOPS estimate (theoretical).
    ///
    /// Formula: cu_count * simd_per_cu * wave_front_size * 2 (FMA) * clock_hz
    /// For RX 9060 XT: 32 * 2 * 32 * 2 * 2.78GHz = ~11.4 TFLOPS
    pub fn peak_fp32_tflops(&self) -> f64 {
        let clk_ghz = self.max_engine_clk_fcompute_mhz as f64 / 1000.0;
        let fp32_per_clock = self.cu_count() as f64
            * self.simd_per_cu as f64
            * self.wave_front_size as f64
            * 2.0; // FMA = 2 FLOPS
        fp32_per_clock * clk_ghz / 1000.0 // TFLOPS
    }

    /// VRAM bandwidth estimate in GB/s.
    ///
    /// Not directly available from sysfs — returns 0 if unknown.
    /// RX 9060 XT: ~448 GB/s (GDDR6 128-bit).
    pub fn vram_bandwidth_gbps(&self) -> f64 {
        // Bandwidth is not in KFD topology; use device_id lookup as fallback
        match self.device_id {
            0x7550 => 448.0, // RX 9060 XT (Navi 48)
            0x7448 => 960.0, // RX 7900 XTX (Navi 31)
            0x744C => 800.0, // RX 7900 XT (Navi 31)
            _ => 0.0,
        }
    }

    // ── CWSR Sizing ──

    /// Compute CWSR (context save/restore) sizes.
    ///
    /// Uses the same formula as KFD kernel module:
    ///   wave_num  = cu_num * 32  (RDNA: 32 waves/CU max)
    ///   ctl_stack = PAGE_ALIGN(40 + wave_num * 12 + 8)
    ///   wg_data   = cu_num * (vgpr + sgpr + lds + hwreg)
    ///
    /// Returns (ctx_save_restore_size, ctl_stack_size, wave_num).
    pub fn cwsr_sizes(&self) -> (u32, u32, u32) {
        // Use sysfs values if kernel reports them
        if self.cwsr_size > 0 && self.ctl_stack_size > 0 {
            let wave_num = self.cu_count() * 32;
            return (self.cwsr_size, self.ctl_stack_size, wave_num);
        }

        let cu_num = self.cu_count();
        let wave_num = cu_num * 32; // RDNA: 32 waves per CU

        // Control stack
        let ctl_stack = ((40u64 + wave_num as u64 * 12 + 8 + 4095) / 4096 * 4096) as u32;

        // VGPR allocation per CU (GFX1200/GFX1151: 0x60000 = 384KB)
        let vgpr: u64 = match self.gfx_target_version {
            110001 | 110002 | 120000 | 120001 => 0x60000,
            _ => 0x40000,
        };

        let wg_data = cu_num as u64 * (vgpr + 0x4000 + (self.lds_size_in_kb as u64 * 1024) + 0x1000);
        let wg_data_page = ((wg_data + 4095) / 4096 * 4096) as u32;
        let ctx_save = ctl_stack + wg_data_page;

        (ctx_save, ctl_stack, wave_num)
    }

    // ── Display ──

    /// Print a summary of GPU topology to stderr.
    pub fn dump(&self) {
        eprintln!("═══════════════════════════════════════════════");
        eprintln!("GPU Topology: {} (device_id=0x{:04x})", self.gfx_str(), self.device_id);
        eprintln!("  CUs: {} (simd_count={}, simd_per_cu={})", self.cu_count(), self.simd_count, self.simd_per_cu);
        eprintln!("  Wave: {} lanes, max {} waves/SIMD", self.wave_front_size, self.max_waves_per_simd);
        eprintln!("  Clock: {} MHz (fcompute) / {} MHz (ccompute)",
            self.max_engine_clk_fcompute_mhz, self.max_engine_clk_ccompute_mhz);
        eprintln!("  LDS: {} KB/CU, GDS: {} KB", self.lds_size_in_kb, self.gds_size_in_kb);
        eprintln!("  XCC: {}, Arrays: {}, CUs/Array: {}",
            self.num_xcc, self.array_count, self.cu_per_simd_array);
        eprintln!("  DRM: {} (minor {})", self.drm_render_path(), self.drm_render_minor);
        eprintln!("  SDMA: {} engines, {} queues/engine",
            self.num_sdma_engines, self.num_sdma_queues_per_engine);
        eprintln!("  Peak FP32: {:.1} TFLOPS", self.peak_fp32_tflops());
        let (ctx, ctl, waves) = self.cwsr_sizes();
        eprintln!("  CWSR: ctx={} KB, ctl_stack={} KB, waves={}", ctx / 1024, ctl / 1024, waves);
        eprintln!("═══════════════════════════════════════════════");
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a u32 value from sysfs properties file content.
///
/// The properties file has lines like:
///   simd_count 64
///   gfx_target_version 120000
fn parse_u32_prop(content: &str, name: &str) -> Option<u32> {
    for line in content.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(name) {
            return parts.next()?.parse().ok();
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_u32_prop() {
        let content = "simd_count 64\nsimd_per_cu 2\nlds_size_in_kb 64\n";
        assert_eq!(parse_u32_prop(content, "simd_count"), Some(64));
        assert_eq!(parse_u32_prop(content, "simd_per_cu"), Some(2));
        assert_eq!(parse_u32_prop(content, "lds_size_in_kb"), Some(64));
        assert_eq!(parse_u32_prop(content, "missing"), None);
    }

    #[test]
    fn test_parse_u32_prop_multiline() {
        let content = "\
cpu_cores_count 0
simd_count 64
mem_banks_count 1
caches_count 70
gfx_target_version 120000
max_engine_clk_fcompute 2780
max_engine_clk_ccompute 3000
num_xcc 1
";
        assert_eq!(parse_u32_prop(content, "simd_count"), Some(64));
        assert_eq!(parse_u32_prop(content, "gfx_target_version"), Some(120000));
        assert_eq!(parse_u32_prop(content, "max_engine_clk_fcompute"), Some(2780));
        assert_eq!(parse_u32_prop(content, "num_xcc"), Some(1));
    }

    #[test]
    fn test_gfx_str() {
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 15351808, ctl_stack_size: 16384,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        assert_eq!(probe.gfx_str(), "gfx1200");
        assert_eq!(probe.cu_count(), 32); // 64 / (2 * 1)
        assert!(probe.is_rdna());
        assert!(!probe.is_cdna());
        assert!(probe.is_gfx1200());
        assert!(!probe.is_gfx1100());
        assert_eq!(probe.max_waves_per_cu(), 32); // 16 * 2
        assert_eq!(probe.max_waves_total(), 1024); // 32 * 32
        assert_eq!(probe.drm_render_path(), "/dev/dri/renderD128");
    }

    #[test]
    fn test_cwsr_sizes_with_sysfs_values() {
        // When kernel provides cwsr_size and ctl_stack_size, use those
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 15351808, ctl_stack_size: 16384,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        let (ctx, ctl, waves) = probe.cwsr_sizes();
        assert_eq!(ctx, 15351808);
        assert_eq!(ctl, 16384);
        assert_eq!(waves, 1024); // 32 CUs * 32
    }

    #[test]
    fn test_cwsr_sizes_fallback() {
        // When cwsr_size=0 (not reported by kernel), compute from formula
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 0, ctl_stack_size: 0,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        let (ctx, ctl, waves) = probe.cwsr_sizes();
        assert_eq!(waves, 1024);
        // ctl_stack = PAGE_ALIGN(40 + 1024*12 + 8) = PAGE_ALIGN(12336) = 16384
        assert_eq!(ctl, 16384);
        assert!(ctx > 0);
    }

    #[test]
    fn test_peak_fp32_tflops() {
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 15351808, ctl_stack_size: 16384,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        let tflops = probe.peak_fp32_tflops();
        // 32 CUs * 2 simd * 32 lanes * 2 FMA * 2.78 GHz = 11.38 TFLOPS
        assert!(tflops > 10.0 && tflops < 13.0,
            "Expected ~11.4 TFLOPS, got {:.1}", tflops);
    }

    #[test]
    fn test_gpu_clock_ghz() {
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 15351808, ctl_stack_size: 16384,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        // max(2780, 3000) = 3000 MHz = 3.0 GHz
        assert!((probe.gpu_clock_ghz() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_vram_bandwidth() {
        let probe = GpuProbe {
            node: 1, gpu_id: 1, gfx_target_version: 120000,
            simd_count: 64, simd_per_cu: 2, lds_size_in_kb: 64,
            gds_size_in_kb: 0, max_waves_per_simd: 16, wave_front_size: 32,
            array_count: 4, simd_arrays_per_engine: 2, cu_per_simd_array: 8,
            max_slots_scratch_cu: 32, num_xcc: 1, vendor_id: 4098,
            device_id: 0x7550, location_id: 1024, domain: 0,
            drm_render_minor: 128, num_sdma_engines: 2,
            num_sdma_queues_per_engine: 6, num_cp_queues: 4,
            cwsr_size: 15351808, ctl_stack_size: 16384,
            max_engine_clk_ccompute_mhz: 3000,
            max_engine_clk_fcompute_mhz: 2780,
            num_gws: 64, fw_version: 3390, capability: 0,
        };
        assert_eq!(probe.vram_bandwidth_gbps(), 448.0); // RX 9060 XT
    }

    // Integration test: only runs on machines with actual GPU
    #[test]
    #[ignore] // run with: cargo test -- --ignored test_detect_real_gpu
    fn test_detect_real_gpu() {
        let probe = GpuProbe::detect().expect("No GPU detected");
        probe.dump();
        assert!(probe.simd_count > 0, "simd_count should be > 0");
        assert!(probe.cu_count() > 0, "cu_count should be > 0");
        assert!(probe.gfx_target_version > 0);
    }
}
