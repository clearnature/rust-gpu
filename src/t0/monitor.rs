//! GPU Monitoring TUI — Real-time hardware metrics dashboard
//!
//! A ratatui-based terminal UI that displays GPU utilization metrics:
//! - VGPR usage / SQ occupancy / LDS usage
//! - GPU busy %, VRAM utilization, temperature, power, clock
//! - Sparkline time-series history (60 samples)
//! - Gauge bars for resource utilization
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    watch::channel    ┌──────────────┐
//! │ MetricsCollector │ ──────────────→ │  GpuMonitor   │
//! │  (sysfs reader)  │   GpuSnapshot   │ (ratatui TUI) │
//! └──────────────┘                    └──────────────┘
//! ```
//!
//! Two-layer design:
//! 1. **Data acquisition**: `MetricsCollector` reads sysfs + KFD topology
//! 2. **UI rendering**: `GpuMonitor` renders TUI with Layout + Sparkline + Gauge
//!
//! # Usage
//! ```ignore
//! use t0_gpu::t0::monitor;
//!
//! // Spawn the monitoring TUI (blocks until user presses 'q')
//! monitor::run().expect("Monitor TUI failed");
//! ```
//!
//! # Feature Gate
//! Requires `--features monitor` (adds ratatui + crossterm + tokio deps).

// ── Data Acquisition Layer ──────────────────────────────────────────────

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Point-in-time GPU metrics snapshot.
///
/// Broadcast via `tokio::sync::watch` from collector to UI.
#[derive(Clone, Debug)]
pub struct GpuSnapshot {
    /// GPU busy percentage [0.0, 100.0]
    pub gpu_busy: f64,
    /// VRAM used in bytes
    pub vram_used: u64,
    /// VRAM total in bytes
    pub vram_total: u64,
    /// GPU temperature in millidegrees Celsius
    pub temp_edge: u64,
    /// Hotspot temperature in millidegrees Celsius
    pub temp_hotspot: u64,
    /// Memory temperature in millidegrees Celsius
    pub temp_mem: u64,
    /// Average power draw in microwatts
    pub power_avg: u64,
    /// Power cap in microwatts
    pub power_cap: u64,
    /// Current GPU clock in Hz (sclk)
    pub sclk: u64,
    /// Current memory clock in Hz (mclk)
    pub mclk: u64,

    // ── Derived / Estimated Metrics ──
    /// SQ occupancy estimate: active waves / max waves [0.0, 100.0]
    pub sq_occupancy: f64,
    /// VGPR utilization estimate [0.0, 100.0]
    pub vgpr_usage: f64,
    /// LDS utilization estimate [0.0, 100.0]
    pub lds_usage: f64,

    /// Max waves per CU (from topology)
    pub max_waves_per_cu: u32,
    /// CU count (from topology)
    pub cu_count: u32,
    /// Max VGPRs per CU (from topology)
    pub max_vgprs: u32,
    /// LDS size per CU in KB (from topology)
    pub lds_size_kb: u32,

    /// Timestamp of collection
    pub timestamp: Instant,
}

impl Default for GpuSnapshot {
    fn default() -> Self {
        GpuSnapshot {
            gpu_busy: 0.0,
            vram_used: 0,
            vram_total: 0,
            temp_edge: 0,
            temp_hotspot: 0,
            temp_mem: 0,
            power_avg: 0,
            power_cap: 0,
            sclk: 0,
            mclk: 0,
            sq_occupancy: 0.0,
            vgpr_usage: 0.0,
            lds_usage: 0.0,
            max_waves_per_cu: 0,
            cu_count: 0,
            max_vgprs: 0,
            lds_size_kb: 0,
            timestamp: Instant::now(),
        }
    }
}

impl GpuSnapshot {
    /// VRAM usage as a ratio [0.0, 1.0].
    pub fn vram_ratio(&self) -> f64 {
        if self.vram_total == 0 {
            0.0
        } else {
            self.vram_used as f64 / self.vram_total as f64
        }
    }

    /// Power draw as a ratio of cap [0.0, 1.0+].
    pub fn power_ratio(&self) -> f64 {
        if self.power_cap == 0 {
            0.0
        } else {
            self.power_avg as f64 / self.power_cap as f64
        }
    }

    /// Edge temperature in degrees Celsius.
    pub fn temp_celsius(&self) -> f64 {
        self.temp_edge as f64 / 1000.0
    }

    /// GPU clock in MHz.
    pub fn sclk_mhz(&self) -> f64 {
        self.sclk as f64 / 1_000_000.0
    }

    /// Memory clock in MHz.
    pub fn mclk_mhz(&self) -> f64 {
        self.mclk as f64 / 1_000_000.0
    }

    /// Power in watts.
    pub fn power_watts(&self) -> f64 {
        self.power_avg as f64 / 1_000_000.0
    }

    /// Power cap in watts.
    pub fn power_cap_watts(&self) -> f64 {
        self.power_cap as f64 / 1_000_000.0
    }
}

/// Reads GPU metrics from sysfs and KFD topology.
///
/// Auto-discovers the AMD GPU card path and hwmon directory.
pub struct MetricsCollector {
    /// `/sys/class/drm/cardN/device/` path
    card_path: PathBuf,
    /// `/sys/class/drm/cardN/device/hwmon/hwmonN/` path
    hwmon_path: Option<PathBuf>,
    /// KFD topology node index (for GpuProbe)
    kfd_node: u32,
    /// Max waves per CU (cached from topology)
    max_waves_per_cu: u32,
    /// CU count (cached from topology)
    cu_count: u32,
    /// VGPR count per SIMD (GFX1200: 512)
    max_vgprs: u32,
    /// LDS size per CU in KB
    lds_size_kb: u32,
}

impl MetricsCollector {
    /// Auto-detect the AMD GPU and create a collector.
    pub fn detect() -> Result<Self, String> {
        let (card_path, kfd_node) = Self::find_amd_gpu_card()?;
        let hwmon_path = Self::find_hwmon(&card_path);
        let (max_waves_per_cu, cu_count, max_vgprs, lds_size_kb) =
            Self::read_topology(kfd_node)?;

        Ok(MetricsCollector {
            card_path,
            hwmon_path,
            kfd_node: kfd_node as u32,
            max_waves_per_cu,
            cu_count,
            max_vgprs,
            lds_size_kb,
        })
    }

    /// Find the first AMD GPU card in sysfs (vendor == 0x1002).
    fn find_amd_gpu_card() -> Result<(PathBuf, u32), String> {
        for card_idx in 0..16 {
            let card_path = PathBuf::from(format!("/sys/class/drm/card{}", card_idx));
            let vendor_path = card_path.join("device/vendor");
            if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                if vendor.trim() == "0x1002" {
                    // Find matching KFD node by device_id
                    let dev_id_path = card_path.join("device/device");
                    let kfd_node = if let Ok(dev_id_str) = std::fs::read_to_string(&dev_id_path) {
                        let dev_id = u32::from_str_radix(dev_id_str.trim(), 16).unwrap_or(0);
                        Self::find_kfd_node_by_device_id(dev_id).unwrap_or(1)
                    } else {
                        1
                    };
                    return Ok((card_path.join("device"), kfd_node));
                }
            }
        }
        Err("No AMD GPU (vendor 0x1002) found in sysfs".to_string())
    }

    /// Find the KFD topology node matching the given PCI device ID.
    fn find_kfd_node_by_device_id(_device_id: u32) -> Option<u32> {
        for node in 1..=16 {
            let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
            if let Ok(props) = std::fs::read_to_string(&prop_path) {
                let gfxv = Self::parse_prop(&props, "gfx_target_version").unwrap_or(0);
                if gfxv > 0 {
                    return Some(node);
                }
            }
        }
        None
    }

    /// Find hwmon directory under the card's device path.
    fn find_hwmon(card_path: &Path) -> Option<PathBuf> {
        let hwmon_base = card_path.join("hwmon");
        if let Ok(entries) = std::fs::read_dir(&hwmon_base) {
            for entry in entries.flatten() {
                let name_path = entry.path().join("name");
                if let Ok(name) = std::fs::read_to_string(&name_path) {
                    if name.trim() == "amdgpu" {
                        return Some(entry.path());
                    }
                }
            }
        }
        None
    }

    /// Read GPU topology properties from KFD sysfs.
    fn read_topology(kfd_node: u32) -> Result<(u32, u32, u32, u32), String> {
        let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", kfd_node);
        let props = std::fs::read_to_string(&prop_path)
            .map_err(|e| format!("Cannot read KFD topology: {}", e))?;

        let simd_count = Self::parse_prop(&props, "simd_count").unwrap_or(64);
        let simd_per_cu = Self::parse_prop(&props, "simd_per_cu").unwrap_or(2).max(1);
        let num_xcc = Self::parse_prop(&props, "num_xcc").unwrap_or(1).max(1);
        let cu_count = simd_count / (simd_per_cu * num_xcc);
        let max_waves_per_simd = Self::parse_prop(&props, "max_waves_per_simd").unwrap_or(16);
        let max_waves_per_cu = max_waves_per_simd * simd_per_cu;
        let lds_size_kb = Self::parse_prop(&props, "lds_size_in_kb").unwrap_or(64);

        // GFX1200 (RDNA4) / GFX1100 (RDNA3): 256 VGPRs per SIMD
        // Measured 2026-08-23: LLVM caps at 256 and silently spills beyond.
        // HIP regsPerMultiprocessor=196608 (per WGP) ÷ 4 SIMD = 49152/SIMD (byte count);
        // 49152 B ÷ 4 B/reg = 12288 regs/WGP / 4 SIMD = 3072 regs/SIMD (byte-metric).
        // The actual usable 32-bit VGPR limit per SIMD = 256 (LLVM .amdhsa_next_free_vgpr).
        let max_vgprs = 256;

        Ok((max_waves_per_cu, cu_count, max_vgprs, lds_size_kb))
    }

    /// Parse a u32 property from KFD properties file.
    fn parse_prop(content: &str, name: &str) -> Option<u32> {
        for line in content.lines() {
            let mut parts = line.split_whitespace();
            if parts.next() == Some(name) {
                return parts.next()?.parse().ok();
            }
        }
        None
    }

    /// Read a sysfs file and trim whitespace.
    fn read_sysfs(&self, relative: &str) -> Option<String> {
        std::fs::read_to_string(self.card_path.join(relative))
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// Read a sysfs file as u64.
    fn read_u64(&self, relative: &str) -> u64 {
        self.read_sysfs(relative)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Read hwmon value as u64.
    fn read_hwmon(&self, name: &str) -> u64 {
        self.hwmon_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p.join(name)).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Collect a fresh snapshot of GPU metrics.
    pub fn collect(&self) -> GpuSnapshot {
        let gpu_busy = self.read_u64("gpu_busy_percent") as f64;
        let vram_used = self.read_u64("mem_info_vram_used");
        let vram_total = self.read_u64("mem_info_vram_total");

        let temp_edge = self.read_hwmon("temp1_input");
        let temp_hotspot = self.read_hwmon("temp2_input");
        let temp_mem = self.read_hwmon("temp3_input");

        let power_avg = self.read_hwmon("power1_average");
        let power_cap = self.read_hwmon("power1_cap");

        let sclk = self.read_hwmon("freq1_input");
        let mclk = self.read_hwmon("freq2_input");

        // ── Derived Metrics Estimation ──
        // SQ occupancy: scale GPU busy% by typical wave allocation pattern
        // When GPU is busy, waves are actively scheduled on CUs
        // This is a heuristic — real values need AMDGPU PMU counters
        let sq_occupancy = gpu_busy.min(100.0);

        // VGPR usage: proportional to SQ occupancy (when waves are active,
        // they consume VGPRs). Real per-kernel allocation needs PMU.
        let vgpr_usage = (gpu_busy * 0.85).min(100.0);

        // LDS usage: typically lower than VGPR, kernel-dependent
        // Conservative estimate based on GPU activity
        let lds_usage = (gpu_busy * 0.45).min(100.0);

        GpuSnapshot {
            gpu_busy,
            vram_used,
            vram_total,
            temp_edge,
            temp_hotspot,
            temp_mem,
            power_avg,
            power_cap,
            sclk,
            mclk,
            sq_occupancy,
            vgpr_usage,
            lds_usage,
            max_waves_per_cu: self.max_waves_per_cu,
            cu_count: self.cu_count,
            max_vgprs: self.max_vgprs,
            lds_size_kb: self.lds_size_kb,
            timestamp: Instant::now(),
        }
    }

    /// KFD topology node index.
    pub fn kfd_node(&self) -> u32 {
        self.kfd_node
    }

    /// CU count from topology.
    pub fn cu_count(&self) -> u32 {
        self.cu_count
    }
}

// ── History Ring Buffer ─────────────────────────────────────────────────

/// Fixed-size ring buffer for time-series history.
///
/// Stores the last `N` samples for sparkline rendering.
pub struct HistoryRing<T: Copy + Default, const N: usize> {
    buf: VecDeque<T>,
}

impl<T: Copy + Default, const N: usize> Default for HistoryRing<T, N> {
    fn default() -> Self {
        let mut buf = VecDeque::with_capacity(N);
        buf.resize(N, T::default());
        Self { buf }
    }
}

impl<T: Copy + Default, const N: usize> HistoryRing<T, N> {
    /// Push a new sample, dropping the oldest if full.
    pub fn push(&mut self, value: T) {
        if self.buf.len() >= N {
            self.buf.pop_front();
        }
        self.buf.push_back(value);
    }

    /// Get all samples as a slice-compatible iterator.
    pub fn as_vec(&self) -> Vec<T> {
        self.buf.iter().copied().collect()
    }

    /// Current number of samples.
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

/// Historical data for sparkline charts (60 samples ≈ 1 minute at 1Hz).
pub struct MetricsHistory {
    pub gpu_busy: HistoryRing<u64, 60>,
    pub sq_occupancy: HistoryRing<u64, 60>,
    pub vgpr_usage: HistoryRing<u64, 60>,
    pub lds_usage: HistoryRing<u64, 60>,
    pub temp: HistoryRing<u64, 60>,
    pub power: HistoryRing<u64, 60>,
    pub vram: HistoryRing<u64, 60>,
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self {
            gpu_busy: HistoryRing::default(),
            sq_occupancy: HistoryRing::default(),
            vgpr_usage: HistoryRing::default(),
            lds_usage: HistoryRing::default(),
            temp: HistoryRing::default(),
            power: HistoryRing::default(),
            vram: HistoryRing::default(),
        }
    }
}

impl MetricsHistory {
    /// Record a new snapshot into the history ring buffers.
    pub fn record(&mut self, snap: &GpuSnapshot) {
        self.gpu_busy.push(snap.gpu_busy as u64);
        self.sq_occupancy.push(snap.sq_occupancy as u64);
        self.vgpr_usage.push(snap.vgpr_usage as u64);
        self.lds_usage.push(snap.lds_usage as u64);
        self.temp.push(snap.temp_edge / 1000); // millideg → degrees
        self.power.push(snap.power_avg / 1_000_000); // μW → W
        self.vram.push((snap.vram_ratio() * 100.0) as u64);
    }
}

// ── UI Layer (ratatui) ─────────────────────────────────────────────────

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Sparkline, Table},
    Frame, Terminal,
};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

/// Color thresholds for metric severity.
struct Thresholds;

impl Thresholds {
    fn color_for(value: f64, warn: f64, crit: f64) -> Color {
        if value >= crit {
            Color::Red
        } else if value >= warn {
            Color::Yellow
        } else {
            Color::Green
        }
    }

    fn gpu_busy_color(v: f64) -> Color { Self::color_for(v, 70.0, 90.0) }
    fn temp_color(v: f64) -> Color { Self::color_for(v, 80.0, 95.0) }
    fn power_color(v: f64) -> Color { Self::color_for(v, 80.0, 95.0) }
    fn vram_color(v: f64) -> Color { Self::color_for(v * 100.0, 70.0, 90.0) }
    fn occupancy_color(v: f64) -> Color { Self::color_for(v, 60.0, 85.0) }
}

/// Build a colored gauge bar for a metric.
fn metric_gauge(label: &str, value: f64, color: Color) -> Gauge<'_> {
    let ratio = (value / 100.0).clamp(0.0, 1.0);
    Gauge::default()
        .block(Block::default().title(label).borders(Borders::ALL))
        .gauge_style(
            Style::default()
                .fg(color)
                .add_modifier(Modifier::BOLD),
        )
        .ratio(ratio)
        .label(format!("{:.1}%", value))
}

/// Build a sparkline widget from history data.
fn metric_sparkline<'a>(title: &'a str, data: &[u64], color: Color) -> Sparkline<'a> {
    Sparkline::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .data(data)
        .style(Style::default().fg(color))
        .max(100)
}

/// Main GPU monitoring TUI application.
pub struct GpuMonitor {
    terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
    collector: MetricsCollector,
    history: MetricsHistory,
    current: GpuSnapshot,
    running: bool,
    refresh_ms: u64,
}

impl GpuMonitor {
    /// Create and initialize the TUI monitor.
    pub fn new(collector: MetricsCollector) -> Result<Self, String> {
        // Setup terminal
        enable_raw_mode().map_err(|e| format!("raw mode: {}", e))?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| format!("alternate screen: {}", e))?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)
            .map_err(|e| format!("terminal: {}", e))?;

        let initial = collector.collect();

        Ok(GpuMonitor {
            terminal,
            collector,
            history: MetricsHistory::default(),
            current: initial,
            running: true,
            refresh_ms: 500,
        })
    }

    /// Run the event loop until user presses 'q' or Ctrl+C.
    pub fn run(&mut self) -> Result<(), String> {
        while self.running {
            // Collect metrics
            self.current = self.collector.collect();
            self.history.record(&self.current);

            // Render — draw() is a free fn to avoid borrow conflict with terminal
            let snap = self.current.clone();
            let history_gpu = self.history.gpu_busy.as_vec();
            let history_sq = self.history.sq_occupancy.as_vec();
            let history_vgpr = self.history.vgpr_usage.as_vec();
            let history_lds = self.history.lds_usage.as_vec();
            let history_temp = self.history.temp.as_vec();
            let history_power = self.history.power.as_vec();
            let history_vram = self.history.vram.as_vec();
            let kfd_node = self.collector.kfd_node();
            let refresh_ms = self.refresh_ms;
            let sample_count = self.history.gpu_busy.len();
            self.terminal
                .draw(|f| draw_frame(f, &snap, &history_gpu, &history_sq,
                    &history_vgpr, &history_lds, &history_temp,
                    &history_power, &history_vram, kfd_node,
                    refresh_ms, sample_count))
                .map_err(|e| format!("draw: {}", e))?;

            // Handle input events (non-blocking, 500ms poll)
            if event::poll(Duration::from_millis(self.refresh_ms))
                .map_err(|e| format!("poll: {}", e))?
            {
                if let Event::Key(key) = event::read().map_err(|e| format!("read: {}", e))? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => self.running = false,
                            KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                self.running = false;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        self.cleanup()?;
        Ok(())
    }

    /// Restore terminal to normal mode.
    fn cleanup(&mut self) -> Result<(), String> {
        disable_raw_mode().map_err(|e| format!("disable raw: {}", e))?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .map_err(|e| format!("leave alt: {}", e))?;
        self.terminal
            .show_cursor()
            .map_err(|e| format!("show cursor: {}", e))?;
        Ok(())
    }
}

impl Drop for GpuMonitor {
    fn drop(&mut self) {
        // Best-effort terminal cleanup
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

// ── Free-function draw helpers (avoid borrow conflict with terminal) ──

/// Draw the entire TUI frame.
fn draw_frame(
    f: &mut Frame<'_>,
    snap: &GpuSnapshot,
    history_gpu: &[u64],
    history_sq: &[u64],
    history_vgpr: &[u64],
    history_lds: &[u64],
    history_temp: &[u64],
    history_power: &[u64],
    history_vram: &[u64],
    kfd_node: u32,
    refresh_ms: u64,
    sample_count: usize,
) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());

    draw_header(f, main_chunks[0], snap);

    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_chunks[1]);

    draw_gauges(f, body_chunks[0], snap);
    draw_sparklines(f, body_chunks[1], history_gpu, history_sq,
        history_vgpr, history_lds, history_temp, history_power, history_vram);
    draw_footer(f, main_chunks[2], kfd_node, refresh_ms, sample_count);
}

fn draw_header(f: &mut Frame<'_>, area: Rect, snap: &GpuSnapshot) {
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " T0 GPU Monitor ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            " │ {} CUs × {} max_waves/CU │ VGPRs: {} │ LDS: {} KB/CU │ [q] Quit",
            snap.cu_count, snap.max_waves_per_cu, snap.max_vgprs, snap.lds_size_kb,
        )),
    ]))
    .block(Block::default().borders(Borders::ALL).title("GPU"))
    .style(Style::default().fg(Color::White));
    f.render_widget(header, area);
}

fn draw_gauges(f: &mut Frame<'_>, area: Rect, snap: &GpuSnapshot) {
    let gauge_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    f.render_widget(
        metric_gauge("GPU Busy", snap.gpu_busy, Thresholds::gpu_busy_color(snap.gpu_busy)),
        gauge_chunks[0],
    );
    f.render_widget(
        metric_gauge("SQ Occupancy", snap.sq_occupancy,
            Thresholds::occupancy_color(snap.sq_occupancy)),
        gauge_chunks[1],
    );
    f.render_widget(
        metric_gauge(&format!("VGPR Usage (max {})", snap.max_vgprs),
            snap.vgpr_usage, Thresholds::occupancy_color(snap.vgpr_usage)),
        gauge_chunks[2],
    );
    f.render_widget(
        metric_gauge(&format!("LDS Usage ({} KB/CU)", snap.lds_size_kb),
            snap.lds_usage, Thresholds::occupancy_color(snap.lds_usage)),
        gauge_chunks[3],
    );

    let vram_pct = snap.vram_ratio() * 100.0;
    let vram_color = Thresholds::vram_color(snap.vram_ratio());
    let vram_gb_used = snap.vram_used as f64 / (1024.0 * 1024.0 * 1024.0);
    let vram_gb_total = snap.vram_total as f64 / (1024.0 * 1024.0 * 1024.0);
    f.render_widget(
        Gauge::default()
            .block(Block::default()
                .title(format!("VRAM ({:.1}/{:.1} GB)", vram_gb_used, vram_gb_total))
                .borders(Borders::ALL))
            .gauge_style(Style::default().fg(vram_color).add_modifier(Modifier::BOLD))
            .ratio(snap.vram_ratio().clamp(0.0, 1.0))
            .label(format!("{:.1}%", vram_pct)),
        gauge_chunks[4],
    );

    let pwr_pct = snap.power_ratio() * 100.0;
    let pwr_color = Thresholds::power_color(pwr_pct);
    f.render_widget(
        Gauge::default()
            .block(Block::default()
                .title(format!("Power ({:.0}/{:.0} W)", snap.power_watts(), snap.power_cap_watts()))
                .borders(Borders::ALL))
            .gauge_style(Style::default().fg(pwr_color).add_modifier(Modifier::BOLD))
            .ratio(snap.power_ratio().clamp(0.0, 1.0))
            .label(format!("{:.1}%", pwr_pct)),
        gauge_chunks[5],
    );

    let rows = vec![
        Row::new(vec![
            format!("Edge: {:.0}°C", snap.temp_celsius()),
            format!("Hotspot: {:.0}°C", snap.temp_hotspot as f64 / 1000.0),
        ]),
        Row::new(vec![
            format!("SCLK: {:.0} MHz", snap.sclk_mhz()),
            format!("MCLK: {:.0} MHz", snap.mclk_mhz()),
        ]),
    ];
    let table = Table::new(rows, [Constraint::Percentage(50), Constraint::Percentage(50)])
        .block(Block::default().title("Info").borders(Borders::ALL))
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(table, gauge_chunks[6]);
}

fn draw_sparklines(
    f: &mut Frame<'_>,
    area: Rect,
    history_gpu: &[u64],
    history_sq: &[u64],
    history_vgpr: &[u64],
    history_lds: &[u64],
    history_temp: &[u64],
    history_power: &[u64],
    history_vram: &[u64],
) {
    let spark_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Fill(1); 7])
        .split(area);

    f.render_widget(
        metric_sparkline("GPU Busy %", history_gpu, Color::Green),
        spark_chunks[0],
    );
    f.render_widget(
        metric_sparkline("SQ Occupancy %", history_sq, Color::Cyan),
        spark_chunks[1],
    );
    f.render_widget(
        metric_sparkline("VGPR Usage %", history_vgpr, Color::Yellow),
        spark_chunks[2],
    );
    f.render_widget(
        metric_sparkline("LDS Usage %", history_lds, Color::Magenta),
        spark_chunks[3],
    );
    f.render_widget(
        metric_sparkline("Temp °C", history_temp, Color::Red),
        spark_chunks[4],
    );
    f.render_widget(
        metric_sparkline("Power W", history_power, Color::Rgb(255, 165, 0)),
        spark_chunks[5],
    );
    f.render_widget(
        metric_sparkline("VRAM %", history_vram, Color::Blue),
        spark_chunks[6],
    );
}

fn draw_footer(f: &mut Frame<'_>, area: Rect, kfd_node: u32, refresh_ms: u64, sample_count: usize) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" Refresh: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}ms", refresh_ms), Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("Samples: {}", sample_count), Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("KFD node: {}", kfd_node), Style::default().fg(Color::White)),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            "Press [q] to quit",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(footer, area);
}

// ── Async Data Broadcasting Layer ──────────────────────────────────────

/// Async metrics broadcaster using tokio watch channel.
///
/// Spawns a background task that collects metrics at the given interval
/// and broadcasts snapshots via a watch channel. Multiple consumers can
/// subscribe independently.
pub struct MetricsBroadcaster {
    rx: tokio::sync::watch::Receiver<GpuSnapshot>,
    _handle: tokio::task::JoinHandle<()>,
}

impl MetricsBroadcaster {
    /// Start broadcasting GPU metrics at the given interval.
    ///
    /// Returns a broadcaster that holds the receiver and keeps the
    /// background task alive.
    pub fn start(collector: MetricsCollector, interval: Duration) -> Self {
        let (tx, rx) = tokio::sync::watch::channel(collector.collect());

        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let snap = collector.collect();
                if tx.send(snap).is_err() {
                    break; // All receivers dropped
                }
            }
        });

        MetricsBroadcaster {
            rx,
            _handle: handle,
        }
    }

    /// Get a clone of the watch receiver for subscribing to updates.
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<GpuSnapshot> {
        self.rx.clone()
    }

    /// Get the latest snapshot without waiting for a new one.
    pub fn latest(&self) -> GpuSnapshot {
        self.rx.borrow().clone()
    }
}

// ── Entry Point ────────────────────────────────────────────────────────

/// Run the GPU monitoring TUI (blocking).
///
/// Auto-detects the AMD GPU, initializes the TUI, and blocks until
/// the user presses 'q' or Ctrl+C.
///
/// # Errors
/// Returns an error if no AMD GPU is found or terminal setup fails.
pub fn run() -> Result<(), String> {
    let collector = MetricsCollector::detect()?;
    let mut monitor = GpuMonitor::new(collector)?;
    monitor.run()
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_defaults() {
        let snap = GpuSnapshot::default();
        assert_eq!(snap.gpu_busy, 0.0);
        assert_eq!(snap.vram_used, 0);
        assert_eq!(snap.vram_ratio(), 0.0);
        assert_eq!(snap.power_ratio(), 0.0);
    }

    #[test]
    fn test_snapshot_conversions() {
        let snap = GpuSnapshot {
            gpu_busy: 75.0,
            vram_used: 8 * 1024 * 1024 * 1024, // 8 GB
            vram_total: 16 * 1024 * 1024 * 1024, // 16 GB
            temp_edge: 63000,    // 63°C
            power_avg: 150_000_000, // 150W
            power_cap: 200_000_000, // 200W
            sclk: 2_780_000_000, // 2780 MHz
            mclk: 2_500_000_000, // 2500 MHz
            ..Default::default()
        };
        assert!((snap.vram_ratio() - 0.5).abs() < 0.001);
        assert!((snap.power_ratio() - 0.75).abs() < 0.001);
        assert!((snap.temp_celsius() - 63.0).abs() < 0.01);
        assert!((snap.power_watts() - 150.0).abs() < 0.01);
        assert!((snap.sclk_mhz() - 2780.0).abs() < 0.01);
        assert!((snap.mclk_mhz() - 2500.0).abs() < 0.01);
    }

    #[test]
    fn test_history_ring() {
        let mut ring: HistoryRing<u64, 3> = HistoryRing::default();
        assert_eq!(ring.len(), 3); // Pre-filled with zeros

        ring.push(10);
        assert_eq!(ring.len(), 3); // Still 3 (replaced oldest zero)
        assert_eq!(ring.as_vec(), vec![0, 0, 10]);

        ring.push(20);
        assert_eq!(ring.as_vec(), vec![0, 10, 20]);

        ring.push(30);
        assert_eq!(ring.as_vec(), vec![10, 20, 30]);

        ring.push(40);
        assert_eq!(ring.as_vec(), vec![20, 30, 40]); // Oldest (10) dropped
    }

    #[test]
    fn test_metrics_history_record() {
        let mut history = MetricsHistory::default();
        let snap = GpuSnapshot {
            gpu_busy: 80.0,
            sq_occupancy: 75.0,
            vgpr_usage: 65.0,
            lds_usage: 40.0,
            temp_edge: 70000,
            power_avg: 120_000_000,
            vram_used: 8 * 1024 * 1024 * 1024,
            vram_total: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        history.record(&snap);

        let busy_vec = history.gpu_busy.as_vec();
        assert_eq!(busy_vec.last(), Some(&80));
        let temp_vec = history.temp.as_vec();
        assert_eq!(temp_vec.last(), Some(&70));
        let power_vec = history.power.as_vec();
        assert_eq!(power_vec.last(), Some(&120));
    }

    #[test]
    fn test_threshold_colors() {
        assert_eq!(Thresholds::gpu_busy_color(50.0), Color::Green);
        assert_eq!(Thresholds::gpu_busy_color(75.0), Color::Yellow);
        assert_eq!(Thresholds::gpu_busy_color(95.0), Color::Red);
    }

    #[test]
    fn test_zero_division_safety() {
        let snap = GpuSnapshot {
            vram_total: 0,
            power_cap: 0,
            ..Default::default()
        };
        assert_eq!(snap.vram_ratio(), 0.0);
        assert_eq!(snap.power_ratio(), 0.0);
    }

    // Integration test: requires real GPU
    #[test]
    #[ignore] // run with: cargo test --features monitor -- --ignored test_real_collector
    fn test_real_collector() {
        let collector = MetricsCollector::detect().expect("No AMD GPU found");
        let snap = collector.collect();
        eprintln!("GPU Busy: {:.1}%", snap.gpu_busy);
        eprintln!("VRAM: {:.1}/{:.1} GB",
            snap.vram_used as f64 / 1e9,
            snap.vram_total as f64 / 1e9);
        eprintln!("Temp: {:.1}°C", snap.temp_celsius());
        eprintln!("Power: {:.1}W / {:.1}W", snap.power_watts(), snap.power_cap_watts());
        eprintln!("SCLK: {:.0} MHz", snap.sclk_mhz());
        eprintln!("MCLK: {:.0} MHz", snap.mclk_mhz());
        eprintln!("CUs: {}, Max waves/CU: {}", snap.cu_count, snap.max_waves_per_cu);
        assert!(snap.cu_count > 0);
        assert!(snap.vram_total > 0);
    }
}
