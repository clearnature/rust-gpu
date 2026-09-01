//! KfdDevice — bare-metal GPU device handle.

use std::sync::Arc;
use std::os::unix::io::RawFd;
use super::buffer::GpuBuffer;
use super::ioctl::*;
use super::aql::AqlQueue;
use super::pm4::Pm4Queue;
use super::ignore_sigpipe;
pub struct KfdDevice {
    pub kfd_fd: RawFd,
    drm_fd: RawFd,
    pub gpu_id: u32,
    /// GFX target version from sysfs decimal format (e.g. 110000 for gfx1100, 120000 for gfx1200)
    pub gfx_target_version: u32,
    /// Base VA for user allocations (auto-incremented, 2MB aligned)
    next_va: std::sync::atomic::AtomicU64,
    /// Event page handle for cleanup (allocated during open)
    event_page_handle: u64,
    /// Event page VA for munmap during cleanup
    event_page_va: u64,
}

/// Global singleton: KFD driver only allows one ACQUIRE_VM per process.
/// All callers of KfdDevice::open() get the same Arc<KfdDevice>.
static GLOBAL_KFD_DEVICE: std::sync::OnceLock<Arc<KfdDevice>> = std::sync::OnceLock::new();

/// Global mutex serializing all CREATE_QUEUE ioctls.
///
/// On RDNA4 (gfx1200), the MES firmware has a race condition when handling
/// concurrent CREATE_QUEUE calls — doorbell offset assignment can collide,
/// causing queues to share a doorbell and produce undefined behavior.
/// This mutex ensures only one CREATE_QUEUE ioctl is in-flight at a time.
static QUEUE_CREATE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl KfdDevice {
    /// Open the GPU device and acquire VM.
    /// Returns a cached global singleton — KFD only allows one ACQUIRE_VM per process.
    pub fn open() -> Result<Arc<Self>, String> {
        Self::open_with_gpu_id(0) // auto-detect
    }

    /// Force a fresh device open (bypasses singleton cache).
    /// Use only when you know the previous device has been fully dropped.
    pub fn open_fresh() -> Result<Arc<Self>, String> {
        Self::open_fresh_with_gpu_id(0)
    }

    fn open_fresh_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        Self::open_device_impl(gpu_id_override)
    }

    pub fn open_with_gpu_id(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        // Return cached singleton if it exists (KFD only allows one ACQUIRE_VM per process)
        if let Some(dev) = GLOBAL_KFD_DEVICE.get() {
            return Ok(Arc::clone(dev));
        }
        let dev = Self::open_device_impl(gpu_id_override)?;
        // Try to store into OnceLock; if another thread raced us, use theirs
        match GLOBAL_KFD_DEVICE.set(Arc::clone(&dev)) {
            Ok(()) => Ok(dev),
            Err(_) => Ok(Arc::clone(GLOBAL_KFD_DEVICE.get().unwrap())),
        }
    }

    /// Internal: actual device open + VM acquire (called once per process)
    fn open_device_impl(gpu_id_override: u32) -> Result<Arc<Self>, String> {
        // CRITICAL: ignore SIGPIPE before any GPU work.
        // When running under `cargo test ... | grep ... | head -N`, the `head`
        // command closes its stdin after reading enough lines, sending SIGPIPE
        // up the pipe chain. Without this, our process dies instantly and
        // Drop handlers (DESTROY_QUEUE, close fd) never run, leaving the GPU
        // executing buggy kernels → KFD MODE1 reset → next run hard-hangs.
        ignore_sigpipe();

        // Open /dev/kfd (with retry for GPU recovery after MODE1 reset)
        let kfd_fd = Self::open_kfd_with_retry()?;

        // Get KFD version
        let mut ver = KfdGetVersionArgs::default();
        ioctl_safe(kfd_fd, AMDKFD_IOC_GET_VERSION, &mut ver as *mut _ as *mut u8)?;
        eprintln!("[KFD] Version {}.{}", ver.major_version, ver.minor_version);

        // Determine gpu_id from topology
        let gpu_id = if gpu_id_override != 0 {
            gpu_id_override
        } else {
            Self::detect_gpu_id()?
        };
        eprintln!("[KFD] GPU ID: {}", gpu_id);

        // Open /dev/dri/renderDXXX — detect correct minor from KFD topology
        let render_minor = Self::detect_drm_render_minor(gpu_id).unwrap_or(128);
        let drm_path = format!("/dev/dri/renderD{}\0", render_minor);
        let drm_fd = unsafe { open(drm_path.as_ptr(), 2) };
        if drm_fd < 0 {
            unsafe { close(kfd_fd); }
            return Err(format!("Failed to open /dev/dri/renderD{}: {}", render_minor, std::io::Error::last_os_error()));
        }
        eprintln!("[KFD] Using /dev/dri/renderD{}", render_minor);

        // Acquire VM (bind DRM fd to KFD for this gpu)
        let mut acq = KfdAcquireVmArgs { drm_fd: drm_fd as u32, gpu_id };
        ioctl_safe(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, &mut acq as *mut _ as *mut u8)?;
        eprintln!("[KFD] VM acquired");

        // RUNTIME_ENABLE - required on KFD >= 1.14 to activate AQL dispatch
        // Without this, doorbell writes for AQL queues are not processed by CP/MEC
        if ver.minor_version >= 14 {
            let mut rt = KfdRuntimeEnableArgs::default();
            ioctl_safe(kfd_fd, AMDKFD_IOC_RUNTIME_ENABLE, &mut rt as *mut _ as *mut u8)?;
            eprintln!("[KFD] Runtime enabled (caps=0x{:X})", rt.capabilities_mask);
        }

        // Event page + CREATE_EVENT — tinygrad does this before creating queues
        // This sets up the KFD event infrastructure required for MEC processing
        let event_page = {
            // Allocate a small uncached buffer for event page
            let va = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    0x8000, // 32KB event page
                    0, // PROT_NONE (reserve only)
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if va == MAP_FAILED || va.is_null() {
                return Err("Failed to reserve VA for event page".to_string());
            }
            let va_addr = va as u64;

            let mut alloc_args = KfdAllocMemoryArgs {
                va_addr,
                size: 0x8000,
                handle: 0,
                mmap_offset: 0,
                gpu_id,
                flags: KFD_IOC_ALLOC_MEM_FLAGS_GTT
                    | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                    | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                    | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
                    | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
            };
            ioctl_safe(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY, &mut alloc_args as *mut _ as *mut u8)
                .map_err(|e| format!("Event page ALLOC_MEMORY failed: {}", e))?;

            // Map to GPU
            let mut gpu_ids = [gpu_id];
            let mut map_args = KfdMapMemoryArgs {
                handle: alloc_args.handle,
                device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
                n_devices: 1,
                n_success: 0,
            };
            ioctl_safe(kfd_fd, AMDKFD_IOC_MAP_MEMORY, &mut map_args as *mut _ as *mut u8)
                .map_err(|e| format!("Event page MAP_MEMORY failed: {}", e))?;

            // mmap to CPU
            let host_ptr = unsafe {
                mmap(
                    va,
                    0x8000,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    drm_fd,
                    alloc_args.mmap_offset as i64,
                )
            };
            if host_ptr == MAP_FAILED {
                return Err("Event page CPU mmap failed".to_string());
            }
            // Zero it
            unsafe { std::ptr::write_bytes(host_ptr, 0, 0x8000); }

            alloc_args.handle
        };

        // Read gfx_target_version from sysfs topology (decimal format: 110000=gfx1100, 120000=gfx1200)
        let gfx_target_version = {
            let mut ver = 110000u32; // default: gfx1100
            for node in 1..=8 {
                // Check gpu_id from separate sysfs file
                let gpu_id_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
                let node_gpu_id: u32 = std::fs::read_to_string(&gpu_id_path)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                if node_gpu_id != gpu_id { continue; }
                // Found matching node — read gfx_target_version from properties
                let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
                if let Ok(props) = std::fs::read_to_string(&prop_path) {
                    for line in props.lines() {
                        let mut it = line.split_whitespace();
                        if let (Some("gfx_target_version"), Some(v)) = (it.next(), it.next()) {
                            ver = v.parse().unwrap_or(110000);
                            break;
                        }
                    }
                }
                break;
            }
            ver
        };
        eprintln!("[KFD] GFX target version: {} (gfx{})", gfx_target_version, gfx_target_version / 10000);

        // CREATE_EVENT with event_page_offset = handle
        // This initializes the KFD event page in the kernel
        #[repr(C)]
        #[derive(Default)]
        struct KfdCreateEventArgs {
            event_page_offset: u64,
            event_trigger_data: u32,
            event_type: u32,
            auto_reset: u32,
            node_id: u32,
            event_id: u32,
            event_slot_index: u32,
        }
        let mut ev = KfdCreateEventArgs {
            event_page_offset: event_page,
            ..Default::default()
        };
        ioctl_safe(kfd_fd, AMDKFD_IOC_CREATE_EVENT, &mut ev as *mut _ as *mut u8)
            .map_err(|e| format!("CREATE_EVENT failed: {}", e))?;
        eprintln!("[KFD] Event page created (event_id={}, slot={})", ev.event_id, ev.event_slot_index);

        let device = Arc::new(Self {
            kfd_fd,
            drm_fd,
            gpu_id,
            gfx_target_version,
            next_va: std::sync::atomic::AtomicU64::new(0x1_0000_0000),
            event_page_handle: event_page,
            event_page_va: 0, // VA is managed by kernel, not tracked for munmap
        });

        // GPU health probe: allocate a small buffer, write, read-back, verify.
        // If the GPU just recovered from MODE1 reset, VRAM may be unstable.
        // This catches it early instead of hanging on the first kernel dispatch.
        Self::gpu_health_probe(&device)?;

        Ok(device)
    }

    /// Open /dev/kfd with retry logic for GPU recovery after MODE1 reset.
    /// KFD driver may temporarily refuse open() while GPU is resetting.
    fn open_kfd_with_retry() -> Result<RawFd, String> {
        for attempt in 0..5 {
            let fd = unsafe { open(b"/dev/kfd\0".as_ptr(), 2 /* O_RDWR */) };
            if fd >= 0 {
                return Ok(fd);
            }
            let err = std::io::Error::last_os_error();
            if attempt < 4 {
                eprintln!("[KFD] /dev/kfd open failed (attempt {}): {} — GPU may be recovering from reset, retrying in 1s...",
                    attempt + 1, err);
                std::thread::sleep(std::time::Duration::from_secs(1));
            } else {
                return Err(format!("Failed to open /dev/kfd after 5 attempts: {}", err));
            }
        }
        unreachable!()
    }

    /// GPU health probe: allocate tiny GTT buffer, write pattern, read back.
    /// Catches post-MODE1-reset VRAM instability before any real work.
    fn gpu_health_probe(device: &Arc<Self>) -> Result<(), String> {
        for attempt in 0..3 {
            match Self::health_probe_once(device) {
                Ok(()) => {
                    eprintln!("[KFD] GPU health check PASSED");
                    return Ok(());
                }
                Err(e) => {
                    if attempt < 2 {
                        eprintln!("[KFD] GPU health check FAILED (attempt {}): {} — retrying in 2s...",
                            attempt + 1, e);
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    } else {
                        return Err(format!("GPU health check failed after 3 attempts: {}", e));
                    }
                }
            }
        }
        unreachable!()
    }

    fn health_probe_once(device: &Arc<Self>) -> Result<(), String> {
        // Allocate a small GTT buffer (coherent, visible to both CPU and GPU)
        let probe_buf = device.alloc_gtt(4096)?;
        // Write a known pattern
        let pattern: u64 = 0xDEAD_BEEF_CAFE_F00D;
        probe_buf.write_val::<u64>(0, pattern);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        // Read it back
        let readback: u64 = probe_buf.read_val(0);
        if readback != pattern {
            return Err(format!("GPU memory readback mismatch: wrote 0x{:X}, read 0x{:X}", pattern, readback));
        }
        // Buffer dropped here → RAII cleanup
        Ok(())
    }

    fn detect_gpu_id() -> Result<u32, String> {
        // Read gpu_id from sysfs topology
        // Try node 1 first (node 0 is usually CPU)
        for node in 1..=8 {
            let path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let id: u32 = content.trim().parse().unwrap_or(0);
                if id > 0 {
                    return Ok(id);
                }
            }
        }
        Err("No GPU found in KFD topology".to_string())
    }

    /// Detect DRM render minor for a given GPU ID from KFD topology
    fn detect_drm_render_minor(gpu_id: u32) -> Result<u32, String> {
        for node in 0..=8 {
            let gpu_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            if let Ok(content) = std::fs::read_to_string(&gpu_path) {
                let id: u32 = content.trim().parse().unwrap_or(0);
                if id == gpu_id {
                    let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
                    if let Ok(props) = std::fs::read_to_string(&prop_path) {
                        for line in props.lines() {
                            if line.starts_with("drm_render_minor") {
                                if let Some(val) = line.split_whitespace().nth(1) {
                                    return val.parse::<u32>().map_err(|e| e.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        Err("drm_render_minor not found".to_string())
    }

    /// Pre-reserve a VA range via mmap(PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS)
    /// KFD requires the VA to be pre-reserved before ALLOC_MEMORY can use it.
    fn alloc_va(&self, size: usize) -> u64 {
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                0, // PROT_NONE
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1, // no fd
                0,
            )
        };
        if ptr == MAP_FAILED || ptr.is_null() {
            panic!("Failed to reserve VA space: {}", std::io::Error::last_os_error());
        }
        ptr as u64
    }

    /// Allocate VRAM buffer (writable, public, CPU-visible via mmap)
    pub fn alloc_vram(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
    }

    /// Allocate executable VRAM (for kernel machine code)
    /// After writing code, call `hdp_flush()` or read back one byte to flush HDP.
    pub fn alloc_code(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)
    }

    /// Allocate GTT memory (host-visible, for kernargs, signals, etc.)
    pub fn alloc_gtt(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_COHERENT)
    }

    /// Allocate uncached GTT memory (for ring buffer, wr/rd ptrs, signals, kernargs).
    /// Keep this non-executable: EXECUTABLE GTT mappings can trigger CPF permission faults.
    pub fn alloc_uncached(self: &Arc<Self>, size: usize) -> Result<GpuBuffer, String> {
        self.alloc_memory(size,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
            KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
            KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | // Required: CP fetches ring buffer as instructions
            KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
            KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
            KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
            KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    }

    /// Internal: allocate GPU memory via KFD ioctl
    fn alloc_memory(self: &Arc<Self>, size: usize, flags: u32) -> Result<GpuBuffer, String> {
        let page_size = 4096usize;
        let aligned_size = ((size + page_size - 1) / page_size) * page_size;
        let va_addr = self.alloc_va(aligned_size);

        let mut args = KfdAllocMemoryArgs {
            va_addr,
            size: aligned_size as u64,
            handle: 0,
            mmap_offset: 0,
            gpu_id: self.gpu_id,
            flags,
        };

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_ALLOC_MEMORY, &mut args as *mut _ as *mut u8)
            .map_err(|e| format!("ALLOC_MEMORY failed (size={}, flags=0x{:X}): {}", aligned_size, flags, e))?;

        // Map memory to GPU
        let mut gpu_ids = [self.gpu_id];
        let mut map_args = KfdMapMemoryArgs {
            handle: args.handle,
            device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        ioctl_safe(self.kfd_fd, AMDKFD_IOC_MAP_MEMORY, &mut map_args as *mut _ as *mut u8)
            .map_err(|e| format!("MAP_MEMORY failed: {}", e))?;
        if map_args.n_success != 1 {
            return Err(format!("MAP_MEMORY incomplete: n_success={}", map_args.n_success));
        }

        // mmap to CPU address space using MAP_FIXED on the pre-reserved VA
        // CRITICAL: VRAM mmap must use drm_fd (/dev/dri/renderD128), NOT kfd_fd!
        let host_ptr = unsafe {
            mmap(
                args.va_addr as *mut u8, // MAP_FIXED on pre-reserved address
                aligned_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED,
                self.drm_fd,
                args.mmap_offset as i64,
            )
        };
        if host_ptr == MAP_FAILED || host_ptr.is_null() {
            return Err(format!("mmap failed for KFD buffer: {}", std::io::Error::last_os_error()));
        }
        assert_eq!(host_ptr as u64, args.va_addr, "MAP_FIXED returned wrong address");

        Ok(GpuBuffer {
            handle: args.handle,
            va_addr: args.va_addr,
            host_ptr,
            size: aligned_size,
            device: Arc::clone(self),
        })
    }

    /// Determine CWSR sizes for this GPU. Reads the KFD topology properties
    /// once, then either trusts the kernel-reported values (cwsr_size /
    /// ctl_stack_size, exported by modern kernels) or falls back to the
    /// official libhsakmt formula from update_ctx_save_restore_size()
    /// (rocr-runtime/libhsakmt/src/queues.c):
    ///   cu_num  = NumFComputeCores / NumSIMDPerCU / NumXcc
    ///   wave_num = cu_num * 32          (gfxv >= NAVI10)
    ///   ctl_stack = PAGE_ALIGN(sizeof(HsaUserContextSaveAreaHeader=40) + wave_num*12 + 8)
    ///   wg_data   = cu_num * (vgpr + sgpr + lds + hwreg)
    ///     where vgpr=0x60000 for gfx1151/gfx1200/gfx1201 (hsakmt_get_vgpr_size_per_cu),
    ///           sgpr=0x4000 (SGPR_SIZE_PER_CU), hwreg=0x1000 (HWREG_SIZE_PER_CU),
    ///           lds from topology lds_size_in_kb
    ///   ctx_save_restore_size = ctl_stack + PAGE_ALIGN(wg_data)
    /// Returns (ctx_save_restore_size, ctl_stack_size, wave_num).
    fn cwsr_sizes(&self) -> Result<(u32, u32, u32), String> {
        let mut simd_count = 0u32;
        let mut simd_per_cu = 2u32;
        let mut num_xcc = 1u32;
        let mut lds_kb = 64u32;
        let mut gfxv: u32 = 120000; // default gfx1200 (sysfs decimal format)
        let mut sysfs_cwsr = 0u32;
        let mut sysfs_ctl = 0u32;
        for node in 0..=8 {
            let gpu_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);
            if let Ok(content) = std::fs::read_to_string(&gpu_path) {
                let id: u32 = content.trim().parse().unwrap_or(0);
                if id == self.gpu_id {
                    let prop_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
                    if let Ok(props) = std::fs::read_to_string(&prop_path) {
                        for line in props.lines() {
                            let mut it = line.split_whitespace();
                            match (it.next(), it.next()) {
                                (Some("simd_count"), Some(v)) => simd_count = v.parse().unwrap_or(0),
                                (Some("simd_per_cu"), Some(v)) => simd_per_cu = v.parse().unwrap_or(2),
                                (Some("num_xcc"), Some(v)) => num_xcc = v.parse().unwrap_or(1),
                                (Some("lds_size_in_kb"), Some(v)) => lds_kb = v.parse().unwrap_or(64),
                                (Some("gfx_target_version"), Some(v)) => gfxv = v.parse().unwrap_or(120000),
                                (Some("cwsr_size"), Some(v)) => sysfs_cwsr = v.parse().unwrap_or(0),
                                (Some("ctl_stack_size"), Some(v)) => sysfs_ctl = v.parse().unwrap_or(0),
                                _ => {}
                            }
                        }
                    }
                    break;
                }
            }
        }
        if simd_count == 0 || simd_per_cu == 0 {
            return Err("cannot read simd_count/simd_per_cu from KFD topology".to_string());
        }
        let cu_num = simd_count / simd_per_cu / num_xcc.max(1);
        let wave_num = cu_num * 32; // gfxv >= NAVI10 (all RDNA+)
        if sysfs_cwsr > 0 && sysfs_ctl > 0 {
            return Ok((sysfs_cwsr, sysfs_ctl, wave_num));
        }
        let ctl_stack = (40u64 + wave_num as u64 * 12 + 8).div_ceil(4096) as u32 * 4096;
        let vgpr: u32 = match gfxv {
            110001 | 110002 | 120000 | 120001 => 0x60000, // gfx1151/gfx1200/gfx1201 (plum_bonito/wheat_nas class)
            _ => 0x40000,
        };
        let wg_data = cu_num as u64 * (vgpr as u64 + 0x4000 + ((lds_kb as u64) << 10) + 0x1000);
        let wg_data_page = wg_data.div_ceil(4096) as u32 * 4096;
        let ctx_save = ctl_stack + wg_data_page;
        Ok((ctx_save, ctl_stack, wave_num))
    }

    /// Create an AQL compute queue with default ring size (4MB = 65536 packets)
    pub fn create_queue(self: &Arc<Self>) -> Result<AqlQueue, String> {
        self.create_queue_sized(4 << 20)  // 4MB default
    }

    /// Create an AQL compute queue with specified ring buffer size in bytes.
    /// Ring size must be power of 2. Each packet is 64 bytes.
    /// Recommended: 1<<20 (1MB, 16K pkts), 4<<20 (4MB, 64K pkts), 16<<20 (16MB, 256K pkts)
    ///
    /// **Thread safety**: Uses a global mutex to serialize CREATE_QUEUE ioctls.
    /// On RDNA4 (gfx1200), the MES firmware has a race condition when handling
    /// concurrent CREATE_QUEUE calls — the doorbell offset assignment can
    /// collide, causing one queue to get a duplicate doorbell. Serializing
    /// all queue creation through this mutex eliminates the race.
    pub fn create_queue_sized(self: &Arc<Self>, ring_size: u32) -> Result<AqlQueue, String> {
        let _guard = QUEUE_CREATE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        assert!(ring_size.is_power_of_two(), "AQL ring_size must be power of 2, got {}", ring_size);

        // Allocate ring buffer (uncached GTT — tinygrad pattern)
        let ring_buffer = self.alloc_uncached(ring_size as usize)?;

        // Zero the ring buffer, then initialize all packet headers to INVALID(1)
        // WARNING: Header=0 means HSA_PACKET_TYPE_VENDOR_SPECIFIC (not empty!)
        //          Only Header=1 (HSA_PACKET_TYPE_INVALID) marks a slot as free.
        //          CP prefetches slots and will choke on VENDOR_SPECIFIC(0) headers.
        unsafe {
            std::ptr::write_bytes(ring_buffer.host_ptr, 0, ring_buffer.size);
            let num_packets = ring_size as usize / 64;
            for i in 0..num_packets {
                let pkt_ptr = ring_buffer.host_ptr.add(i * 64) as *mut u16;
                std::ptr::write_volatile(pkt_ptr, 1u16); // HSA_PACKET_TYPE_INVALID
            }
        }

        // Allocate write/read pointer memory (uncached GTT)
        let wr_ptrs = self.alloc_uncached(4096)?; // page for write_ptr + read_ptr
        unsafe { std::ptr::write_bytes(wr_ptrs.host_ptr, 0, wr_ptrs.size); }
        // Write/read pointer addresses — use GPU VA (same as Tinygrad pattern)
        let write_ptr_va = wr_ptrs.va_addr;
        let read_ptr_va = wr_ptrs.va_addr + 8;

        // Allocate EOP buffer (uncached GTT)
        let eop_buffer = self.alloc_uncached(4096)?;

        // CWSR (Context Wave Save Restore) buffer
        // Sizes are ASIC-specific. Preferred source: the kernel's own topology
        // report (sysfs properties `cwsr_size` / `ctl_stack_size`) — the kernel
        // validates CREATE_QUEUE against these exact values.
        // Fallback: compute via the official libhsakmt formula (see
        // compute_cwsr_sizes), which reproduces the sysfs numbers on gfx1200
        // (RX 9070 XT: 32 WGP-CUs × 0x75000 → cwsr=15351808, ctl=16384).
        // On GPUs without CWSR support, set KFD_NO_CWSR=1 to skip allocation.
        let cwsr_disabled = std::env::var("KFD_NO_CWSR").map(|v| v == "1").unwrap_or(false);
        let (cwsr_buffer, cwsr_size, ctl_stack_size) = if !cwsr_disabled {
            // debug memory follows libhsakmt: wave_num * 32B, 64B-aligned, appended
            // after ctx_save_restore (total page-aligned at alloc time below).
            let (ctx_save, ctl_stack, wave_num) = self.cwsr_sizes()?;
            let debug_mem = (wave_num as u64 * 32).div_ceil(64) as u32 * 64;
            let total_cwsr_alloc: usize = (ctx_save as u64 + debug_mem as u64).div_ceil(4096) as usize * 4096;
            if std::env::var("KFD_DEBUG").is_ok() {
                eprintln!("[KFD] CWSR: cwsr_size={} ctl_stack={} wave_num={} debug_mem={} total_alloc={}",
                    ctx_save, ctl_stack, wave_num, debug_mem, total_cwsr_alloc);
            }
            let buf = self.alloc_uncached(total_cwsr_alloc)?;
            (Some(buf), ctx_save, ctl_stack)
        } else {
            (None, 0u32, 0u32)
        };

        let mut args = KfdCreateQueueArgs {
            ring_base_address: ring_buffer.va_addr,
            write_pointer_address: write_ptr_va,
            read_pointer_address: read_ptr_va,
            doorbell_offset: 0, // returned by kernel
            ring_size,
            gpu_id: self.gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: 100,
            queue_priority: 7, // medium priority
            queue_id: 0,       // returned by kernel
            eop_buffer_address: eop_buffer.va_addr,
            eop_buffer_size: eop_buffer.size as u64,
            ctx_save_restore_address: cwsr_buffer.as_ref().map(|b| b.va_addr).unwrap_or(0),
            ctx_save_restore_size: cwsr_size,
            ctl_stack_size,
            sdma_engine_id: 0,
            pad: 0,
        };

        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] CREATE_QUEUE args:");
            eprintln!("  ring_base=0x{:X} ring_size={}", args.ring_base_address, args.ring_size);
            eprintln!("  write_ptr=0x{:X} read_ptr=0x{:X}", args.write_pointer_address, args.read_pointer_address);
            eprintln!("  gpu_id={} queue_type={} pct={} pri={}", args.gpu_id, args.queue_type, args.queue_percentage, args.queue_priority);
            eprintln!("  eop_addr=0x{:X} eop_size={}", args.eop_buffer_address, args.eop_buffer_size);
            eprintln!("  cwsr_addr=0x{:X} cwsr_size={} ctl_stack={}", args.ctx_save_restore_address, args.ctx_save_restore_size, args.ctl_stack_size);
        }

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &mut args as *mut _ as *mut u8)
            .map_err(|e| format!("CREATE_QUEUE failed: {}", e))?;

        eprintln!("[KFD] Queue {} created (doorbell_offset=0x{:X})", args.queue_id, args.doorbell_offset);

        // mmap doorbell from /dev/kfd using the returned offset
        // doorbell_offset is the mmap offset from KFD — use directly
        // Use &!0x1FFF (8KB/two-page alignment) per tinygrad's proven approach
        let doorbell_base = args.doorbell_offset & !0x1FFF; // two-page aligned
        let doorbell_in_page = (args.doorbell_offset - doorbell_base) as usize;
        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] doorbell raw=0x{:X} base=0x{:X} in_page=0x{:X}",
                args.doorbell_offset, doorbell_base, doorbell_in_page);
        }
        let doorbell_mmap = unsafe {
            mmap(
                std::ptr::null_mut(),
                0x2000, // two pages
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd,
                doorbell_base as i64,
            )
        };
        if doorbell_mmap == MAP_FAILED || doorbell_mmap.is_null() {
            return Err(format!("mmap doorbell failed: {}", std::io::Error::last_os_error()));
        }
        let doorbell_ptr = unsafe { doorbell_mmap.add(doorbell_in_page) };
        if std::env::var("KFD_DEBUG").is_ok() {
            eprintln!("[KFD] doorbell mmap={:?} ptr={:?}", doorbell_mmap, doorbell_ptr);
        }

        // Allocate completion buffer for PM4-in-AQL — GPU writes seqno here after dispatch
        let completion_buf = self.alloc_uncached(64)?;
        // Zero it initially
        unsafe { std::ptr::write_bytes(completion_buf.host_ptr, 0, 64); }

        Ok(AqlQueue {
            queue_id: args.queue_id,
            ring_buffer,
            ring_size,
            write_ptr_host: wr_ptrs.host_ptr as *mut u64,
            read_ptr_host: unsafe { wr_ptrs.host_ptr.add(8) as *mut u64 },
            doorbell_ptr: doorbell_ptr as *mut u64,
            doorbell_mmap_base: doorbell_mmap,
            doorbell_mmap_size: 0x2000,
            pm4_ib: None,
            pm4_ib_offset: 0,
            completion_buf,
            completion_seqno: 0,
            _wr_ptrs: wr_ptrs,
            _eop_buffer: eop_buffer,
            _cwsr_buffer: cwsr_buffer,
            device: Arc::clone(self),
        })
    }

    /// Create a PM4 compute queue (type=0)
    /// PM4 queues use raw PACKET3 commands instead of AQL packets.
    /// Doorbell is u32 byte-offset into ring buffer.
    ///
    /// Serialized via QUEUE_CREATE_MUTEX to avoid MES firmware race on RDNA4.
    pub fn create_pm4_queue(self: &Arc<Self>) -> Result<Pm4Queue, String> {
        let _guard = QUEUE_CREATE_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        let ring_size: u32 = 16 << 20; // 16MB ring (same as tinygrad)
        let ring_buffer = self.alloc_uncached(ring_size as usize)?;
        // Zero ring
        unsafe { std::ptr::write_bytes(ring_buffer.host_ptr, 0, ring_buffer.size); }

        let wr_ptrs = self.alloc_uncached(4096)?;
        unsafe { std::ptr::write_bytes(wr_ptrs.host_ptr, 0, wr_ptrs.size); }
        let write_ptr_va = wr_ptrs.va_addr;
        let read_ptr_va = wr_ptrs.va_addr + 8;

        let eop_buffer = self.alloc_uncached(4096)?;
        unsafe { std::ptr::write_bytes(eop_buffer.host_ptr, 0, eop_buffer.size); }

        // CWSR — same pattern as AQL queue
        let cwsr_disabled = std::env::var("KFD_NO_CWSR").map(|v| v == "1").unwrap_or(false);
        let (cwsr_buffer, cwsr_size, ctl_stack_size) = if !cwsr_disabled {
            let total_cwsr_alloc: usize = 46145536;
            let buf = self.alloc_uncached(total_cwsr_alloc)?;
            unsafe { std::ptr::write_bytes(buf.host_ptr, 0, buf.size); }
            (Some(buf), 46047232u32, 40960u32)
        } else {
            (None, 0u32, 0u32)
        };

        let mut args = KfdCreateQueueArgs {
            ring_base_address: ring_buffer.va_addr,
            write_pointer_address: write_ptr_va,
            read_pointer_address: read_ptr_va,
            doorbell_offset: 0,
            ring_size,
            gpu_id: self.gpu_id,
            queue_type: KFD_IOC_QUEUE_TYPE_COMPUTE, // PM4!
            queue_percentage: 100,
            queue_priority: 7,
            queue_id: 0,
            eop_buffer_address: eop_buffer.va_addr,
            eop_buffer_size: eop_buffer.size as u64,
            ctx_save_restore_address: cwsr_buffer.as_ref().map(|b| b.va_addr).unwrap_or(0),
            ctx_save_restore_size: cwsr_size,
            ctl_stack_size,
            sdma_engine_id: 0,
            pad: 0,
        };

        println!("[KFD] PM4 CREATE_QUEUE args:");
        println!("  ring_base=0x{:X} ring_size={}", ring_buffer.va_addr, ring_size);
        println!("  queue_type=0 (COMPUTE/PM4)");

        ioctl_safe(self.kfd_fd, AMDKFD_IOC_CREATE_QUEUE,
            &mut args as *mut _ as *mut u8)?;

        println!("[KFD] PM4 Queue {} created (doorbell_offset=0x{:X})",
            args.queue_id, args.doorbell_offset);

        // Map doorbell — use offset directly (no >>1 shift!)
        // KFD returns doorbell_offset as a direct mmap offset for /dev/kfd.
        // Use &!0x1fff (8KB/two-page alignment) per tinygrad's proven approach.
        let db_offset_raw = args.doorbell_offset;
        let db_base = db_offset_raw & !0x1FFF; // two-page aligned
        let db_in_page = (db_offset_raw - db_base) as usize;
        eprintln!("[KFD] PM4 doorbell: raw=0x{:X} base=0x{:X} in_page=0x{:X}",
            db_offset_raw, db_base, db_in_page);
        let doorbell_base = unsafe {
            mmap(
                std::ptr::null_mut(),
                8192,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd,
                db_base as i64,
            )
        };
        if doorbell_base == MAP_FAILED {
            return Err("PM4 doorbell mmap failed".to_string());
        }
        let doorbell_ptr = unsafe { doorbell_base.add(db_in_page) as *mut u64 };

        Ok(Pm4Queue {
            queue_id: args.queue_id,
            ring_buffer,
            ring_size,
            write_ptr_host: wr_ptrs.host_ptr as *mut u64,
            read_ptr_host: unsafe { wr_ptrs.host_ptr.add(8) as *mut u64 },
            doorbell_ptr,
            write_offset: 0, // byte offset into ring
            doorbell_mmap_base: doorbell_base,
            _wr_ptrs: wr_ptrs,
            _eop_buffer: eop_buffer,
            _cwsr_buffer: cwsr_buffer,
            device: Arc::clone(self),
        })
    }

    /// Flush HDP (Host Data Path) cache
    /// Forces CPU writes to VRAM to be visible to GPU.
    /// Uses PCIe read-after-write ordering: reading any byte from the
    /// VRAM buffer forces all pending writes to drain.
    pub fn hdp_flush(buf: &GpuBuffer) {
        let _ = unsafe { std::ptr::read_volatile(buf.host_ptr) };
    }
}

impl Drop for KfdDevice {
    fn drop(&mut self) {
        // Release event page memory (allocated during open)
        if self.event_page_handle != 0 {
            // Unmap from GPU
            let mut gpu_ids = [self.gpu_id];
            let mut unmap = KfdUnmapMemoryArgs {
                handle: self.event_page_handle,
                device_ids_array_ptr: gpu_ids.as_mut_ptr() as u64,
                n_devices: 1,
                n_success: 0,
            };
            let _ = ioctl_safe(self.kfd_fd, AMDKFD_IOC_UNMAP_MEMORY,
                &mut unmap as *mut _ as *mut u8);
            // Free GPU memory
            let mut free = KfdFreeMemoryArgs { handle: self.event_page_handle };
            let _ = ioctl_safe(self.kfd_fd, AMDKFD_IOC_FREE_MEMORY,
                &mut free as *mut _ as *mut u8);
        }
        unsafe {
            close(self.drm_fd);
            close(self.kfd_fd);
        }
    }
}

impl KfdDevice {
    /// Prepare kernel arguments in a GPU buffer.
    /// For our kernels: typically 6 args = 4 pointers (8B each) + 2 u32s (4B each) = 40 bytes
    pub fn prepare_kernargs(self: &Arc<Self>, args_data: &[u8]) -> Result<GpuBuffer, String> {
        let buf = self.alloc_uncached(std::cmp::max(args_data.len(), 256))?;
        buf.write(args_data);
        Ok(buf)
    }

    /// Allocate a completion signal buffer (8 bytes, GTT, coherent)
    pub fn alloc_signal(self: &Arc<Self>) -> Result<GpuBuffer, String> {
        self.alloc_uncached(4096) // full page, uncached for GPU-CPU coherence
    }
}

// =============================================================================
// DispatchPool — auto-growing kernargs pool (no fixed slot limit)
// =============================================================================

