use crate::universal::core::{
    Arch, Block, ComputeQueue, CopyQueue, DeviceInfo, DriverFactory, GpuDevice,
    GpuMemory, Grid, Kernel, MemType, QueueConfig, Signal, Vendor,
};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use std::time::Duration;

// ═══════════════════════════════════════════════════════
// NVIDIA 驱动 ioctl 常量
// ═══════════════════════════════════════════════════════

// ioctl magic: 'F' = 0x46
const NV_IOCTL_MAGIC: u8 = 0x46;

// RM ioctls (on /dev/nvidiactl)
const NV_ESC_CARD_INFO: u64 = 200;
const NV_ESC_REGISTER_FD: u64 = 201;
const NV_ESC_RM_ALLOC: u64 = 0x2B;
const NV_ESC_RM_CONTROL: u64 = 0x2A;
const NV_ESC_RM_FREE: u64 = 0x29;
const NV_ESC_RM_MAP_MEMORY: u64 = 0x4E;
const NV_ESC_RM_ALLOC_MEMORY: u64 = 0x27;

// UVM ioctls (on /dev/nvidia-uvm)
const UVM_INITIALIZE: u64 = 39;
const UVM_REGISTER_GPU: u64 = 37;
const UVM_REGISTER_GPU_VASPACE: u64 = 25;
const UVM_REGISTER_CHANNEL: u64 = 27;

// RM object classes
const NV01_ROOT_CLIENT: u32 = 0x41;
const NV01_DEVICE_0: u32 = 0x80;
const NV20_SUBDEVICE_0: u32 = 0x2080;

// ═══════════════════════════════════════════════════════
// NvDriver — NVIDIA GPU 驱动
// ═══════════════════════════════════════════════════════

pub struct NvDriver {
    available: bool,
}

impl NvDriver {
    pub fn new() -> Self {
        Self {
            available: Self::is_available_fn(),
        }
    }

    pub fn is_available_fn() -> bool {
        std::path::Path::new("/dev/nvidiactl").exists()
            && std::path::Path::new("/dev/nvidia-uvm").exists()
    }

    /// 构造 ioctl 命令号
    fn ioctl_nr(nr: u64, size: usize) -> u64 {
        // direction=3 (_IOWR) | size(13 bits) | magic(8 bits) | nr(8 bits)
        let size = size as u64;
        (3u64 << 30) | ((size & 0x1FFF) << 16) | ((NV_IOCTL_MAGIC as u64) << 8) | (nr & 0xFF)
    }
}

impl DriverFactory for NvDriver {
    fn enumerate(&self) -> Vec<DeviceInfo> {
        if !self.available { return Vec::new(); }

        // 打开 /dev/nvidiactl
        let ctl_fd = match OpenOptions::new().read(true).write(true).open("/dev/nvidiactl") {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        // NV_ESC_CARD_INFO: 枚举 GPU
        let mut cards = [NvCardInfo::default(); 64];
        let cmd = Self::ioctl_nr(NV_ESC_CARD_INFO, std::mem::size_of_val(&cards));
        let ret = unsafe {
            libc::ioctl(ctl_fd.as_raw_fd(), cmd, cards.as_mut_ptr())
        };

        if ret < 0 {
            return Vec::new();
        }

        let mut devices = Vec::new();
        for (i, card) in cards.iter().enumerate() {
            if card.valid == 0 { continue; }

            // 检测架构 (根据 device_id)
            let arch = match card.device_id {
                0x2684..=0x26FF => Arch::Sm89,  // Ada Lovelace (RTX 4090)
                0x2200..=0x22FF => Arch::Sm80,  // Ampere (A100)
                0x2400..=0x24FF => Arch::Sm86,  // Ampere (RTX 3090)
                0x2700..=0x27FF => Arch::Sm90,  // Hopper (H100)
                _ => Arch::Sm89, // 默认 Ada
            };

            devices.push(DeviceInfo {
                id: i as u32,
                name: format!("NVIDIA {:?} (GPU {})", arch, card.gpu_id),
                vendor: Vendor::NVIDIA,
                arch,
                vram_size: card.gpu_memory_size,
                compute_units: 0,
                max_vgprs: 255,
                max_sgprs: 0,
                lds_size_per_cu: 49152,
                wave_size: 32,
                clock_mhz: 0,
                memory_bandwidth_gbps: 0.0,
                compute_tflops: 0.0,
                supports_fp16: true,
                supports_bf16: true,
                supports_fp8: true,
                supports_fp4: true,
                supports_wmma: false,
                supports_tensor_core: true,
            });
        }

        devices
    }

    fn open(&self, device_id: u32) -> Result<Box<dyn GpuDevice>, String> {
        if !self.available {
            return Err("NVIDIA driver not available".into());
        }

        // 打开设备文件
        let ctl_fd = OpenOptions::new().read(true).write(true).open("/dev/nvidiactl")
            .map_err(|e| format!("Failed to open /dev/nvidiactl: {}", e))?;

        let uvm_fd = OpenOptions::new().read(true).write(true).open("/dev/nvidia-uvm")
            .map_err(|e| format!("Failed to open /dev/nvidia-uvm: {}", e))?;

        // 枚举获取 minor number
        let mut cards = [NvCardInfo::default(); 64];
        let cmd = Self::ioctl_nr(NV_ESC_CARD_INFO, std::mem::size_of_val(&cards));
        unsafe { libc::ioctl(ctl_fd.as_raw_fd(), cmd, cards.as_mut_ptr()); }

        let card = cards.iter().find(|c| c.valid != 0)
            .ok_or("No NVIDIA GPU found")?;

        // 打开 per-GPU 设备文件
        let dev_path = format!("/dev/nvidia{}", card.minor_number);
        let dev_fd = OpenOptions::new().read(true).write(true).open(&dev_path)
            .map_err(|e| format!("Failed to open {}: {}", dev_path, e))?;

        // 注册 fd
        #[repr(C)]
        struct RegisterFd { ctl_fd: i32 }
        let mut reg = RegisterFd { ctl_fd: ctl_fd.as_raw_fd() };
        let cmd = Self::ioctl_nr(NV_ESC_REGISTER_FD, std::mem::size_of_val(&reg));
        unsafe { libc::ioctl(dev_fd.as_raw_fd(), cmd, &mut reg); }

        // 创建 root client
        let root = rm_alloc(&ctl_fd, NV01_ROOT_CLIENT, 0, None)?;

        // 创建 device
        let device = rm_alloc(&ctl_fd, NV01_DEVICE_0, root, None)?;

        // 创建 subdevice
        let subdevice = rm_alloc(&ctl_fd, NV20_SUBDEVICE_0, device, None)?;

        // 初始化 UVM
        uvm_init(&uvm_fd)?;

        Ok(Box::new(NvDevice {
            ctl_fd,
            uvm_fd,
            dev_fd,
            root,
            device,
            subdevice,
            gpu_id: card.gpu_id,
            minor: card.minor_number,
            info: DeviceInfo::default(),
        }))
    }

    fn is_available(&self) -> bool { self.available }
    fn name(&self) -> &str { "NVIDIA" }
}

// ═══════════════════════════════════════════════════════
// NvDevice — NVIDIA GPU 设备
// ═══════════════════════════════════════════════════════

struct NvDevice {
    ctl_fd: File,
    uvm_fd: File,
    dev_fd: File,
    root: u32,
    device: u32,
    subdevice: u32,
    gpu_id: u32,
    minor: u32,
    info: DeviceInfo,
}

unsafe impl Send for NvDevice {}
unsafe impl Sync for NvDevice {}

impl GpuDevice for NvDevice {
    fn info(&self) -> &DeviceInfo { &self.info }

    fn alloc(&self, size: usize, mem_type: MemType) -> Result<GpuMemory, String> {
        // 简化实现: 使用 mmap 分配匿名内存
        // TODO: 使用 RM alloc + UVM map 实现真正的 GPU 内存分配
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                prot,
                flags,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
        }

        Ok(GpuMemory {
            device_addr: ptr as u64,
            host_ptr: Some(ptr as u64),
            size,
            mem_type,
            handle: 0,
        })
    }

    fn free(&self, mem: GpuMemory) -> Result<(), String> {
        if let Some(ptr) = mem.host_ptr {
            unsafe { libc::munmap(ptr as *mut libc::c_void, mem.size); }
        }
        Ok(())
    }

    fn map_to_cpu(&self, mem: &GpuMemory) -> Result<*mut u8, String> {
        Ok(mem.host_ptr.unwrap_or(0) as *mut u8)
    }

    fn unmap_from_cpu(&self, _mem: &GpuMemory) -> Result<(), String> { Ok(()) }

    fn copy_from_host(&self, dst: &GpuMemory, src: &[u8]) -> Result<(), String> {
        let ptr = dst.host_ptr.ok_or("Not CPU-mapped")? as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len()); }
        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: &GpuMemory) -> Result<(), String> {
        let ptr = src.host_ptr.ok_or("Not CPU-mapped")? as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()); }
        Ok(())
    }

    fn copy_device(&self, _dst: &GpuMemory, _src: &GpuMemory, _size: usize) -> Result<(), String> {
        Err("NVIDIA copy_device not yet implemented".into())
    }

    fn create_compute_queue(&self, _config: QueueConfig) -> Result<Box<dyn ComputeQueue>, String> {
        // GPFIFO 队列创建
        // 参考 tinygrad ops_nv.py _new_gpu_fifo
        //
        // 1. 分配 gpfifo_area (VRAM, write-combined)
        // 2. 分配 notifier buffer (uncached)
        // 3. rm_alloc(GPFIFO_CLASS) — 创建 GPFIFO 通道
        // 4. 分配 compute engine 对象
        // 5. UVM_REGISTER_CHANNEL
        // 6. GET_WORK_SUBMIT_TOKEN — doorbell token

        // 简化实现: 使用 mmap 分配 ring buffer
        let ring_size = 4 * 1024 * 1024; // 4MB
        let ring_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ring_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ring_ptr == libc::MAP_FAILED {
            return Err("Failed to allocate GPFIFO ring".into());
        }

        // 分配 doorbell 页
        let doorbell_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if doorbell_ptr == libc::MAP_FAILED {
            return Err("Failed to allocate doorbell".into());
        }

        Ok(Box::new(NvGpfifoQueue {
            ring: ring_ptr as *mut u64,
            ring_size: ring_size / 8, // 以 u64 为单位
            put: 0,
            doorbell: doorbell_ptr as *mut u32,
            doorbell_token: 0,
        }))
    }

    fn create_copy_queue(&self) -> Result<Box<dyn CopyQueue>, String> {
        Err("NVIDIA copy queue not yet implemented".into())
    }

    fn create_signal(&self, _initial_value: u64) -> Result<Box<dyn Signal>, String> {
        // 简化: 使用匿名 mmap 作为信号量
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err("Failed to allocate signal".into());
        }

        Ok(Box::new(NvSignal { host_ptr: ptr as *mut u8, gpu_addr: ptr as u64 }))
    }

    fn wait_idle(&self) -> Result<(), String> { Ok(()) }

    fn load_kernel(&self, _elf_bytes: &[u8], _name: &str) -> Result<Box<dyn Kernel>, String> {
        // TODO: 加载 CUBIN/ELF
        Err("NVIDIA kernel loading not yet implemented".into())
    }
}

// ═══════════════════════════════════════════════════════
// NvSignal
// ═══════════════════════════════════════════════════════

struct NvSignal {
    host_ptr: *mut u8,
    gpu_addr: u64,
}

unsafe impl Send for NvSignal {}
unsafe impl Sync for NvSignal {}

impl Signal for NvSignal {
    fn value(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.host_ptr as *const u64) }
    }

    fn set(&self, value: u64) {
        unsafe { std::ptr::write_volatile(self.host_ptr as *mut u64, value); }
    }

    fn wait(&self, expected: u64, timeout: Duration) -> Result<(), String> {
        let start = std::time::Instant::now();
        loop {
            if self.value() == expected { return Ok(()); }
            if start.elapsed() > timeout {
                return Err(format!("Signal wait timeout ({}ms)", timeout.as_millis()));
            }
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    fn gpu_addr(&self) -> u64 { self.gpu_addr }
}

// ═══════════════════════════════════════════════════════
// NvGpfifoQueue — GPFIFO 计算队列
// ═══════════════════════════════════════════════════════

/// GPFIFO 队列条目 (64-bit)
/// bits [41:0] = cmdq_addr >> 2
/// bit  [41]   = sync flag
/// bits [63:42] = length (32-bit words)
struct NvGpfifoQueue {
    ring: *mut u64,
    ring_size: usize,
    put: usize,
    doorbell: *mut u32,
    doorbell_token: u32,
}

unsafe impl Send for NvGpfifoQueue {}
unsafe impl Sync for NvGpfifoQueue {}

impl ComputeQueue for NvGpfifoQueue {
    fn submit(
        &mut self,
        kernel: &dyn Kernel,
        grid: Grid,
        block: Block,
        kernargs: &[u8],
        signal: Option<&dyn Signal>,
    ) -> Result<(), String> {
        // 构造 QMD (Queue Meta Data)
        // 参考 tinygrad ops_nv.py QMD 类
        //
        // QMD v3: 256 bytes (0x40 dwords)
        // QMD v5: 384 bytes (0x60 dwords) — Blackwell

        let mut qmd = [0u32; 64]; // QMD v3: 64 dwords = 256 bytes

        // QMD 头部
        qmd[0] = 3 << 0;  // qmd_major_version = 3
        qmd[0] |= 1 << 12; // qmd_type = GRID_CTA

        // 程序地址
        let prog_addr = kernel.gpu_addr();
        qmd[2] = (prog_addr >> 32) as u32;  // program_address_upper
        qmd[3] = prog_addr as u32;           // program_address_lower

        // 寄存器数量
        qmd[4] = kernel.vgpr_count(); // register_count

        // Shared memory
        qmd[5] = kernel.lds_size(); // shared_memory_size

        // Grid 维度
        qmd[6] = grid.0;  // grid_width
        qmd[7] = grid.1;  // grid_height
        qmd[8] = grid.2;  // grid_depth

        // Block 维度
        qmd[9] = block.0 as u32;  // block_width
        qmd[10] = block.1 as u32; // block_height
        qmd[11] = block.2 as u32; // block_depth

        // 常量缓冲区 (kernarg)
        // TODO: 将 kernargs 复制到 constant buffer

        // 写入 GPFIFO ring
        let ring_entry = (1u64 << 41)  // sync flag
            | ((qmd.len() as u64) << 42); // length

        unsafe {
            std::ptr::write_volatile(
                self.ring.add(self.put % self.ring_size),
                ring_entry,
            );
        }

        // 更新 put pointer
        self.put += 1;

        // Doorbell write
        unsafe {
            std::ptr::write_volatile(self.doorbell, self.doorbell_token);
        }

        Ok(())
    }

    fn barrier(&mut self, _signals: &[&dyn Signal]) -> Result<(), String> {
        // GPFIFO 本身是顺序执行的, 不需要显式 barrier
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        // 确保 doorbell 已写入
        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    fn wait_idle(&mut self) -> Result<(), String> {
        // 等待 put == get
        // TODO: 读取 hardware get pointer
        Ok(())
    }

    fn pending_count(&self) -> usize {
        0 // TODO: 计算 pending dispatches
    }
}

// ═══════════════════════════════════════════════════════
// 辅助结构体
// ═══════════════════════════════════════════════════════

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NvCardInfo {
    valid: u32,
    gpu_id: u32,
    minor_number: u32,
    bus: u8,
    device: u8,
    function: u8,
    vendor_id: u16,
    device_id: u16,
    subvendor_id: u16,
    subdevice_id: u16,
    pci_device_id: u32,
    board_id: u32,
    gpu_memory_size: u64,
}

// ═══════════════════════════════════════════════════════
// RM ioctl 辅助函数
// ═══════════════════════════════════════════════════════

fn rm_alloc(fd: &File, class: u32, parent: u32, params: Option<*mut u8>) -> Result<u32, String> {
    #[repr(C)]
    struct RmAllocArgs {
        root: u32,
        parent: u32,
        new_handle: u32,
        class: u32,
        params_ptr: u64,
        params_size: u32,
        status: u32,
    }

    let mut args = RmAllocArgs {
        root: 0,
        parent,
        new_handle: 0,
        class,
        params_ptr: params.map(|p| p as u64).unwrap_or(0),
        params_size: 0,
        status: 0,
    };

    let cmd = NvDriver::ioctl_nr(NV_ESC_RM_ALLOC, std::mem::size_of_val(&args));
    let ret = unsafe { libc::ioctl(fd.as_raw_fd(), cmd, &mut args) };

    if ret < 0 || args.status != 0 {
        return Err(format!("rm_alloc failed: ret={} status={}", ret, args.status));
    }

    Ok(args.new_handle)
}

fn uvm_init(fd: &File) -> Result<(), String> {
    #[repr(C)]
    struct UvmInitArgs {
        flags: u32,
        rm_status: u32,
    }

    let mut args = UvmInitArgs { flags: 0, rm_status: 0 };
    let ret = unsafe { libc::ioctl(fd.as_raw_fd(), UVM_INITIALIZE, &mut args) };

    if ret < 0 || args.rm_status != 0 {
        return Err(format!("UVM init failed: ret={} status={}", ret, args.rm_status));
    }

    Ok(())
}
