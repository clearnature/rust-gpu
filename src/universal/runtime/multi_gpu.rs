use crate::universal::core::{DeviceInfo, DeviceManager, GpuDevice, GpuMemory, MemType};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// 多 GPU 管理器
// ═══════════════════════════════════════════════════════

/// 多 GPU 设备集合
pub struct MultiGpuManager {
    pub devices: HashMap<u32, Arc<dyn GpuDevice>>,
    pub device_infos: Vec<DeviceInfo>,
}

impl MultiGpuManager {
    /// 自动发现并打开所有可用 GPU
    pub fn discover() -> Result<Self, String> {
        let mgr = DeviceManager::discover();
        let mut devices = HashMap::new();
        let mut device_infos = Vec::new();

        for info in mgr.devices() {
            match mgr.open(info.id) {
                Ok(dev) => {
                    device_infos.push(info.clone());
                    devices.insert(info.id, Arc::from(dev));
                    eprintln!("[MultiGPU] Opened device {}: {}", info.id, info.name);
                }
                Err(e) => {
                    eprintln!("[MultiGPU] Failed to open device {}: {}", info.id, e);
                }
            }
        }

        if devices.is_empty() {
            return Err("No GPU devices found".into());
        }

        eprintln!("[MultiGPU] {} devices available", devices.len());
        Ok(Self { devices, device_infos })
    }

    /// 获取设备数量
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// 获取所有设备信息
    pub fn device_infos(&self) -> &[DeviceInfo] {
        &self.device_infos
    }

    /// 获取指定设备
    pub fn get_device(&self, id: u32) -> Option<&Arc<dyn GpuDevice>> {
        self.devices.get(&id)
    }

    /// 获取所有设备 ID
    pub fn device_ids(&self) -> Vec<u32> {
        self.devices.keys().copied().collect()
    }

    /// 跨设备内存传输
    pub fn transfer(
        &self,
        dst_dev_id: u32,
        dst: &GpuMemory,
        src_dev_id: u32,
        src: &GpuMemory,
        size: usize,
    ) -> Result<(), String> {
        if dst_dev_id == src_dev_id {
            // 同设备: 直接 DMA
            let dev = self.devices.get(&dst_dev_id)
                .ok_or(format!("Device {} not found", dst_dev_id))?;
            return dev.copy_device(dst, src, size);
        }

        // 跨设备: CPU bounce buffer
        let src_dev = self.devices.get(&src_dev_id)
            .ok_or(format!("Device {} not found", src_dev_id))?;
        let dst_dev = self.devices.get(&dst_dev_id)
            .ok_or(format!("Device {} not found", dst_dev_id))?;

        // 1. 源设备 → CPU
        let mut host_buf = vec![0u8; size];
        src_dev.copy_to_host(&mut host_buf, src)?;

        // 2. CPU → 目标设备
        dst_dev.copy_from_host(dst, &host_buf)?;

        Ok(())
    }

    /// 广播: 一个设备的数据复制到所有其他设备
    pub fn broadcast(
        &self,
        src_dev_id: u32,
        src: &GpuMemory,
        size: usize,
    ) -> Result<Vec<GpuMemory>, String> {
        let src_dev = self.devices.get(&src_dev_id)
            .ok_or(format!("Device {} not found", src_dev_id))?;

        // 先读到 CPU
        let mut host_buf = vec![0u8; size];
        src_dev.copy_to_host(&mut host_buf, src)?;

        // 写到所有其他设备
        let mut buffers = Vec::new();
        for (&dev_id, dev) in &self.devices {
            let buf = dev.alloc(size, MemType::Vram)?;
            if dev_id != src_dev_id {
                dev.copy_from_host(&buf, &host_buf)?;
            }
            buffers.push(buf);
        }

        Ok(buffers)
    }

    /// AllReduce (简化版): 所有设备的 buffer 求和, 结果广播到所有设备
    pub fn allreduce_sum_f32(
        &self,
        buffers: &[(u32, GpuMemory)], // (device_id, buffer)
        n: usize,
    ) -> Result<(), String> {
        if buffers.len() < 2 {
            return Ok(());
        }

        // 1. 从所有设备读取到 CPU
        let mut host_sum = vec![0.0f32; n];
        for (dev_id, buf) in buffers {
            let dev = self.devices.get(dev_id)
                .ok_or(format!("Device {} not found", dev_id))?;
            let mut host_buf = vec![0u8; n * 4];
            dev.copy_to_host(&mut host_buf, buf)?;
            let values: &[f32] = unsafe {
                std::slice::from_raw_parts(host_buf.as_ptr() as *const f32, n)
            };
            for i in 0..n {
                host_sum[i] += values[i];
            }
        }

        // 2. 写回所有设备
        let sum_bytes: Vec<u8> = host_sum.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        for (dev_id, buf) in buffers {
            let dev = self.devices.get(dev_id)
                .ok_or(format!("Device {} not found", dev_id))?;
            dev.copy_from_host(buf, &sum_bytes)?;
        }

        Ok(())
    }
}
