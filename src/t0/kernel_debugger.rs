//! Kernel Debugger for T0 GPU Kernels
//! 
//! This module provides tools for debugging GPU kernel execution issues:
//! - Probe insertion: write known values to specific memory locations
//! - Execution tracing: track which code paths are executed
//! - Memory validation: verify GPU writes are visible to CPU
//! 
//! Usage:
//! 1. Insert probes into kernel using `KernelDebugger::insert_probe()`
//! 2. Dispatch kernel
//! 3. Read back probes using `KernelDebugger::read_probes()`
//! 4. Analyze execution path based on which probes were written

use std::collections::HashMap;

/// Probe configuration
#[derive(Debug, Clone)]
pub struct Probe {
    /// Unique probe ID
    pub id: u32,
    /// Memory offset (in bytes) from buffer base
    pub offset: u32,
    /// Value to write (f32 as u32 bits)
    pub value: u32,
    /// Description of what this probe tests
    pub description: String,
}

/// Kernel debugger for tracking execution
pub struct KernelDebugger {
    /// List of probes to insert
    probes: Vec<Probe>,
    /// Expected values for validation
    expected: HashMap<u32, u32>,
}

impl KernelDebugger {
    /// Create a new kernel debugger
    pub fn new() -> Self {
        Self {
            probes: Vec::new(),
            expected: HashMap::new(),
        }
    }
    
    /// Insert a probe at the specified offset
    /// 
    /// # Arguments
    /// * `id` - Unique probe ID
    /// * `offset` - Memory offset in bytes
    /// * `value` - Value to write (as f32 bits)
    /// * `description` - What this probe tests
    pub fn insert_probe(&mut self, id: u32, offset: u32, value: f32, description: &str) {
        let probe = Probe {
            id,
            offset,
            value: value.to_bits(),
            description: description.to_string(),
        };
        self.probes.push(probe);
        self.expected.insert(offset, value.to_bits());
    }
    
    /// Get all probes for insertion into kernel
    pub fn get_probes(&self) -> &[Probe] {
        &self.probes
    }
    
    /// Validate probe values read from GPU memory
    /// 
    /// # Arguments
    /// * `memory` - Slice of GPU memory (as u32 values)
    /// * `base_offset` - Base offset of the buffer (in u32 units)
    /// 
    /// # Returns
    /// Map of probe ID -> (expected, actual) for mismatched probes
    pub fn validate_probes(&self, memory: &[u32], base_offset: u32) -> HashMap<u32, (u32, u32)> {
        let mut mismatches = HashMap::new();
        
        for probe in &self.probes {
            let idx = (probe.offset / 4) + base_offset;
            if idx < memory.len() as u32 {
                let actual = memory[idx as usize];
                if actual != probe.value {
                    mismatches.insert(probe.id, (probe.value, actual));
                }
            }
        }
        
        mismatches
    }
    
    /// Analyze execution path based on which probes were written
    /// 
    /// # Arguments
    /// * `memory` - Slice of GPU memory (as u32 values)
    /// * `base_offset` - Base offset of the buffer (in u32 units)
    /// 
    /// # Returns
    /// Execution analysis report
    pub fn analyze_execution(&self, memory: &[u32], base_offset: u32) -> ExecutionAnalysis {
        let mut written_probes = Vec::new();
        let mut missing_probes = Vec::new();
        
        for probe in &self.probes {
            let idx = (probe.offset / 4) + base_offset;
            if idx < memory.len() as u32 {
                let actual = memory[idx as usize];
                if actual == probe.value {
                    written_probes.push(probe.id);
                } else {
                    missing_probes.push(probe.id);
                }
            }
        }
        
        let execution_path = self.determine_execution_path(&written_probes);
        
        ExecutionAnalysis {
            total_probes: self.probes.len(),
            written_probes,
            missing_probes,
            execution_path,
        }
    }
    
    /// Determine execution path based on which probes were written
    fn determine_execution_path(&self, written_probes: &[u32]) -> String {
        let mut path = String::new();
        
        // Check if store phase executed
        let store_probes: Vec<u32> = self.probes.iter()
            .filter(|p| p.description.contains("store"))
            .map(|p| p.id)
            .collect();
        
        let store_written: Vec<u32> = store_probes.iter()
            .filter(|id| written_probes.contains(id))
            .copied()
            .collect();
        
        if store_written.is_empty() {
            path.push_str("STORE PHASE NOT EXECUTED");
        } else if store_written.len() == store_probes.len() {
            path.push_str("STORE PHASE FULLY EXECUTED");
        } else {
            path.push_str(&format!("STORE PHASE PARTIALLY EXECUTED ({}/{})", 
                store_written.len(), store_probes.len()));
        }
        
        path
    }
}

/// Execution analysis result
#[derive(Debug)]
pub struct ExecutionAnalysis {
    /// Total number of probes inserted
    pub total_probes: usize,
    /// List of probe IDs that were written
    pub written_probes: Vec<u32>,
    /// List of probe IDs that were NOT written
    pub missing_probes: Vec<u32>,
    /// Execution path analysis
    pub execution_path: String,
}

impl std::fmt::Display for ExecutionAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Kernel Execution Analysis ===")?;
        writeln!(f, "Total probes: {}", self.total_probes)?;
        writeln!(f, "Written probes: {:?}", self.written_probes)?;
        writeln!(f, "Missing probes: {:?}", self.missing_probes)?;
        writeln!(f, "Execution path: {}", self.execution_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_debugger() {
        let mut debugger = KernelDebugger::new();
        
        // Insert probes
        debugger.insert_probe(1, 0, 42.0, "store phase entry");
        debugger.insert_probe(2, 4, 123.0, "store phase middle");
        debugger.insert_probe(3, 8, 999.0, "store phase exit");
        
        // Simulate GPU memory (all NaN initially)
        let mut memory = vec![0x7FC00000u32; 16]; // NaN
        
        // Simulate store phase execution (write probe values)
        memory[0] = 42.0f32.to_bits();
        memory[1] = 123.0f32.to_bits();
        memory[2] = 999.0f32.to_bits();
        
        // Validate probes
        let mismatches = debugger.validate_probes(&memory, 0);
        assert!(mismatches.is_empty(), "All probes should match");
        
        // Analyze execution
        let analysis = debugger.analyze_execution(&memory, 0);
        assert_eq!(analysis.written_probes.len(), 3);
        assert!(analysis.execution_path.contains("FULLY EXECUTED"));
    }
}
