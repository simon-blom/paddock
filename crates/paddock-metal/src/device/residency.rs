//! Explicit lifetime residency for a pre-admitted fixed GPU working set.
//! Without it, idle private weight copies can be paged out during a large
//! load before the first inference command ever references them. Residency
//! is requested only below Apple's recommendation; no OS limit is modified.
use super::*;
use std::sync::Mutex;

pub(super) struct ResidentSet {
    groups: Mutex<Vec<Group>>,
    device: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    group_bytes: u64,
}
struct Group {
    raw: Obj<dyn MTLResidencySet>,
    bytes: u64,
}
// SAFETY: the Metal set has no UI/thread affinity. Every access is serialized
// by the mutex, and its final endResidency runs only after the last Arc drops.
unsafe impl Send for ResidentSet {}
unsafe impl Sync for ResidentSet {}

impl ResidentSet {
    pub(super) fn new(
        device: &Obj<dyn MTLDevice>,
        queue: &Obj<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        Self::with_group_bytes(device, queue, device.recommendedMaxWorkingSetSize() / 8)
    }
    fn with_group_bytes(
        device: &Obj<dyn MTLDevice>,
        queue: &Obj<dyn MTLCommandQueue>,
        group_bytes: u64,
    ) -> Result<Self> {
        assert!(group_bytes > 0);
        Ok(Self {
            groups: Mutex::new(vec![Self::make_group(device, queue)?]),
            device: device.clone(),
            queue: queue.clone(),
            group_bytes,
        })
    }
    fn make_group(device: &Obj<dyn MTLDevice>, queue: &Obj<dyn MTLCommandQueue>) -> Result<Group> {
        let descriptor = MTLResidencySetDescriptor::new();
        // SAFETY: the descriptor is freshly allocated and exclusively owned.
        unsafe {
            descriptor.setInitialCapacity(4096);
        }
        let set = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::Device(format!("fixed-resident working set: {e}")))?;
        set.requestResidency();
        queue.addResidencySet(&set);
        Ok(Group { raw: set, bytes: 0 })
    }
    pub(super) fn add(&self, buffer: &Obj<dyn MTLBuffer>) -> Result<usize> {
        let mut groups = self.groups.lock().unwrap_or_else(|e| e.into_inner());
        let size = buffer.length() as u64;
        // Metal makes residency decisions per set. Isolate large allocations
        // (notably PLE) instead of coupling them to the entire decoder. Reuse
        // emptied sets; never exceed the queue's 32-set platform limit.
        let index = if let Some(i) = groups
            .iter()
            .position(|g| g.bytes == 0 || g.bytes + size <= self.group_bytes)
        {
            i
        } else if groups.len() < 32 {
            groups.push(Self::make_group(&self.device, &self.queue)?);
            groups.len() - 1
        } else {
            return Err(MetalError::Memory(
                "fixed residency exceeds 32 bounded sets".into(),
            ));
        };
        let group = &mut groups[index];
        group.raw.addAllocation(ProtocolObject::from_ref(&**buffer));
        group.raw.commit();
        group.bytes += size;
        Ok(index)
    }
    pub(super) fn remove(&self, buffer: &Obj<dyn MTLBuffer>, index: usize) {
        let mut groups = self.groups.lock().unwrap_or_else(|e| e.into_inner());
        let group = &mut groups[index];
        group
            .raw
            .removeAllocation(ProtocolObject::from_ref(&**buffer));
        group.raw.commit();
        group.bytes -= buffer.length() as u64;
    }
    #[cfg(test)]
    pub(super) fn allocations(&self) -> usize {
        self.groups
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|g| g.raw.allocationCount())
            .sum()
    }
}
impl Drop for ResidentSet {
    fn drop(&mut self) {
        for group in self.groups.get_mut().unwrap_or_else(|e| e.into_inner()) {
            group.raw.endResidency();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_residency_isolates_large_buffers_and_reuses_empty_groups() {
        let mut d = MetalDevice::new(Some(8 << 20)).unwrap();
        let sets = Arc::new(ResidentSet::with_group_bytes(&d.raw, &d.queue, 65536).unwrap());
        d.residency = Some(sets.clone());
        let a = d.alloc(65536).unwrap();
        let b = d.alloc(131072).unwrap();
        let c = d.alloc(32768).unwrap();
        assert_eq!(sets.groups.lock().unwrap().len(), 3);
        assert_eq!(sets.allocations(), 3);
        drop(b);
        let large = d.alloc(131072).unwrap();
        assert_eq!(sets.groups.lock().unwrap().len(), 3);
        drop(d); // buffers and their residency remain valid
        drop((a, c, large));
        assert_eq!(sets.allocations(), 0);
        assert!(sets.groups.lock().unwrap().iter().all(|g| g.bytes == 0));
    }
}
