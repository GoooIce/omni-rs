use std::collections::HashMap;

struct Entry<V> {
    value: V,
    bytes: usize,
    used: u64,
}

/// O(1) lookup; scan only when an insertion actually needs an eviction.
pub(crate) struct Cache<V> {
    entries: HashMap<Vec<u8>, Entry<V>>,
    clock: u64,
    pub bytes: usize,
}

impl<V> Default for Cache<V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            bytes: 0,
        }
    }
}

impl<V> Cache<V> {
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn sum(&self, size: impl Fn(&V) -> usize) -> usize {
        self.entries.values().map(|e| size(&e.value)).sum()
    }
    pub fn count(&self, predicate: impl Fn(&V) -> bool) -> usize {
        self.entries
            .values()
            .filter(|e| predicate(&e.value))
            .count()
    }
    /// Select a program to unload without removing its compiled graph. Looking
    /// for a victim is not a use and must not make that entry recently used.
    pub fn victim_mut(
        &mut self,
        prefer_below: usize,
        predicate: impl Fn(&V) -> bool,
    ) -> Option<&mut V> {
        self.entries
            .values_mut()
            .filter(|e| predicate(&e.value))
            .min_by_key(|e| (e.bytes >= prefer_below, e.used))
            .map(|e| &mut e.value)
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
    pub fn take(&mut self, key: &[u8]) -> Option<(V, usize)> {
        self.entries.remove(key).map(|e| {
            self.bytes -= e.bytes;
            (e.value, e.bytes)
        })
    }
    pub fn get(&mut self, key: &[u8]) -> Option<&V> {
        let entry = self.entries.get_mut(key)?;
        self.clock += 1;
        entry.used = self.clock;
        Some(&entry.value)
    }
    /// Drop a cached entry, optionally preferring cheap entries before LRU.
    pub fn evict(&mut self, prefer_below: usize) -> bool {
        self.evict_where(prefer_below, |_| true)
    }
    /// Remove only eligible entries; ineligible entries keep their age and budget.
    pub fn evict_where(&mut self, prefer_below: usize, predicate: impl Fn(&V) -> bool) -> bool {
        let key = self
            .entries
            .iter()
            .filter(|(_, e)| predicate(&e.value))
            .min_by_key(|(_, e)| (e.bytes >= prefer_below, e.used))
            .map(|(k, _)| k.clone());
        key.is_some_and(|key| self.take(&key).is_some())
    }
    pub fn insert(
        &mut self,
        key: Vec<u8>,
        value: V,
        bytes: usize,
        count: usize,
        budget: usize,
    ) -> u64 {
        self.take(&key);
        if bytes > budget || count == 0 {
            return 0;
        }
        let mut evictions = 0;
        while self.entries.len() >= count || self.bytes > budget - bytes {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| k.clone());
            let Some(oldest) = oldest else { break };
            self.take(&oldest);
            evictions += 1;
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                used: self.clock,
            },
        );
        evictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_budget_and_recent_use_determine_eviction() {
        let mut c = Cache::default();
        c.insert(vec![1], "a", 6, 3, 12);
        c.insert(vec![2], "b", 6, 3, 12);
        let (a, bytes) = c.take(&[1]).unwrap();
        c.insert(vec![1], a, bytes, 3, 12);
        assert_eq!(c.insert(vec![3], "c", 6, 3, 12), 1);
        assert!(c.take(&[2]).is_none());
        assert_eq!(c.bytes, 12);
        c.insert(vec![4], "oversized", 13, 3, 12);
        assert_eq!(c.len(), 2);
        c.clear();
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn program_slot_eviction_keeps_expensive_weighted_models() {
        let mut c = Cache::default();
        c.insert(vec![1], "weights", 4 * 1024 * 1024, 8, 8 * 1024 * 1024);
        c.insert(vec![2], "attention", 1024, 8, 8 * 1024 * 1024);
        assert!(c.evict(1024 * 1024));
        assert!(c.get(&[1]).is_some());
        assert!(c.get(&[2]).is_none());
        assert!(c.evict(0));
        assert!(!c.evict(0));
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn unloading_preserves_age_and_byte_budget_and_skips_unloaded_entries() {
        let mut c = Cache::default();
        c.insert(vec![1], true, 6, 3, 12);
        c.insert(vec![2], true, 6, 3, 12);
        *c.victim_mut(0, |loaded| *loaded).unwrap() = false;
        assert_eq!(c.count(|loaded| *loaded), 1);
        assert_eq!(c.len(), 2);
        assert_eq!(c.bytes, 12);
        // An unloaded entry remains the oldest: releasing the program slot
        // must not protect it from subsequent compiled-description eviction.
        assert_eq!(c.insert(vec![3], true, 6, 3, 12), 1);
        assert!(c.get(&[1]).is_none());
        assert_eq!(c.count(|loaded| *loaded), 2);
        *c.victim_mut(0, |loaded| *loaded).unwrap() = false;
        *c.victim_mut(0, |loaded| *loaded).unwrap() = false;
        assert!(c.victim_mut(0, |loaded| *loaded).is_none());
        assert_eq!(c.bytes, 12);
    }

    #[test]
    fn filtered_eviction_uses_lru_without_invalidating_external_owners() {
        use std::sync::Arc;
        let mut c = Cache::default();
        let retained = Arc::new(10);
        c.insert(vec![1], (true, Arc::new(1)), 4, 4, 20);
        c.insert(vec![2], (false, retained.clone()), 6, 4, 20);
        c.insert(vec![3], (false, Arc::new(20)), 8, 4, 20);
        assert!(c.evict_where(0, |(loaded, _)| !loaded));
        assert_eq!(Arc::strong_count(&retained), 1);
        assert_eq!(*retained, 10);
        assert_eq!(c.bytes, 12);
        assert!(c.take(&[2]).is_none());
        assert!(c.evict_where(0, |(loaded, _)| !loaded));
        assert!(!c.evict_where(0, |(loaded, _)| !loaded));
        assert_eq!(c.bytes, 4);
        assert_eq!(c.len(), 1);
        assert_eq!(c.take(&[1]).unwrap().0.1.as_ref(), &1);
        assert_eq!(c.bytes, 0);
    }
}
