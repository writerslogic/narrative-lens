use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

struct TrackedAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static MEASURING: AtomicBool = AtomicBool::new(false);

fn allocated(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    if MEASURING.load(Ordering::Relaxed) {
        PEAK.fetch_max(live, Ordering::Relaxed);
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        REQUESTED.fetch_add(bytes, Ordering::Relaxed);
    }
}
unsafe impl GlobalAlloc for TrackedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, old, size) };
        if !new.is_null() {
            LIVE.fetch_sub(old.size(), Ordering::Relaxed);
            allocated(size);
        }
        new
    }
}
#[global_allocator]
static ALLOCATOR: TrackedAllocator = TrackedAllocator;

pub fn measure<T>(action: impl FnOnce() -> T) -> (T, serde_json::Value) {
    let initial = LIVE.load(Ordering::Relaxed);
    PEAK.store(initial, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REQUESTED.store(0, Ordering::Relaxed);
    MEASURING.store(true, Ordering::Relaxed);
    let start = Instant::now();
    let result = action();
    let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
    MEASURING.store(false, Ordering::Relaxed);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(initial);
    let retained = LIVE.load(Ordering::Relaxed).saturating_sub(initial);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let requested = REQUESTED.load(Ordering::Relaxed);
    let metrics = serde_json::json!({
        "ms":milliseconds,
        "peakAdditionalRequestedBytes":peak,
        "retainedAdditionalRequestedBytes":retained,
        "allocationCalls":allocations,
        "totalRequestedBytes":requested,
    });
    (result, metrics)
}
