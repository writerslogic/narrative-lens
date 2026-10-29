use narrative_lens::continuity::world_state::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
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

fn measure<T>(action: impl FnOnce() -> T) -> (T, serde_json::Value) {
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

enum Output {
    Full(WorldStateResult),
    Current(WorldStateCurrentResult),
}

impl Output {
    fn semantic_view(&self) -> serde_json::Value {
        let (facts, violations, knowledge, locations, flow, score, complexity) = match self {
            Self::Full(result) => (
                &result.snapshots.last().unwrap().facts,
                &result.violations,
                &result.character_knowledge,
                &result.location_timeline,
                &result.information_flow,
                result.continuity_score,
                result.world_complexity,
            ),
            Self::Current(result) => (
                &result.facts,
                &result.violations,
                &result.character_knowledge,
                &result.location_timeline,
                &result.information_flow,
                result.continuity_score,
                result.world_complexity,
            ),
        };
        let mut facts: Vec<_> = facts.iter().collect();
        facts.sort();
        serde_json::json!({"facts":facts,"violations":violations,"knowledge":knowledge,"locations":locations,"flow":flow,"score":score,"complexity":complexity})
    }

    fn counts(&self) -> (usize, usize, usize) {
        match self {
            Self::Full(result) => (
                result.snapshots.last().unwrap().facts.len(),
                result
                    .snapshots
                    .iter()
                    .map(|snapshot| snapshot.facts.len())
                    .sum(),
                0,
            ),
            Self::Current(result) => (
                result.facts.len(),
                result.facts.len(),
                result
                    .deltas
                    .iter()
                    .map(|delta| delta.new_facts.len() + delta.invalidated_facts.len())
                    .sum(),
            ),
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let scenes: usize = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(512);
    let repetitions: usize = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(5);
    let mode = args.get(3).map(String::as_str).unwrap_or("full");
    let workload = args.get(4).map(String::as_str).unwrap_or("growth");
    assert!((1..=2048).contains(&scenes) && (1..=20).contains(&repetitions));
    assert!(["full", "current", "deltas", "minimal"].contains(&mode));
    assert!(["growth", "activity"].contains(&workload));
    let source: Vec<_> = (0..scenes).map(|i| if workload == "growth" {
        format!("Alice learned that observation {i} holds. Bob learned that report {i} holds.")
    } else {
        format!("Alice was in chamber{}. Bob was in chamber{}. Alice told Bob that report {i} holds. Alice carried the key. Alice gave the key to Bob. Bob gave the key to Alice.", i % 5, i % 5)
    }).collect();
    let refs: Vec<_> = source.iter().map(String::as_str).collect();
    let cast = ["Alice", "Bob"];
    black_box(track_world_state(&["Alice learned that rain fell."], &cast));
    let mut samples = Vec::new();
    let mut counts = (0, 0, 0);
    let mut fingerprint = 0;
    for _ in 0..repetitions {
        let (result, sample) = measure(|| {
            if mode == "full" {
                Output::Full(track_world_state(black_box(&refs), &cast))
            } else {
                Output::Current(
                    track_world_state_current(
                        black_box(&refs),
                        &cast,
                        &WorldStateOptions {
                            include_deltas: mode == "deltas",
                            include_event_history: mode != "minimal",
                        },
                    )
                    .unwrap(),
                )
            }
        });
        counts = result.counts();
        if workload == "growth" {
            assert_eq!(counts.0, scenes * 2);
        }
        fingerprint = 0xcbf29ce484222325u64;
        for byte in serde_json::to_vec(&result.semantic_view()).unwrap() {
            fingerprint ^= byte as u64;
            fingerprint = fingerprint.wrapping_mul(0x100000001b3);
        }
        samples.push(sample);
        black_box(result);
    }
    let mut elapsed: Vec<_> = samples.iter().map(|v| v["ms"].as_f64().unwrap()).collect();
    elapsed.sort_by(f64::total_cmp);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "mode":mode, "workload":workload,"scenes":scenes,"facts":counts.0,"retainedFactEntries":counts.1,"deltaFactEntries":counts.2,"finalSemanticFingerprint":format!("{fingerprint:016x}"),
        "medianMs":elapsed[elapsed.len()/2],"repetitions":repetitions,"samples":samples,
        "profile":"release;default features;one process;identical generated corpus;regex/cast cache warmed;tracked allocator enabled;compilation excluded",
        "memoryScope":"Rust allocator requested live bytes above pre-call live allocation;retained includes returned result and any remaining lazy regex-cache growth;peak includes reducer intermediates;excludes caller input,allocator metadata,stack,binary,OS RSS,non-Rust allocations",
        "equivalenceScope":"full/current/deltas retain identical final facts,violations,character knowledge,event histories and scores;minimal deliberately omits optional histories and may have a different semantic fingerprint",
    })).unwrap());
}
