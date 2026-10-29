use narrative_lens::substrate::{
    embedding::{CorpusEmbedding, EmbeddingOptions},
    wordfreq::spelling_suggestions,
};
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

fn token(mut number: usize) -> String {
    let mut result = String::from("lexeme");
    for _ in 0..4 {
        result.push((b'a' + (number % 26) as u8) as char);
        number /= 26;
    }
    result
}

fn corpus(vocab: usize, mixed: bool) -> Vec<String> {
    let words: Vec<_> = (0..vocab).map(token).collect();
    let mut documents = Vec::new();
    for round in 0..8 {
        let mut text = String::new();
        for i in 0..vocab {
            let index = if mixed {
                (i * (2 * round + 1) + round * 37) % vocab
            } else {
                let block = i / 32 * 32;
                let size = (vocab - block).min(32);
                block + (i % 32 + round * 3) % size
            };
            text.push_str(&words[index]);
            text.push(' ');
            if (i + 1) % 64 == 0 {
                documents.push(std::mem::take(&mut text));
            }
        }
        if !text.is_empty() {
            documents.push(text);
        }
    }
    documents
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("spelling");
    let mut samples = Vec::new();
    let mut details = serde_json::json!({});
    if mode == "spelling" {
        let repeats: usize = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(5);
        let queries = [
            "teh",
            "recieve",
            "seperate",
            "acommodate",
            "definately",
            "occurence",
            "relevent",
            "writting",
            "adress",
            "somthing",
            "becuase",
            "thier",
            "unneccessary",
            "independant",
            "arguement",
            "enviroment",
            "begining",
            "succesful",
            "tommorrow",
            "zzzzzzzzzzzzzzzzzz",
        ];
        let (first, cold) = measure(|| spelling_suggestions("recieve", 5));
        black_box(first);
        let mut checksum = 0;
        for _ in 0..repeats {
            let (_, sample) = measure(|| {
                for query in &queries {
                    for result in black_box(spelling_suggestions(black_box(query), 5)) {
                        checksum += result.len();
                    }
                }
            });
            samples.push(sample);
        }
        details = serde_json::json!({"queriesPerSample":queries.len(),"coldFirstLookup":cold,"checksum":checksum});
    } else if mode == "embedding" || mode == "embedding-mixed" {
        let vocab: usize = args.get(2).map(|x| x.parse().unwrap()).unwrap_or(1000);
        let dim: usize = args.get(3).map(|x| x.parse().unwrap()).unwrap_or(100);
        let repeats: usize = args.get(4).map(|x| x.parse().unwrap()).unwrap_or(3);
        let documents = corpus(vocab, mode == "embedding-mixed");
        let refs: Vec<_> = documents.iter().map(String::as_str).collect();
        let opts = EmbeddingOptions {
            dim,
            max_vocab: vocab,
            ..Default::default()
        };
        let _ = CorpusEmbedding::train_checked(&["warm token warm token"], &opts).unwrap();
        for _ in 0..repeats {
            let (embedding, sample) = measure(|| {
                CorpusEmbedding::train_checked(black_box(&refs), black_box(&opts)).unwrap()
            });
            assert_eq!(embedding.vocab_size(), vocab);
            let mut fingerprint = 0xcbf29ce484222325u64;
            let mut within = 0.0;
            let mut across = 0.0;
            for i in 0..vocab {
                for value in embedding.word_vector(&token(i)).unwrap() {
                    for byte in value.to_bits().to_le_bytes() {
                        fingerprint ^= byte as u64;
                        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
                    }
                }
                if i < 64.min(vocab) && vocab >= 128 {
                    let block = i / 32 * 32;
                    within += embedding.similarity(&token(i), &token(block + (i % 32 + 7) % 32));
                    across += embedding.similarity(&token(i), &token((i + vocab / 2) % vocab));
                }
            }
            details = serde_json::json!({"vocab":embedding.vocab_size(),"dim":embedding.dim(),"tokens":8*vocab,"documents":refs.len(),"inputBytes":refs.iter().map(|x|x.len()).sum::<usize>(),"neighborPreview":embedding.nearest_words(&token(0),3)});
            details["vectorFingerprint"] = serde_json::json!(format!("{fingerprint:016x}"));
            if vocab >= 128 {
                details["withinBlockCosineMean"] = serde_json::json!(within / 64.0);
                details["crossBlockCosineMean"] = serde_json::json!(across / 64.0);
            }
            samples.push(sample);
            black_box(embedding);
        }
    } else {
        panic!("mode must be spelling, embedding, or embedding-mixed");
    }
    let mut durations: Vec<_> = samples.iter().map(|s| s["ms"].as_f64().unwrap()).collect();
    durations.sort_by(f64::total_cmp);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "mode":mode,"profile":"release; default features; one process; allocation accounting enabled; compilation excluded",
        "memoryScope":"Rust allocator requested live bytes above pre-call live allocation; excludes caller corpus, allocator metadata, stack, binary, OS RSS, non-Rust allocations; includes construction intermediates and retained output",
        "method":"spelling: cold first call then warmed samples of 20 queries; embeddings: identical generated corpus per sample after one tiny initialization warmup; no model runtime",
        "medianMs":durations[durations.len()/2],"details":details,"samples":samples
    })).unwrap());
}
