use super::*;
use paddock_engine::generator::Generator;

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0., f32::max);
    eprintln!("max logit difference {error}");
    assert!(error <= tol, "{error} > {tol}");
}
fn top(a: &[f32]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[test]
#[ignore = "requires elected Laguna XS Q4_K_M in PADDOCK_LAGUNA_TEST_MODEL and M5"]
fn elected_checkpoint_lifecycle() {
    let path = std::env::var("PADDOCK_LAGUNA_TEST_MODEL").unwrap();
    lifecycle(&path, 2048, 20270574592);
}

#[test]
#[ignore = "requires elected three-shard Laguna S UD-Q4_K_XL in PADDOCK_LAGUNA_S_TEST_MODEL and 128GiB M5"]
fn elected_s_checkpoint_lifecycle() {
    let path = std::env::var("PADDOCK_LAGUNA_S_TEST_MODEL").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(std::path::Path::new(&path)).unwrap();
    assert_eq!(map.shard_count(), 3);
    assert!(
        map.gguf().tensors.is_empty(),
        "elected first shard is metadata-only"
    );
    assert_eq!(map.tensor_count(), 814);
    assert_eq!(map.total_len(), 73395172000);
    for (ty, count) in [(0, 287), (8, 386), (12, 92), (13, 47), (14, 2)] {
        assert_eq!(
            map.tensor_infos().filter(|t| t.raw_type == ty).count(),
            count
        );
    }
    drop(map);
    let later = path.replace("00001-of-00003.gguf", "00002-of-00003.gguf");
    let error = Laguna::load(std::path::Path::new(&later), 4096, 4, None)
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("load it via the first shard"),
        "{error}"
    );
    lifecycle(&path, 3072, 73391436800);
}

fn lifecycle(path: &str, width: usize, payload: u64) {
    let path = std::path::Path::new(&path);
    for (ctx, batch) in [(0, 1), (32769, 1), (4096, 0), (4096, 65)] {
        assert!(Laguna::load(path, ctx, batch, None).is_err());
    }
    let limited = Laguna::load(path, 4096, 4, Some(1 << 20))
        .err()
        .expect("reject insufficient budget");
    assert!(matches!(limited, MetalError::Memory(_)), "{limited}");
    let mut m = Laguna::load(path, 4096, 4, None).unwrap();
    let used = m.device.allocated_bytes();
    let geometry = Geometry::for_width(width).unwrap();
    assert_eq!(m.geometry, geometry);
    assert_eq!(m.layers.len(), geometry.layers);
    assert_eq!(m.weight_bytes, payload);
    assert_eq!(
        m.layers.iter().filter(|l| l.experts.is_some()).count(),
        geometry.layers - 1
    );
    eprintln!(
        "weights={} KV={} scratch={} total={} budget={}",
        m.weight_bytes,
        m.kv_bytes,
        used - m.weight_bytes - m.kv_bytes,
        used,
        m.device.budget_bytes()
    );
    let boundary = (0..513).map(|i| 200 + (i % 41) as u32).collect::<Vec<_>>();
    let a = m.prefill(3, &boundary).unwrap();
    let lo = a.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("logit range {lo}..{hi}, top {}", top(&a));
    assert!(
        hi - lo > 1.,
        "a degenerate constant distribution is not a model lifecycle pass"
    );
    m.reset();
    let b = m.prefill(0, &boundary).unwrap();
    assert_eq!(m.take_prefill_reused(0), 512);
    close(&a, &b, 0.00001);
    m.reset();
    let prompts = (0..4)
        .map(|i| {
            (0..(145 + i * 7))
                .map(|j| 100 + i as u32 + (j % 31) as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut individual = Vec::new();
    for p in &prompts {
        m.reset();
        while m.radix.evict_lru(&mut m.pool).is_some() {}
        individual.push(m.prefill(0, p).unwrap());
    }
    m.reset();
    while m.radix.evict_lru(&mut m.pool).is_some() {}
    for (slot, p) in prompts.iter().enumerate() {
        m.prefill_begin(slot, p.clone()).unwrap();
    }
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        done.extend(m.forward_mixed(&[], 127).unwrap().1);
    }
    assert_eq!(done.len(), 4);
    for (slot, logits, n) in &done {
        assert_eq!(*n, prompts[*slot].len());
        close(logits, &individual[*slot], 0.1);
        assert_eq!(top(logits), top(&individual[*slot]));
    }
    let decodes = done
        .iter()
        .map(|r| (r.0, top(&r.1) as u32, r.2 as u32))
        .collect::<Vec<_>>();
    let decoded = m.forward_mixed(&decodes, 0).unwrap().0;
    assert_eq!(decoded.len(), 4 * VOCAB);
    // An actual mixed tick: a sparse live decoder and a different slot's
    // unfinished prompt share the projection/expert pass. Don't confuse a
    // pure-decode call to forward_mixed with this lifecycle contract.
    let (row, decoder) = decodes.iter().enumerate().find(|(_, r)| r.0 == 3).unwrap();
    m.reset();
    m.prefill(3, &prompts[3]).unwrap();
    m.prefill_begin(0, vec![227; 193]).unwrap();
    let mixed = m.forward_mixed(&[*decoder], 127).unwrap();
    assert!(mixed.1.is_empty());
    assert_eq!(mixed.0.len(), VOCAB);
    let expected = &decoded[row * VOCAB..(row + 1) * VOCAB];
    close(&mixed.0, expected, 0.1);
    assert_eq!(top(&mixed.0), top(expected));
    assert!(m.prefill_abort(0));
    m.release_inactive_slots(&[false; 4]);
    assert!(m.slots.iter().all(|s| s.history.is_empty()));
    m.prefill_begin(0, vec![103; 256]).unwrap();
    m.forward_mixed(&[], 31).unwrap();
    assert!(m.prefill_abort(0));
    assert!(m.pending.is_empty());
    assert!(m.slots[0].history.is_empty());
    assert!(m.prefill(4, &[1]).is_err());
    assert!(m.prefill(0, &[]).is_err());
    assert!(m.prefill(0, &[VOCAB as u32]).is_err());
    assert!(m.prefill(0, &vec![100; 4097]).is_err());
    assert_eq!(used, m.device.allocated_bytes());
}
