use super::*;
use std::slice::from_ref;

fn key(offset: usize, tokens: usize, value: u8) -> ImageKey {
    ImageKey {
        hash: 17,
        rgb: vec![value; 3].into(),
        w: 1,
        h: 1,
        offset,
        tokens,
    }
}
#[test]
fn image_identity_checks_bytes_and_geometry_even_on_hash_collision() {
    let a = key(10, 81, 1);
    let b = key(10, 81, 2);
    assert!(!a.same_image(&b));
    let mut shifted = a.clone();
    shifted.offset = 30;
    assert!(a.same_image(&shifted));
    let mut reshape = a.clone();
    reshape.w = 3;
    assert!(!a.same_image(&reshape));
    assert_eq!(prefix_cut(from_ref(&a), from_ref(&a), 100), 100);
    assert_eq!(prefix_cut(from_ref(&a), &[b], 100), 10);
    assert_eq!(prefix_cut(from_ref(&a), &[], 100), 10);
    assert_eq!(prefix_cut(&[], from_ref(&a), 100), 10);
    assert_eq!(prefix_cut(from_ref(&a), &[shifted], 100), 10);
    for cut in 11..91 {
        assert_eq!(prefix_cut(from_ref(&a), from_ref(&a), cut), 10);
    }
    assert_eq!(prefix_cut(from_ref(&a), from_ref(&a), 91), 91);
}
#[test]
fn atomic_spans_preserve_decode_capacity_and_separate_image_domains() {
    let layout = Layout {
        ids: vec![0; 600],
        keys: vec![key(10, 256, 1), key(290, 280, 2)],
        images: Vec::new(),
    };
    assert_eq!(layout.grant(0, 128, 509), 10);
    assert_eq!(layout.grant(10, 128, 509), 256);
    assert_eq!(layout.grant(10, 128, 255), 0);
    assert_eq!(layout.grant(266, 246, 246), 24);
    assert_eq!(layout.grant(290, 1, 509), 280);
    assert_eq!(layout.grant(570, 512, 509), 30);
    assert_eq!(layout.grant(10, 0, 509), 0);
    for pos in 0..600 {
        assert_eq!(
            layout.limit(pos),
            if (10..266).contains(&pos) {
                265
            } else if (290..570).contains(&pos) {
                569
            } else {
                pos
            }
        );
    }
}
