use super::*;

// Test-thread-local baseline selection only; absent from serving builds and
// unable to affect other tests running on different threads.
#[cfg(test)]
thread_local! {
    pub(super) static BASELINE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static CANONICAL_MLX_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_ATTENTION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static CANONICAL_PROMPT_PHASE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_AFFINE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Model-local election: shared loads help mixed IQ4 gate/Q5 up at R1, but
/// regress IQ4 R2 and Q5-only R1. Keep tiny alpha/beta on the original route.
/// Wider MPP prefill/verification and other families are unchanged.
pub(super) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    m: usize,
    workspace: &Buffer,
) {
    if planes[0].0.ty == crate::affine::AFFINE4 {
        return crate::affine::project(cmd, planes, input, m, workspace);
    }
    let eligible = planes
        .iter()
        .all(|(w, _)| matches!(w.ty, 12..=14 | 23) && w.n >= 1024);
    let iq4 = planes.iter().any(|(w, _)| w.ty == 23);
    let elected = match m {
        1 => planes.len() == 2 && iq4,
        2 => !iq4,
        3 | 4 => true,
        _ => false,
    };
    #[cfg(test)]
    let elected = elected && !BASELINE_FOR_TEST.with(|value| value.get());
    if eligible && elected {
        pair(cmd, planes, input, m);
    } else if planes.len() == 1 {
        planes[0]
            .0
            .linear(cmd, input, planes[0].1, m, 1., workspace);
    } else {
        projections(cmd, planes, input, m, workspace);
    }
}

/// F32 column-coarsened decode. The baseline in weights remains independent
/// for exact GPU comparison and other model families.
pub(super) fn pair(cmd: &Commands<'_>, planes: &[(&Weight, &Buffer)], input: &Buffer, m: usize) {
    assert!((1..=4).contains(&m) && (1..=3).contains(&planes.len()));
    let k = planes[0].0.k;
    assert!(
        planes
            .iter()
            .all(|(w, _)| w.k == k && matches!(w.ty, 12..=14 | 23))
    );
    let columns = if m == 1 { 8 } else { 32 };
    if planes.len() == 1 {
        let (w, out) = planes[0];
        cmd.dispatch(
            ["qwen_pair1", "qwen_pair2", "qwen_pair3", "qwen_pair4"][m - 1],
            &[&w.buffer, input, out],
            &[k as u32, w.n as u32, m as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(columns), 1, 1],
            128,
        );
    } else {
        let third = planes.get(2).unwrap_or(&planes[1]);
        cmd.dispatch(
            [
                "qwen_multi_pair1",
                "qwen_multi_pair2",
                "qwen_multi_pair3",
                "qwen_multi_pair4",
            ][m - 1],
            &[
                &planes[0].0.buffer,
                &planes[1].0.buffer,
                &third.0.buffer,
                input,
                planes[0].1,
                planes[1].1,
                third.1,
            ],
            &[
                k as u32,
                planes[0].0.n as u32,
                planes[1].0.n as u32,
                if planes.len() == 3 {
                    third.0.n as u32
                } else {
                    0
                },
                m as u32,
                planes[0].0.ty,
                planes[1].0.ty,
                third.0.ty,
            ],
            [
                planes.iter().map(|(w, _)| w.n.div_ceil(columns)).sum(),
                1,
                1,
            ],
            128,
        );
    }
}
