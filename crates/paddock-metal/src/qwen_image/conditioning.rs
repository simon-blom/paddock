//! Checked multimodal layouts. Token order, attention bounds and rotary
//! positions are distinct: an image occupies many rows but one grid frame.
use super::*;

pub(super) fn slots(
    ids: &[u32],
    drop: usize,
    pad: u32,
    grids: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != pad {
            i += 1;
            continue;
        }
        let start = i;
        while i < ids.len() && ids[i] == pad {
            i += 1;
        }
        runs.push((start, i - start));
    }
    if runs.len() != grids.len()
        || runs.iter().zip(grids).any(|(&(off, len), &(w, h))| {
            off < drop || w == 0 || h == 0 || w.checked_mul(h) != Some(len)
        })
    {
        return Err(error(
            "reference pictures do not match the prompt's image slots",
        ));
    }
    Ok(runs)
}

pub(super) fn text_positions(
    rows: usize,
    runs: &[(usize, usize)],
    grids: &[(usize, usize)],
) -> Vec<u32> {
    let mut out = Vec::with_capacity(rows * 3);
    let (mut row, mut cursor) = (0, 0);
    for (&(off, len), &(w, h)) in runs.iter().zip(grids) {
        while row < off {
            out.extend([cursor; 3]);
            cursor += 1;
            row += 1;
        }
        for y in 0..h {
            for x in 0..w {
                out.extend([cursor, cursor + y as u32, cursor + x as u32]);
            }
        }
        row += len;
        cursor += w.max(h) as u32;
    }
    while row < rows {
        out.extend([cursor; 3]);
        cursor += 1;
        row += 1;
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Segment {
    Text {
        start: usize,
        rows: usize,
    },
    Image {
        index: usize,
        width: usize,
        height: usize,
    },
}
impl Segment {
    pub fn rows(&self) -> usize {
        match *self {
            Self::Text { rows, .. } => rows,
            Self::Image { width, height, .. } => width * height,
        }
    }
}
pub(super) fn segments(
    total: usize,
    drop: usize,
    runs: &[(usize, usize)],
    grids: &[(usize, usize)],
) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut row = 0;
    for (index, (&(off, len), &(width, height))) in runs.iter().zip(grids).enumerate() {
        let start = off - drop;
        if start > row {
            out.push(Segment::Text {
                start: row,
                rows: start - row,
            });
        }
        out.push(Segment::Image {
            index,
            width,
            height,
        });
        row = start + len;
    }
    if total > row {
        out.push(Segment::Text {
            start: row,
            rows: total - row,
        });
    }
    out
}
pub(super) fn prefix_layout(segments: &[Segment]) -> (Vec<u32>, Vec<u32>, usize) {
    let mut pos = Vec::new();
    let mut meta = Vec::new();
    let (mut cursor, mut off) = (0, 0);
    for seg in segments {
        match *seg {
            Segment::Text { rows, .. } => {
                for i in 0..rows {
                    pos.extend([(cursor + i) as u32; 3]);
                    meta.extend([0, (off + i) as u32]);
                }
                cursor += rows;
            }
            Segment::Image { width, height, .. } => {
                for i in 0..width * height {
                    pos.extend([
                        cursor as u32,
                        ((i / width) as i32 - (height - height / 2) as i32) as u32,
                        ((i % width) as i32 - (width - width / 2) as i32) as u32,
                    ]);
                    meta.extend([0, (off + width * height - 1) as u32]);
                }
                cursor += width.max(height);
            }
        }
        off += seg.rows();
    }
    (pos, meta, cursor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slots_reject_missing_extra_wrong_size_and_system_images() {
        let ids = [1, 2, 9, 9, 9, 9, 3];
        assert_eq!(slots(&ids, 2, 9, &[(2, 2)]).unwrap(), [(2, 4)]);
        for grids in [vec![], vec![(1, 2)], vec![(2, 2), (1, 1)]] {
            assert!(slots(&ids, 2, 9, &grids).is_err());
        }
        assert!(slots(&ids, 3, 9, &[(2, 2)]).is_err());
    }
    #[test]
    fn rectangular_images_have_bidirectional_blocks_not_future_text() {
        let s = [
            Segment::Text { start: 0, rows: 2 },
            Segment::Image {
                index: 0,
                width: 4,
                height: 2,
            },
            Segment::Text { start: 3, rows: 1 },
            Segment::Image {
                index: 1,
                width: 2,
                height: 4,
            },
        ];
        let (pos, meta, frame) = prefix_layout(&s);
        assert_eq!(frame, 11);
        assert_eq!(&pos[6..9], &[2, (-1i32) as u32, (-2i32) as u32]);
        assert_eq!(&pos[30..33], &[6; 3]);
        let read: Vec<_> = meta.chunks_exact(2).map(|v| v[1]).collect();
        assert_eq!(&read[..2], &[0, 1]);
        assert_eq!(&read[2..10], &[9; 8]);
        assert_eq!(read[10], 10);
        assert_eq!(&read[11..], &[18; 8]);
        assert_eq!(
            text_positions(7, &[(2, 4)], &[(2, 2)]),
            [
                0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 4, 4, 4
            ]
        );
    }
}
