//! Canvas M-RoPE position tables: the port of the official `get_rope_index`.
//!
//! Text spans use plain 1-D positions. Each image frame owns a 2-D canvas:
//! the thumbnail is `linspace`-resampled onto it (degenerates to `arange` when
//! unsliced), detail slices tile it row-major, and the `<image>`/`</image>`/
//! `<slice>`/`</slice>` markers are pinned to the canvas corners. The
//! generation-time 1-D offset (`rope_delta`) is `max(positions) + 1 - seq_len`.
use crate::chat::Grid;

/// Token ids the canvas layout recognizes as markers.
#[derive(Debug, Clone, Copy)]
pub struct Markers {
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

#[derive(Debug, Clone, Copy)]
struct Crop {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct Frame {
    /// Span of the thumbnail's pad tokens (exclusive end).
    thumb: Crop,
    /// One past the markers that close this frame (image end, slice ends...).
    frame_end: usize,
    slices: Vec<Crop>,
}

#[derive(Debug)]
struct Group {
    /// Index of the `<image>` marker opening the group.
    start: usize,
    frames: Vec<Frame>,
}

fn group_frames(input_ids: &[u32], types: &[u8], markers: &Markers) -> Vec<Group> {
    let is_marker = |id: u32| {
        id == markers.image_start
            || id == markers.image_end
            || id == markers.slice_start
            || id == markers.slice_end
            || id == markers.newline
    };
    // Maximal non-text runs are crops, in sequence order.
    let mut crops: Vec<Crop> = Vec::new();
    let mut start = None;
    for (index, &kind) in types.iter().enumerate() {
        if kind == 0 {
            if let Some(begin) = start.take() {
                crops.push(Crop {
                    start: begin,
                    end: index,
                });
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(begin) = start {
        crops.push(Crop {
            start: begin,
            end: types.len(),
        });
    }

    let mut groups: Vec<Group> = Vec::new();
    let mut previous_end = 0usize;
    for crop in crops {
        let is_slice = !groups.is_empty()
            && crop.start > 0
            && input_ids[crop.start - 1] == markers.slice_start;
        if is_slice {
            groups
                .last_mut()
                .unwrap()
                .frames
                .last_mut()
                .unwrap()
                .slices
                .push(crop);
        } else {
            let adjacent = groups.last().is_some_and(|_group| {
                input_ids[previous_end..crop.start.saturating_sub(1)]
                    .iter()
                    .all(|&id| is_marker(id))
            });
            let frame = Frame {
                thumb: crop,
                frame_end: 0,
                slices: Vec::new(),
            };
            if adjacent {
                groups.last_mut().unwrap().frames.push(frame);
            } else {
                groups.push(Group {
                    start: crop.start - 1,
                    frames: vec![frame],
                });
            }
        }
        previous_end = crop.end;
    }

    let seq_len = input_ids.len();
    let markers_after = |mut index: usize| -> usize {
        while index < seq_len && is_marker(input_ids[index]) {
            index += 1;
        }
        index
    };
    for group in &mut groups {
        for frame in &mut group.frames {
            let close = if frame.slices.is_empty() {
                frame.thumb.end
            } else {
                frame.slices.last().unwrap().end
            };
            frame.frame_end = markers_after(close);
        }
    }
    groups
}

/// `torch.linspace(0, stop, count)` with torch's ties-to-even rounding.
fn linspace(stop: usize, count: usize) -> Vec<i64> {
    if count == 1 {
        return vec![0];
    }
    let step = stop as f64 / (count - 1) as f64;
    (0..count)
        .map(|i| (i as f64 * step).round_ties_even() as i64)
        .collect()
}

/// 3-D positions for one grid on a canvas; mirrors `get_vision_position_ids`.
fn vision_positions(
    origin: i64,
    grid: Grid,
    canvas_height: usize,
    canvas_width: usize,
    h_offset: usize,
    w_offset: usize,
    merge_factor: usize,
) -> Vec<[i64; 3]> {
    let llm_h = grid.height / merge_factor;
    let llm_w = grid.width / merge_factor;
    let canvas_height = if canvas_height == 0 {
        llm_h
    } else {
        canvas_height
    };
    let canvas_width = if canvas_width == 0 {
        llm_w
    } else {
        canvas_width
    };
    let mut out = Vec::with_capacity(llm_h * llm_w);
    let h_coords = linspace(canvas_height - 1, llm_h);
    let w_coords = linspace(canvas_width - 1, llm_w);
    for h in &h_coords {
        for w in &w_coords {
            out.push([
                origin,
                h + h_offset as i64 + origin,
                w + w_offset as i64 + origin,
            ]);
        }
    }
    out
}

/// The `(3, seq)` canvas position table plus the generation-time delta.
pub(crate) fn rope_index(
    input_ids: &[u32],
    types: &[u8],
    target_sizes: &[Grid],
    merge_factor: usize,
    markers: &Markers,
) -> (Vec<[i64; 3]>, i64) {
    assert_eq!(input_ids.len(), types.len());
    let seq_len = input_ids.len();
    let mut positions: Vec<[i64; 3]> = (0..seq_len as i64).map(|p| [p, p, p]).collect();
    let mut grids = target_sizes.iter();
    let groups = group_frames(input_ids, types, markers);

    let mut current_pos: i64 = 0;
    let mut current_idx = 0usize;
    for group in &groups {
        if group.start > current_idx {
            for slot in &mut positions[current_idx..group.start] {
                *slot = [current_pos; 3];
                current_pos += 1;
            }
            current_idx = group.start;
        }
        let group_end = group.frames.last().unwrap().frame_end;
        for frame in &group.frames {
            let thumb_grid = *grids.next().expect("target size per crop");
            let frame_start = frame.thumb.start - 1;
            if frame_start > current_idx {
                let gap = (frame_start - current_idx) as i64;
                for slot in &mut positions[current_idx..frame_start] {
                    *slot = [current_pos; 3];
                    current_pos += 1;
                }
                current_pos += gap - (frame_start - current_idx) as i64 + 1;
            }
            let frame_end = frame.frame_end.min(group_end);
            let origin = current_pos;
            let halo = (origin - 1).max(0);

            let mut canvas_height = thumb_grid.height / merge_factor;
            let mut canvas_width = thumb_grid.width / merge_factor;
            #[allow(unused_assignments)]
            let mut num_rows = 0;
            let mut num_cols = 0;
            let mut slice_grid = Grid {
                height: 0,
                width: 0,
            };
            if !frame.slices.is_empty() {
                num_cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].start - frame.slices[k].end > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                num_rows = frame.slices.len() / num_cols.max(1);
                if num_rows * num_cols != frame.slices.len() {
                    num_rows = 1;
                    num_cols = frame.slices.len();
                }
                slice_grid = *grids.next().expect("target size per slice");
                canvas_height = num_rows * (slice_grid.height / merge_factor);
                canvas_width = num_cols * (slice_grid.width / merge_factor);
            }

            // Base coat, halo and closing marker pins.
            for slot in &mut positions[frame_start..frame_end] {
                *slot = [origin; 3];
            }
            positions[frame_start][1] = halo;
            positions[frame_start][2] = halo;
            if frame.thumb.end < frame_end {
                positions[frame.thumb.end][1] = origin + canvas_height as i64;
                positions[frame.thumb.end][2] = origin + canvas_width as i64;
            }

            let thumbnail = vision_positions(
                origin,
                thumb_grid,
                canvas_height,
                canvas_width,
                0,
                0,
                merge_factor,
            );
            for (slot, value) in positions[frame.thumb.start..frame.thumb.end]
                .iter_mut()
                .zip(thumbnail)
            {
                *slot = value;
            }

            for (k, slice) in frame.slices.iter().enumerate() {
                let h_off = (k / num_cols) * (slice_grid.height / merge_factor);
                let w_off = (k % num_cols) * (slice_grid.width / merge_factor);
                let grid = if k == 0 {
                    slice_grid
                } else {
                    *grids.next().expect("target size per slice")
                };
                let slice_h = grid.height / merge_factor;
                let slice_w = grid.width / merge_factor;

                let marker = slice.start - 1;
                if marker >= frame_start {
                    positions[marker][1] = origin + h_off as i64;
                    positions[marker][2] = origin + w_off as i64;
                }
                if slice.end < frame_end {
                    positions[slice.end][1] = origin + (h_off + slice_h) as i64 - 1;
                    positions[slice.end][2] = origin + (w_off + slice_w) as i64 - 1;
                }
                let values = vision_positions(origin, grid, 0, 0, h_off, w_off, merge_factor);
                for (slot, value) in positions[slice.start..slice.end].iter_mut().zip(values) {
                    *slot = value;
                }
            }

            // Newlines that end a slice row sit just past that row's right edge.
            for k in 0..frame.slices.len().saturating_sub(1) {
                let (gap_start, gap_end) = (frame.slices[k].end, frame.slices[k + 1].start);
                if gap_end - gap_start <= 2 {
                    continue;
                }
                let boundary_h = (k / num_cols + 1) * (slice_grid.height / merge_factor) - 1;
                let right_edge = num_cols * (slice_grid.width / merge_factor);
                for position in &mut positions[gap_start + 1..gap_end - 1] {
                    position[1] = origin + boundary_h as i64;
                    position[2] = origin + right_edge as i64;
                }
            }

            current_pos = origin + canvas_height.max(canvas_width) as i64 + 1;
            current_idx = frame_end;
        }
        if current_idx < group_end {
            for slot in &mut positions[current_idx..group_end] {
                *slot = [current_pos; 3];
                current_pos += 1;
            }
        }
        current_idx = group_end;
    }
    if current_idx < seq_len {
        for slot in &mut positions[current_idx..] {
            *slot = [current_pos; 3];
            current_pos += 1;
        }
    }
    let max_position = positions
        .iter()
        .flat_map(|p| p.iter())
        .copied()
        .max()
        .unwrap_or(0);
    (positions, max_position + 1 - seq_len as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/canvas-positions.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn grid(values: &[serde_json::Value]) -> Grid {
        Grid {
            height: values[0].as_u64().unwrap() as usize,
            width: values[1].as_u64().unwrap() as usize,
        }
    }

    #[test]
    fn canvas_positions_match_official_table() {
        let cases = fixtures();
        for (name, case) in cases.as_object().unwrap() {
            let ids: Vec<u32> = case["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let types: Vec<u8> = case["mm_token_type_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let target_sizes: Vec<Grid> = case["target_sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| grid(v.as_array().unwrap()))
                .collect();
            let markers = Markers {
                image_start: 248_078,
                image_end: 248_079,
                slice_start: 248_088,
                slice_end: 248_089,
                newline: 198,
            };
            let (positions, delta) = rope_index(&ids, &types, &target_sizes, 4, &markers);
            // The fixture stores (3, seq) channels; compare channel-major.
            let mut actual: Vec<Vec<i64>> = vec![Vec::new(); 3];
            for token in &positions {
                for (channel, value) in token.iter().enumerate() {
                    actual[channel].push(*value);
                }
            }
            let expected: Vec<Vec<i64>> = case["positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect()
                })
                .collect();
            let official_delta = case["rope_deltas"][0][0].as_i64().unwrap();
            assert_eq!(actual, expected, "{name} position table");
            assert_eq!(delta, official_delta, "{name} rope delta");
        }
    }
}
