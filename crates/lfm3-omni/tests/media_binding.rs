use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use omni_lfm3::{
    media::{Segment, SegmentKind},
    prepared::{MediaIds, PreparedMedia, bind_embeddings, bind_speech_frames},
};

type Cpu = NdArray<f32>;
#[test]
fn media_embeddings_replace_only_their_ordered_placeholder_tokens() {
    let device = Default::default();
    let ids = [7, 100, 100, 8, 101, 101, 9, 102, 101, 101];
    let text = Tensor::<Cpu, 3>::from_data(TensorData::new(vec![0.25; 20], [1, 10, 2]), &device);
    let image = Tensor::from_data(TensorData::new(vec![1., 2., 3., 4.], [2, 2]), &device);
    let audio = Tensor::from_data(TensorData::new(vec![5., 6., 7., 8.], [2, 2]), &device);
    let video = Tensor::from_data(TensorData::new(vec![9., 10.], [1, 2]), &device);
    let soundtrack = Tensor::from_data(TensorData::new(vec![11., 12., 13., 14.], [2, 2]), &device);
    let media = vec![
        PreparedMedia::Image(image),
        PreparedMedia::Audio(audio),
        PreparedMedia::Video {
            frames: vec![video],
            soundtrack: Some(soundtrack),
            frame_indices: vec![0, 1],
            fps: 2.0,
            temporal_patch_size: 2,
        },
    ];
    let segments = vec![
        Segment {
            media_index: 0,
            kind: SegmentKind::Image,
            tokens: 2,
        },
        Segment {
            media_index: 1,
            kind: SegmentKind::Audio,
            tokens: 2,
        },
        Segment {
            media_index: 2,
            kind: SegmentKind::VideoFrame { frame: 0 },
            tokens: 1,
        },
        Segment {
            media_index: 2,
            kind: SegmentKind::VideoAudio { offset: 0 },
            tokens: 2,
        },
    ];
    let result = bind_embeddings(
        &ids,
        text,
        &segments,
        &media,
        MediaIds {
            image: 100,
            audio: 101,
            video: 102,
        },
    )
    .unwrap();
    assert_eq!(
        result.into_data().to_vec::<f32>().unwrap(),
        vec![
            0.25, 0.25, 1., 2., 3., 4., 0.25, 0.25, 5., 6., 7., 8., 0.25, 0.25, 9., 10., 11., 12.,
            13., 14.
        ]
    );
}

#[test]
fn assistant_speech_frames_replace_matching_slots_in_order() {
    let d = Default::default();
    let ids = [5, 99, 6, 99, 7];
    let text = Tensor::<Cpu, 3>::from_data(TensorData::new(vec![0.; 10], [1, 5, 2]), &d);
    let frames = vec![
        Tensor::from_data(TensorData::new(vec![1., 2.], [1, 1, 2]), &d),
        Tensor::from_data(TensorData::new(vec![3., 4.], [1, 1, 2]), &d),
    ];
    let actual = bind_speech_frames(&ids, text.clone(), 99, &frames).unwrap();
    assert_eq!(
        actual.into_data().to_vec::<f32>().unwrap(),
        vec![0., 0., 1., 2., 0., 0., 3., 4., 0., 0.]
    );
    assert!(bind_speech_frames(&ids, text, 99, &frames[..1]).is_err());
}
