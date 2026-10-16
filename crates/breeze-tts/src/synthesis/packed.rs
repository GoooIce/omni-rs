use super::*;

pub(super) fn advance<B: Backend>(
    model: &Model<B>,
    members: &mut [Member<'_, B>],
    ready: &[usize],
    emit: &mut impl FnMut(usize, &[f32]) -> Result<()>,
    completed: &mut BTreeMap<usize, Result<SynthesisReport>>,
) -> Result<()> {
    let mut lane = 0;
    let spans: Vec<_> = ready
        .iter()
        .map(|&i| {
            let count = if members[i].request.uncond.is_some() {
                2
            } else {
                1
            };
            let start = lane;
            lane += count;
            (i, start, count)
        })
        .collect();
    let hidden = Tensor::cat(
        ready.iter().map(|&i| members[i].request.hidden()).collect(),
        0,
    );
    let scores = model.head.forward(hidden.clone());
    let vocab = scores.dims()[2];
    let scores = Tensor::cat(
        spans
            .iter()
            .map(|&(i, start, count)| {
                members[i].request.guided(scores.clone().slice([
                    start..start + count,
                    0..1,
                    0..vocab,
                ]))
            })
            .collect(),
        0,
    );
    let scores = pollster::block_on(scores.into_data_async())
        .map_err(|e| Error::Inference(e.to_string()))?
        .to_vec::<f32>()
        .map_err(|e| Error::Inference(e.to_string()))?;
    let mut frames = Vec::new();
    let mut ended = Vec::new();
    for (row, &(i, start, count)) in spans.iter().enumerate() {
        let member = &mut members[i];
        if member.cancel.load(Ordering::Acquire) {
            ended.push((i, End::Interrupted));
            continue;
        }
        let hidden = hidden
            .clone()
            .slice([start..start + count, 0..1, 0..hidden.dims()[2]]);
        match member
            .request
            .begin_frame(model, &scores[row * vocab..(row + 1) * vocab], hidden)?
        {
            Some(frame) => frames.push((i, frame)),
            None => ended.push((i, End::Eos)),
        }
    }
    if !frames.is_empty() {
        let mut cache = model.depth.cache();
        for book in 1..CODEBOOKS {
            if frames
                .iter()
                .all(|(i, _)| members[*i].cancel.load(Ordering::Acquire))
            {
                break;
            }
            let input = Tensor::cat(frames.iter().map(|(_, f)| f.input.clone()).collect(), 0);
            let scores = depth_logits(model, input, &mut cache, book);
            let mut start = 0;
            for (i, frame) in &mut frames {
                let count = frame.lanes;
                members[*i].request.sample_depth(
                    model,
                    frame,
                    book,
                    scores
                        .clone()
                        .slice([start..start + count, 0..1, 0..model.config.vocab_size]),
                )?;
                start += count;
            }
        }
        for (i, frame) in frames {
            let member = &mut members[i];
            if member.cancel.load(Ordering::Acquire) {
                ended.push((i, End::Interrupted));
                continue;
            }
            match member
                .request
                .accept_frame(model, frame, member.cancel, &mut |pcm| emit(i, pcm))
            {
                Ok(Some(end)) => ended.push((i, end)),
                Ok(None) => {}
                Err(error) => {
                    completed.insert(i, Err(error));
                }
            }
        }
    }
    for (i, end) in ended.into_iter().rev() {
        let member = &mut members[i];
        completed.insert(
            i,
            member
                .request
                .finish(model, end, member.cancel, &mut |pcm| emit(i, pcm)),
        );
    }
    Ok(())
}
