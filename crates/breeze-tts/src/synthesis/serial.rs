use super::*;

pub(super) fn advance<B: Backend>(
    model: &Model<B>,
    request: &mut Synthesis<B>,
    cache: &mut TransformerCache<B>,
    cancel: &AtomicBool,
    emit: &mut impl FnMut(&[f32]) -> Result<()>,
) -> Result<Option<End>> {
    let hidden = request.hidden();
    let scores = Model::<B>::logits(
        model.head.forward(hidden.clone()),
        request.options.guidance_scale,
    )?;
    let Some(mut frame) = request.begin_frame(model, &scores, hidden)? else {
        return Ok(Some(End::Eos));
    };
    cache.reset();
    for book in 1..CODEBOOKS {
        if cancel.load(Ordering::Acquire) {
            return Ok(Some(End::Interrupted));
        }
        let scores = depth_logits(model, frame.input.clone(), cache, book);
        request.sample_depth(model, &mut frame, book, scores)?;
    }
    request.accept_frame(model, frame, cancel, true, emit)
}
