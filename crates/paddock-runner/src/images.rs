//! `POST /v1/images/generations` - text-to-image, the OpenAI Images API in
//! its GPT-image-model form - and `POST /v1/images/edits`, the same with
//! reference pictures (multipart), served when the model's vision tower is
//! wired and refused by name when it is not.
//!
//! What is served: any `WIDTHxHEIGHT` on the model's 32-pixel grid up to
//! its native 2K (or `width` / `height` as Paddock extras), `n` images on
//! one seed (each a different draw), `quality` as a step count, `background:
//! transparent` as the model's own RGBA prompt form, and `output_format`
//! png / webp (alpha kept) / jpeg (flattened). Every image comes back as
//! `b64_json` - the GPT image models return nothing else, and neither does
//! this. `seed` defaults to 42, so an unadorned request is reproducible.
//!
//! `stream: true` is the GPT-image stream: `partial_images` (0-3)
//! progressive previews as `image_generation.partial_image` events - the
//! engine's x0 estimate decoded part way through the render, so a preview is
//! the picture the render is converging on - then `image_generation.completed`
//! carrying the final image and the usage. Data-only SSE events discriminated
//! by `type`, which is what the SDK's `Stream[ImageGenStreamEvent]` parses;
//! one image per streamed request, as upstream.
//!
//! What is refused, by name, with a 400: the DALL-E fields (`style`,
//! `response_format: url`), `n > 1` while streaming, previews without the
//! stream, and a size off the grid - the pipeline would silently round it and
//! the caller would never learn why their 1000x1000 came back 992x992.

use std::io::Cursor;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

use paddock_api::ErrorBody;
use paddock_api::images::{
    Background, ImageData, ImageEditParams, ImageGenerationRequest, ImageStreamEvent,
    ImageTokenDetails, ImageUsage, ImagesResponse, OutputFormat, parse_size,
};
use paddock_engine::gpu_model::qwen_image::{Rgba, t2i_prompt, ti2i_prompt};
use paddock_engine::image::{ImageEvent, ImageReply, ImageRequest, ReferenceImage};

use crate::extract::OaiJson;
use crate::routes::AppState;
use crate::serving::ImageModel;

/// The model's native ceiling per side (its own 16:9 recommendation is
/// 2752 x 1536).
pub const MAX_SIDE: usize = 2752;
/// `size: auto` and the default.
pub const DEFAULT_SIDE: usize = 1024;
/// The model's own default step count.
pub const DEFAULT_STEPS: usize = 40;
pub const MAX_STEPS: usize = 100;
pub const MAX_N: usize = 10;
/// Previews per streamed render - the API's own 0..3.
pub const MAX_PARTIAL_IMAGES: usize = 3;
/// Reference pictures per edit - the model card's own ceiling.
pub const MAX_REFERENCES: usize = 10;
/// What `/v1/models` advertises as `supported_parameters`.
pub const SUPPORTED_PARAMETERS: &[&str] = &[
    "prompt",
    "model",
    "n",
    "size",
    "width",
    "height",
    "quality",
    "output_format",
    "output_compression",
    "background",
    "moderation",
    "response_format",
    "user",
    "stream",
    "partial_images",
    "seed",
    "steps",
    "guidance",
    "negative_prompt",
];
/// The prompt form the model card recommends for a transparent image.
const RGBA_PREFIX: &str = "This is an RGBA image with transparency. ";
const RGBA_SUFFIX: &str = ". The image has alpha channel and the background is transparent.";

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn bad(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

/// The request, validated and resolved to what the engine takes.
struct Plan {
    prompt: String,
    negative: Option<String>,
    width: usize,
    height: usize,
    steps: usize,
    quality: String,
    n: usize,
    seed: u64,
    guidance: f32,
    format: OutputFormat,
    compression: u8,
    stream: bool,
    /// previews on the stream (0 when not streaming)
    partials: usize,
}

/// Validate and resolve the request. Every refusal here is a 400 with the
/// message returned, which the handler wraps. `default_size` is what `auto`
/// (and nothing) means: the model's own square for a generation, the last
/// reference's shape at that area for an edit.
fn plan(
    req: &ImageGenerationRequest,
    size_multiple: usize,
    default_size: (usize, usize),
) -> Result<Plan, String> {
    if req.prompt.trim().is_empty() {
        return Err("prompt is required".into());
    }
    if req.prompt.len() > 32_000 {
        return Err("prompt is longer than the 32000-character maximum".into());
    }
    if let Some(rf) = req.response_format.as_deref()
        && rf != "b64_json"
    {
        return Err(format!(
            "response_format '{rf}' is not served: images are returned as b64_json only"
        ));
    }
    if req.style.is_some() {
        return Err("style is a DALL-E 3 parameter; this model has no style setting".into());
    }
    let stream = req.stream == Some(true);
    let partials = req.partial_images.unwrap_or(0) as usize;
    if partials > MAX_PARTIAL_IMAGES {
        return Err(format!(
            "partial_images must be between 0 and {MAX_PARTIAL_IMAGES}"
        ));
    }
    if partials > 0 && !stream {
        return Err("partial_images needs stream: true - previews arrive as stream events".into());
    }
    if let Some(m) = req.moderation.as_deref()
        && !matches!(m, "low" | "auto")
    {
        return Err(format!("moderation '{m}': expected 'low' or 'auto'"));
    }
    let n = req.n.unwrap_or(1) as usize;
    if n == 0 || n > MAX_N {
        return Err(format!("n must be between 1 and {MAX_N}"));
    }
    if stream && n > 1 {
        return Err("stream renders one image at a time: n must be 1 when streaming".into());
    }
    // size: `size` or the width/height pair, never both
    let (width, height) = match (req.size.as_deref(), req.width, req.height) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err("pass either size or width/height, not both".into());
        }
        (None, None, None) | (Some("auto"), None, None) => default_size,
        (Some(s), None, None) => match parse_size(s) {
            Some((w, h)) => (w as usize, h as usize),
            None => return Err(format!("size '{s}': expected WIDTHxHEIGHT or auto")),
        },
        (None, w, h) => match (w, h) {
            (Some(w), Some(h)) => (w as usize, h as usize),
            _ => return Err("width and height must be given together".into()),
        },
    };
    for (what, v) in [("width", width), ("height", height)] {
        if v == 0 || v % size_multiple != 0 || v > MAX_SIDE {
            return Err(format!(
                "{what} {v} is off this model's grid: a multiple of {size_multiple} up to {MAX_SIDE}"
            ));
        }
    }
    // quality picks the step count; the steps extra overrides it. The echo
    // is the quality the render USES, never `auto`: the model's own default
    // is its full step count, so auto (and gpt-image-2.5's xhigh/max, which
    // have no higher rung here) answers as `high` - the SDK's response
    // literal is low | medium | high and a strict client reads it.
    let (quality, mut steps) = match req.quality.as_deref() {
        None | Some("auto" | "high" | "xhigh" | "max") => ("high".to_owned(), DEFAULT_STEPS),
        Some("medium") => ("medium".to_owned(), 30),
        Some("low") => ("low".to_owned(), 20),
        Some(q @ ("standard" | "hd")) => {
            return Err(format!(
                "quality '{q}' is DALL-E's; use auto, low, medium or high"
            ));
        }
        Some(q) => {
            return Err(format!("quality '{q}': expected auto, low, medium or high"));
        }
    };
    if let Some(s) = req.steps {
        steps = s as usize;
        if steps == 0 || steps > MAX_STEPS {
            return Err(format!("steps must be between 1 and {MAX_STEPS}"));
        }
    }
    let guidance = req.guidance.unwrap_or(1.0);
    if !guidance.is_finite() || guidance < 0.0 {
        return Err("guidance must be a non-negative number".into());
    }
    if req
        .negative_prompt
        .as_deref()
        .is_some_and(|s| !s.is_empty())
        && guidance <= 1.0
    {
        return Err(
            "negative_prompt needs guidance above 1; the model runs without guidance by default"
                .into(),
        );
    }
    let format = req.output_format.unwrap_or_default();
    let compression = match req.output_compression {
        None => 100,
        Some(_) if format == OutputFormat::Png => {
            return Err("output_compression applies to webp and jpeg only".into());
        }
        Some(c) if c > 100 => return Err("output_compression must be between 0 and 100".into()),
        Some(c) => c as u8,
    };
    let prompt = match req.background {
        Some(Background::Transparent) => {
            if !format.keeps_alpha() {
                return Err("background 'transparent' needs output_format png or webp".into());
            }
            format!(
                "{RGBA_PREFIX}{}{RGBA_SUFFIX}",
                req.prompt.trim().trim_end_matches('.')
            )
        }
        _ => req.prompt.clone(),
    };
    Ok(Plan {
        prompt,
        negative: req.negative_prompt.clone().filter(|s| !s.is_empty()),
        width,
        height,
        steps,
        quality,
        n,
        seed: req.seed.unwrap_or(42),
        guidance,
        format,
        compression,
        stream,
        partials,
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The alpha below which a pixel counts as see-through. An opaque image's
/// alpha plane idles a few counts under 255 (251-254 on both this engine and
/// sd.cpp - the decoder's own noise), so "any pixel below 255" would call
/// every image transparent; a transparent one has whole regions near 0.
const TRANSPARENT_ALPHA: u8 = 200;

/// Encode one decoded image. PNG/WebP keep the alpha plane when the model
/// used it (any clearly see-through pixel) and drop it otherwise; JPEG is
/// flattened over white.
fn encode(img: &Rgba, format: OutputFormat, compression: u8) -> Result<(Vec<u8>, bool), String> {
    let (w, h) = (img.width as u32, img.height as u32);
    let transparent = img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[3] < TRANSPARENT_ALPHA);
    let mut out = Cursor::new(Vec::new());
    match format {
        OutputFormat::Png | OutputFormat::Webp => {
            let f = if format == OutputFormat::Png {
                image::ImageFormat::Png
            } else {
                image::ImageFormat::WebP
            };
            if transparent {
                let im =
                    image::RgbaImage::from_raw(w, h, img.pixels.clone()).ok_or("rgba buffer")?;
                im.write_to(&mut out, f).map_err(|e| e.to_string())?;
            } else {
                let rgb: Vec<u8> = img
                    .pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .collect();
                let im = image::RgbImage::from_raw(w, h, rgb).ok_or("rgb buffer")?;
                im.write_to(&mut out, f).map_err(|e| e.to_string())?;
            }
        }
        OutputFormat::Jpeg => {
            let rgb: Vec<u8> = img
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let a = p[3] as u32;
                    [0, 1, 2].map(|c| ((p[c] as u32 * a + 255 * (255 - a)) / 255) as u8)
                })
                .collect();
            let im = image::RgbImage::from_raw(w, h, rgb).ok_or("rgb buffer")?;
            let enc =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, compression.max(1));
            im.write_with_encoder(enc).map_err(|e| e.to_string())?;
        }
    }
    Ok((out.into_inner(), transparent))
}

pub async fn generations(
    State(state): State<Arc<AppState>>,
    OaiJson(req): OaiJson<ImageGenerationRequest>,
) -> Response {
    let Some(model) = state.image.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no image-generation model is loaded (start paddock with a Qwen-Image DiT as `model`, \
             plus --text-encoder and --vae)",
        );
    };
    let p = match plan(
        &req,
        model.service.info().size_multiple,
        (DEFAULT_SIDE, DEFAULT_SIDE),
    ) {
        Ok(p) => p,
        Err(msg) => return bad(msg),
    };
    // the model's raw template; the system block's ids are dropped from the
    // conditioning, so both halves are tokenized
    let tokenize = |text: &str| -> Result<(Vec<u32>, usize), String> {
        let (system, full) = t2i_prompt(text);
        let sys = model.tokenizer.encode(&system).map_err(|e| e.to_string())?;
        let ids = model.tokenizer.encode(&full).map_err(|e| e.to_string())?;
        if ids.len() < sys.len() || ids[..sys.len()] != sys[..] {
            return Err("prompt template tokenized inconsistently".into());
        }
        Ok((ids, sys.len()))
    };
    let (prompt_ids, drop) = match tokenize(&p.prompt) {
        Ok(v) => v,
        Err(msg) => return bad(msg),
    };
    let negative = match p.negative.as_deref().map(tokenize) {
        Some(Ok(v)) => Some(v),
        Some(Err(msg)) => return bad(msg),
        None => None,
    };
    let request = ImageRequest {
        prompt_ids,
        drop,
        negative,
        width: p.width,
        height: p.height,
        steps: p.steps,
        seed: p.seed,
        guidance: p.guidance,
        n: p.n,
        partial_images: p.partials,
        references: Vec::new(),
        image_pad_id: model.image_pad_id.unwrap_or(0),
    };
    if p.stream {
        return stream_generations(model, p, request, 0);
    }
    let reply = match model.service.generate(request).await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    };
    finish(p, reply, 0).await
}

/// The oneshot response: every image encoded off the runtime, then the
/// usage. `image_tokens` are the reference pictures' tokens inside the
/// prompt (0 for a generation), reported apart from the text's.
async fn finish(p: Plan, reply: ImageReply, image_tokens: u64) -> Response {
    // encoding a 2K PNG is real CPU work - off the runtime
    let (format, compression) = (p.format, p.compression);
    let encoded = tokio::task::spawn_blocking(move || {
        reply
            .images
            .iter()
            .map(|img| encode(img, format, compression))
            .collect::<Result<Vec<_>, String>>()
            .map(|v| (v, reply.prompt_tokens, reply.latent_tokens))
    })
    .await;
    let (encoded, prompt_tokens, latent_tokens) = match encoded {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                e.to_string(),
            );
        }
    };
    let transparent = encoded.iter().any(|(_, t)| *t);
    let data = encoded
        .into_iter()
        .map(|(bytes, _)| ImageData {
            b64_json: Some(B64.encode(bytes)),
            revised_prompt: None,
            url: None,
        })
        .collect::<Vec<_>>();
    let n = data.len() as u64;
    let body = ImagesResponse {
        created: unix_now(),
        data,
        background: Some(if transparent { "transparent" } else { "opaque" }.to_owned()),
        output_format: Some(format),
        quality: Some(p.quality),
        size: Some(format!("{}x{}", p.width, p.height)),
        usage: Some(ImageUsage {
            input_tokens: prompt_tokens as u64,
            input_tokens_details: ImageTokenDetails {
                image_tokens,
                text_tokens: (prompt_tokens as u64).saturating_sub(image_tokens),
            },
            output_tokens: latent_tokens as u64 * n,
            total_tokens: prompt_tokens as u64 + latent_tokens as u64 * n,
        }),
    };
    Json(body).into_response()
}

/// The streaming form: the previews as `image_generation.partial_image`
/// events (`partial_image_index` 0..), then `image_generation.completed` with
/// the usage. Each event carries its picture encoded in the requested format,
/// off the runtime like the oneshot path. An engine failure goes out as an
/// `error` event, which the SDK raises, and the stream always ends with
/// `[DONE]`.
fn stream_generations(
    model: &ImageModel,
    p: Plan,
    request: ImageRequest,
    image_tokens: u64,
) -> Response {
    let mut rx = match model.service.generate_stream(request) {
        Ok(rx) => rx,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    };
    let (format, compression, quality) = (p.format, p.compression, p.quality);
    let size = format!("{}x{}", p.width, p.height);
    let created_at = unix_now();
    let sse = stream! {
        while let Some(ev) = rx.recv().await {
            let (image, index, usage) = match ev {
                ImageEvent::Partial { index, image } => (image, Some(index as u32), None),
                ImageEvent::Done(reply) => {
                    let Some(image) = reply.images.into_iter().next() else {
                        yield error_event("the render produced no image");
                        break;
                    };
                    let (pt, lt) = (reply.prompt_tokens as u64, reply.latent_tokens as u64);
                    let usage = ImageUsage {
                        input_tokens: pt,
                        input_tokens_details: ImageTokenDetails {
                            image_tokens,
                            text_tokens: pt.saturating_sub(image_tokens),
                        },
                        output_tokens: lt,
                        total_tokens: pt + lt,
                    };
                    (image, None, Some(usage))
                }
                ImageEvent::Error(e) => {
                    yield error_event(&e);
                    break;
                }
            };
            let done = index.is_none();
            let encoded =
                tokio::task::spawn_blocking(move || encode(&image, format, compression)).await;
            let (bytes, transparent) = match encoded {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    yield error_event(&e);
                    break;
                }
                Err(e) => {
                    yield error_event(&e.to_string());
                    break;
                }
            };
            let event = ImageStreamEvent {
                kind: if done {
                    "image_generation.completed"
                } else {
                    "image_generation.partial_image"
                }
                .to_owned(),
                b64_json: B64.encode(bytes),
                background: if transparent { "transparent" } else { "opaque" }.to_owned(),
                created_at,
                output_format: format,
                quality: quality.clone(),
                size: size.clone(),
                partial_image_index: index,
                usage,
            };
            yield Ok::<_, std::convert::Infallible>(
                Event::default().data(serde_json::to_string(&event).unwrap_or_default()),
            );
            if done {
                break;
            }
        }
        yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
    };
    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn error_event(msg: &str) -> Result<Event, std::convert::Infallible> {
    let body = serde_json::json!({ "error": { "message": msg, "type": "internal_error" } });
    Ok(Event::default().event("error").data(body.to_string()))
}

/// `POST /v1/images/edits` - the GPT-image edit form: `image` (one or
/// more, `image[]` in the SDKs) plus `prompt` and the generation fields,
/// as multipart. Every reference is resized to its own aspect at the
/// output area (1024^2) on the 32-pixel grid - the pipeline's rule - and
/// the output takes the LAST reference's shape unless `size` says
/// otherwise. `mask` is refused by name: this model edits from the
/// instruction and the pictures, not a mask.
///
/// The whole form is read before any refusal about it goes out: a response
/// written while the request body sits unread in the socket makes the
/// close a TCP reset, and a client on Windows then loses the response and
/// sees "connection aborted" instead of the 400 - which is how the
/// conformance gate first met this endpoint.
pub async fn edits(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let Some(model) = state.image.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no image-generation model is loaded",
        );
    };
    // the extractor by hand, so a body that is not a multipart form gets the
    // OpenAI error shape rather than axum's plain-text 400
    let mp = match Multipart::from_request(req, &state).await {
        Ok(mp) => mp,
        Err(e) => {
            return bad(format!(
                "the edit form must be multipart/form-data with the picture(s) as `image` ({e})"
            ));
        }
    };
    let params = match read_edit_params(mp).await {
        Ok(p) => p,
        Err(msg) => return bad(msg),
    };
    let Some(pad_id) = model.image_pad_id.filter(|_| model.service.info().edit) else {
        return bad(
            "editing needs the vision tower: start this model with its mmproj (the Qwen3-VL \
             vision file) - text-to-image is served without it",
        );
    };
    if params.images.is_empty() {
        return bad("image is required: at least one reference picture");
    }
    if params.images.len() > MAX_REFERENCES {
        return bad(format!(
            "{} reference pictures: this model takes up to {MAX_REFERENCES}",
            params.images.len()
        ));
    }
    if params.mask.is_some() {
        return bad(
            "mask is not served: this model edits from the instruction and the pictures, without a mask",
        );
    }
    if let Some(f) = params.input_fidelity.as_deref()
        && !matches!(f, "low" | "high")
    {
        return bad(format!("input_fidelity '{f}': expected low or high"));
    }
    // The shipped pipeline has ONE `output_resolution` (1024 by default) that
    // sizes both the references and the picture: each reference is fitted to
    // its own aspect at that area, and `auto` takes the last one's shape at
    // it. An explicit `size` therefore moves the reference area with it - a
    // 512^2 render conditions on references at 512^2, never on 1024^2 ones,
    // a pairing the model never sees (tried: it copied the picture and
    // ignored the instruction). `width`/`height` are not part of the edit
    // form, so `size` is the only place the area can come from.
    let ref_area = match params.size.as_deref() {
        None | Some("auto") => DEFAULT_SIDE * DEFAULT_SIDE,
        Some(s) => match parse_size(s) {
            Some((w, h)) => w as usize * h as usize,
            None => return bad(format!("size '{s}': expected WIDTHxHEIGHT or auto")),
        },
    };
    // decode + resize off the runtime: a 4K JPEG is real CPU work
    let images = params.images;
    let refs = match tokio::task::spawn_blocking(move || {
        images
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| {
                prepare_reference(&bytes, ref_area).map_err(|e| format!("image {}: {e}", i + 1))
            })
            .collect::<Result<Vec<_>, String>>()
    })
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(msg)) => return bad(msg),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                e.to_string(),
            );
        }
    };
    let last = refs
        .last()
        .map(|r| (r.width, r.height))
        .unwrap_or((DEFAULT_SIDE, DEFAULT_SIDE));
    let body = ImageGenerationRequest {
        prompt: params.prompt,
        model: params.model,
        background: params.background,
        moderation: None,
        n: params.n,
        output_compression: params.output_compression,
        output_format: params.output_format,
        partial_images: params.partial_images,
        quality: params.quality,
        response_format: params.response_format,
        size: params.size,
        style: None,
        user: params.user,
        stream: params.stream,
        seed: params.seed,
        steps: params.steps,
        guidance: params.guidance,
        negative_prompt: params.negative_prompt,
        width: None,
        height: None,
    };
    let p = match plan(&body, model.service.info().size_multiple, last) {
        Ok(p) => p,
        Err(msg) => return bad(msg),
    };
    // the editing template, its one `<|image_pad|>` per picture expanded to
    // the picture's merged-grid token count
    let grids: Vec<usize> = refs
        .iter()
        .map(|r| (r.width / 32) * (r.height / 32))
        .collect();
    let image_tokens: usize = grids.iter().sum();
    let tokenize = |text: &str| -> Result<(Vec<u32>, usize), String> {
        let (system, full) = ti2i_prompt(text, refs.len());
        let sys = model.tokenizer.encode(&system).map_err(|e| e.to_string())?;
        let raw = model.tokenizer.encode(&full).map_err(|e| e.to_string())?;
        if raw.len() < sys.len() || raw[..sys.len()] != sys[..] {
            return Err("prompt template tokenized inconsistently".into());
        }
        let mut ids = Vec::with_capacity(raw.len() + image_tokens);
        let mut slot = 0;
        for &id in &raw {
            if id == pad_id {
                let Some(&n) = grids.get(slot) else {
                    return Err("the template carries more image slots than pictures".into());
                };
                ids.extend(std::iter::repeat_n(pad_id, n));
                slot += 1;
            } else {
                ids.push(id);
            }
        }
        if slot != grids.len() {
            return Err(
                "the template did not tokenize its image slots as single control tokens".into(),
            );
        }
        if ids.len() > crate::serving::IMAGE_TEXT_CTX {
            return Err(format!(
                "the prompt with its {} pictures is {} tokens; this model reads up to {}",
                grids.len(),
                ids.len(),
                crate::serving::IMAGE_TEXT_CTX
            ));
        }
        Ok((ids, sys.len()))
    };
    let (prompt_ids, drop) = match tokenize(&p.prompt) {
        Ok(v) => v,
        Err(msg) => return bad(msg),
    };
    let negative = match p.negative.as_deref().map(tokenize) {
        Some(Ok(v)) => Some(v),
        Some(Err(msg)) => return bad(msg),
        None => None,
    };
    let request = ImageRequest {
        prompt_ids,
        drop,
        negative,
        width: p.width,
        height: p.height,
        steps: p.steps,
        seed: p.seed,
        guidance: p.guidance,
        n: p.n,
        partial_images: p.partials,
        references: refs,
        image_pad_id: pad_id,
    };
    if p.stream {
        return stream_generations(model, p, request, image_tokens as u64);
    }
    let reply = match model.service.generate(request).await {
        Ok(r) => r,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    };
    finish(p, reply, image_tokens as u64).await
}

/// The edit form, every part read: `image` / `image[]` (bytes, in order),
/// `mask`, and the text fields. Unknown fields are refused by name like the
/// JSON body's `deny_unknown_fields` does.
async fn read_edit_params(mut mp: Multipart) -> Result<ImageEditParams, String> {
    let mut p = ImageEditParams::default();
    let mut unknown: Vec<String> = Vec::new();
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return Err(e.to_string()),
        };
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "image" | "image[]" => {
                let bytes = field.bytes().await.map_err(|e| e.to_string())?;
                p.images.push(bytes.to_vec());
            }
            "mask" => {
                let bytes = field.bytes().await.map_err(|e| e.to_string())?;
                p.mask = Some(bytes.to_vec());
            }
            _ => {
                let text = field.text().await.map_err(|e| e.to_string())?;
                let num = |what: &str| -> Result<u32, String> {
                    text.trim()
                        .parse::<u32>()
                        .map_err(|_| format!("{what} '{text}' is not a whole number"))
                };
                match name.as_str() {
                    "prompt" => p.prompt = text,
                    "model" => p.model = Some(text),
                    "background" => {
                        p.background = Some(match text.trim() {
                            "transparent" => Background::Transparent,
                            "opaque" => Background::Opaque,
                            "auto" => Background::Auto,
                            other => {
                                return Err(format!(
                                    "background '{other}': expected transparent, opaque or auto"
                                ));
                            }
                        })
                    }
                    "input_fidelity" => p.input_fidelity = Some(text),
                    "n" => p.n = Some(num("n")?),
                    "output_compression" => p.output_compression = Some(num("output_compression")?),
                    "output_format" => {
                        p.output_format = Some(match text.trim() {
                            "png" => OutputFormat::Png,
                            "jpeg" => OutputFormat::Jpeg,
                            "webp" => OutputFormat::Webp,
                            other => {
                                return Err(format!(
                                    "output_format '{other}': expected png, jpeg or webp"
                                ));
                            }
                        })
                    }
                    "partial_images" => p.partial_images = Some(num("partial_images")?),
                    "quality" => p.quality = Some(text),
                    "response_format" => p.response_format = Some(text),
                    "size" => p.size = Some(text),
                    "user" => p.user = Some(text),
                    "stream" => {
                        p.stream = Some(match text.trim() {
                            "true" | "1" => true,
                            "false" | "0" => false,
                            other => {
                                return Err(format!("stream '{other}': expected true or false"));
                            }
                        })
                    }
                    "seed" => {
                        p.seed = Some(
                            text.trim()
                                .parse::<u64>()
                                .map_err(|_| format!("seed '{text}' is not a whole number"))?,
                        )
                    }
                    "steps" => p.steps = Some(num("steps")?),
                    "guidance" => {
                        p.guidance = Some(
                            text.trim()
                                .parse::<f32>()
                                .map_err(|_| format!("guidance '{text}' is not a number"))?,
                        )
                    }
                    "negative_prompt" => p.negative_prompt = Some(text),
                    other => unknown.push(other.to_owned()),
                }
            }
        }
    }
    if !unknown.is_empty() {
        return Err(format!("unknown field(s): {}", unknown.join(", ")));
    }
    Ok(p)
}

/// A reference picture as the engine takes it: decoded, resized to its own
/// aspect at `area` pixels - the output area - on the 32-pixel grid
/// (diffusers' `calculate_dimensions(output_resolution^2, w / h)`, Lanczos),
/// RGBA in [-1, 1].
fn prepare_reference(bytes: &[u8], area: usize) -> Result<ReferenceImage, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("could not be decoded ({e})"))?
        .to_rgba8();
    let (w0, h0) = img.dimensions();
    if w0 == 0 || h0 == 0 {
        return Err("is empty".into());
    }
    let (w, h) = fit_area(area as f64, w0 as f64 / h0 as f64, 32);
    let resized = if (w as u32, h as u32) == (w0, h0) {
        img
    } else {
        image::imageops::resize(
            &img,
            w as u32,
            h as u32,
            image::imageops::FilterType::Lanczos3,
        )
    };
    let rgba: Vec<f32> = resized
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    Ok(ReferenceImage {
        rgba,
        width: w,
        height: h,
    })
}

/// diffusers' `calculate_dimensions`: the size of `ratio` (w / h) with about
/// `area` pixels, each side rounded to the grid.
fn fit_area(area: f64, ratio: f64, grid: usize) -> (usize, usize) {
    let width = (area * ratio).sqrt();
    let height = width / ratio;
    let snap = |v: f64| ((v / grid as f64).round() as usize * grid).max(grid);
    (snap(width), snap(height))
}
