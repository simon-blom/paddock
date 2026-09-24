//! Image-generation service seam - the `segment.rs` shape: a dedicated GPU
//! thread owning the model, oneshot request/response, requests served FIFO.
//!
//! An image request is minutes of GPU work with nothing to interleave at
//! the token level, so the scheduler is a queue: one request at a time, its
//! `n` images in turn. What a request shares across its images (the prompt's
//! prefix K/V) and what neighbouring requests could share (the same prompt,
//! the negative prompt) is the cross-request prefix cache the design note
//! lists for the editing phase; today every image re-encodes its prompt,
//! which is a few hundred milliseconds against a step budget of many
//! seconds.
//!
//! The reply carries the images already decoded to 8-bit RGBA; encoding
//! them into a PNG / WebP / JPEG is host work the runner does off this
//! thread. A streaming request additionally gets the render's progressive
//! previews (`ImageEvent::Partial`) on the way - the x0 estimate decoded at
//! evenly spaced steps, see `QwenImage::generate_with`.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use tokio::sync::oneshot;

pub mod model;
pub mod schedule;
use model::{GenerateRequest, Reference, Rgba, SIZE_MULTIPLE, VAE_SCALE};

/// A worker-owned image pipeline. Construction happens on the engine thread;
/// no GPU resource or API credential crosses the serving seam.
pub trait ImageBackend {
    fn weights_bytes(&self) -> u64;
    fn can_edit(&self) -> bool;
    fn generate(
        &mut self,
        request: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> Result<(), String>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Rgba, String>;
}

#[cfg(feature = "cuda")]
impl ImageBackend for crate::gpu_model::qwen_image::QwenImage {
    fn weights_bytes(&self) -> u64 {
        self.weights_bytes()
    }
    fn can_edit(&self) -> bool {
        self.can_edit()
    }
    fn generate(
        &mut self,
        request: &GenerateRequest<'_>,
        partials: usize,
        on_partial: &mut dyn FnMut(usize, Rgba) -> Result<(), String>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Rgba, String> {
        self.generate_controlled(
            request,
            partials,
            &mut |index, image| {
                on_partial(index, image)
                    .map_err(crate::gpu_model::gpt_oss::GpuModelError::Unsupported)
            },
            cancelled,
        )
        .map_err(|e| e.to_string())
    }
}

/// One request, already tokenized (the runner owns the tokenizer).
pub struct ImageRequest {
    pub prompt_ids: Vec<u32>,
    /// leading ids that are the system block, dropped from the hidden states
    pub drop: usize,
    /// the negative prompt, same shape, when guidance is on
    pub negative: Option<(Vec<u32>, usize)>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    /// images to make; image `i` draws its noise at offset `i` on the same
    /// seed (stable-diffusion.cpp's `-b` convention)
    pub n: usize,
    /// progressive previews of the render, delivered through
    /// [`ImageService::generate_stream`] (0 = none; ignored on the oneshot
    /// path, and only the first of `n` images is previewed)
    pub partial_images: usize,
    /// reference pictures for editing, in the order of their `<|image_pad|>`
    /// runs in `prompt_ids` (already expanded to each picture's grid)
    pub references: Vec<ReferenceImage>,
    /// the tokenizer's `<|image_pad|>` id
    pub image_pad_id: u32,
}

/// A reference picture, resized by the caller to the pipeline's rule: RGBA
/// in [-1, 1], NHWC f32, sides multiples of 32.
pub struct ReferenceImage {
    pub rgba: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// What a streaming request delivers, in order: the previews as they are
/// decoded, then the reply (or the error) exactly once.
pub enum ImageEvent {
    Partial { index: usize, image: Rgba },
    Done(ImageReply),
    Error(String),
}

pub struct ImageReply {
    pub images: Vec<Rgba>,
    /// prompt tokens the encoder saw after the system block
    pub prompt_tokens: usize,
    /// latent tokens per image (the output-token count in the images API)
    pub latent_tokens: usize,
    pub elapsed_s: f32,
}

/// What the runner needs to describe the endpoint without touching the GPU.
pub struct ImageInfo {
    pub weights_bytes: u64,
    /// output sides must be multiples of this
    pub size_multiple: usize,
    /// pixels per latent token per side
    pub vae_scale: usize,
    /// reference pictures are taken (the vision tower is wired)
    pub edit: bool,
}

/// Where a request's answer goes: one reply at the end, or a stream that
/// carries the previews first. Both know when the caller is gone.
enum Reply {
    Once(oneshot::Sender<Result<ImageReply, String>>),
    Stream(tokio::sync::mpsc::UnboundedSender<ImageEvent>),
}

impl Reply {
    fn is_closed(&self) -> bool {
        match self {
            Reply::Once(tx) => tx.is_closed(),
            Reply::Stream(tx) => tx.is_closed(),
        }
    }
}

type Incoming = (ImageRequest, Reply);

/// Handle to the image thread. Cloneable; requests are served FIFO.
#[derive(Clone)]
pub struct ImageService {
    tx: Sender<Incoming>,
    info: Arc<ImageInfo>,
}

impl ImageService {
    /// Spawn the image thread. `build` constructs the model on that thread
    /// (the CUDA context binds to it) and may fail; spawn blocks until it
    /// has and propagates the error - the `Segmenter::spawn` contract.
    pub fn spawn<F, M: ImageBackend + 'static>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<M, String> + Send + 'static,
    {
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel::<Result<ImageInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-image".into())
            .spawn(move || {
                let model = match build() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(ImageInfo {
                            weights_bytes: m.weights_bytes(),
                            size_multiple: SIZE_MULTIPLE,
                            vae_scale: VAE_SCALE,
                            edit: m.can_edit(),
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                serve(model, rx);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Generate a request's images; resolves when the last one is decoded.
    pub async fn generate(&self, req: ImageRequest) -> Result<ImageReply, String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send((req, Reply::Once(tx)))
            .map_err(|_| "image thread gone".to_string())?;
        rx.await
            .map_err(|_| "image thread dropped the request".to_string())?
    }

    /// The streaming form: `req.partial_images` previews of the first image
    /// arrive as they are decoded, then the reply. Dropping the receiver
    /// stops the render at its next cancellation checkpoint (at least once
    /// per denoising step; native Metal also checks transformer blocks).
    pub fn generate_stream(
        &self,
        req: ImageRequest,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<ImageEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.tx
            .send((req, Reply::Stream(tx)))
            .map_err(|_| "image thread gone".to_string())?;
        Ok(rx)
    }
}

fn serve(mut model: impl ImageBackend, rx: Receiver<Incoming>) {
    while let Ok((req, reply)) = rx.recv() {
        // a caller that went away takes its request with it - a closed tab
        // must not keep the GPU for a minute
        if reply.is_closed() {
            continue;
        }
        let t0 = Instant::now();
        let mut images = Vec::with_capacity(req.n);
        let mut err = None;
        let references: Vec<Reference<'_>> = req
            .references
            .iter()
            .map(|r| Reference {
                rgba: &r.rgba,
                width: r.width,
                height: r.height,
            })
            .collect();
        for i in 0..req.n {
            // previews ride the stream for the first image only; a preview
            // nobody is listening to stops the render, the same way a
            // dropped oneshot stops the loop below
            let (partials, stream) = match &reply {
                Reply::Stream(tx) if i == 0 => (req.partial_images, Some(tx)),
                _ => (0, None),
            };
            let mut on_partial = |index: usize, image: Rgba| {
                let Some(tx) = stream else { return Ok(()) };
                tx.send(ImageEvent::Partial { index, image })
                    .map_err(|_| "client went away".to_string())
            };
            let r = model.generate(
                &GenerateRequest {
                    prompt_ids: &req.prompt_ids,
                    drop: req.drop,
                    negative: req.negative.as_ref().map(|(ids, d)| (ids.as_slice(), *d)),
                    width: req.width,
                    height: req.height,
                    steps: req.steps,
                    seed: req.seed,
                    noise_offset: i as u32,
                    guidance: req.guidance,
                    references: &references,
                    image_pad_id: req.image_pad_id,
                },
                partials,
                &mut on_partial,
                &|| reply.is_closed(),
            );
            match r {
                Ok(img) => images.push(img),
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            }
            if reply.is_closed() {
                break;
            }
        }
        let out = match err {
            Some(e) => Err(e),
            None => {
                let scale = VAE_SCALE;
                Ok(ImageReply {
                    latent_tokens: (req.width / scale) * (req.height / scale),
                    prompt_tokens: req.prompt_ids.len().saturating_sub(req.drop),
                    images,
                    elapsed_s: t0.elapsed().as_secs_f32(),
                })
            }
        };
        match reply {
            Reply::Once(tx) => {
                let _ = tx.send(out);
            }
            Reply::Stream(tx) => {
                let _ = tx.send(match out {
                    Ok(r) => ImageEvent::Done(r),
                    Err(e) => ImageEvent::Error(e),
                });
            }
        }
    }
}
