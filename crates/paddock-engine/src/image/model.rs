//! Backend-independent Qwen-Image request geometry and output.

/// The three-axis rope split of the 2.1 DiT (frame, height, width), summing
/// to the head dim. Not stamped in the GGUF; the config.json constant.
pub const AXES_DIMS_ROPE: [usize; 3] = [16, 56, 56];
/// Rope base.
pub const ROPE_THETA: f32 = 10000.0;
/// Latent channels of the VAE / DiT.
pub const LATENT_CHANNELS: usize = 64;
/// Pixels per latent token per side.
pub const VAE_SCALE: usize = 16;
/// Output sides must be multiples of this (`vae_scale_factor * 2`).
pub const SIZE_MULTIPLE: usize = 32;

/// One generation request, already tokenized: the caller owns the tokenizer.
pub struct GenerateRequest<'a> {
    /// The full templated prompt's ids.
    pub prompt_ids: &'a [u32],
    /// How many leading ids are the system block (dropped from the hidden
    /// states) - the tokenized system block's length.
    pub drop: usize,
    /// The negative prompt, same shape, when guidance is on.
    pub negative: Option<(&'a [u32], usize)>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    /// Which draw on `seed` this image is: image `i` of an `n`-image request
    /// draws at offset `i` (stable-diffusion.cpp's batch convention; a
    /// single image is offset 0).
    pub noise_offset: u32,
    /// `true_cfg_scale`; guidance is off at <= 1 (the model's own default).
    pub guidance: f32,
    /// Reference pictures for editing, in the order their `<|image_pad|>`
    /// runs appear in `prompt_ids` (and in `negative`, which must carry the
    /// same slots). Empty for text-to-image.
    pub references: &'a [Reference<'a>],
    /// The tokenizer's `<|image_pad|>` id - how the runs are found.
    pub image_pad_id: u32,
}

/// A reference picture as the editing lane reads it: RGBA in [-1, 1], NHWC
/// f32, sides multiples of 32 (the caller resizes by the pipeline's rule:
/// its own aspect at the output area). The VAE encodes all four channels;
/// the vision tower sees it composited over white.
pub struct Reference<'a> {
    pub rgba: &'a [f32],
    pub width: usize,
    pub height: usize,
}

impl Reference<'_> {
    /// The tower's merged-grid token count: `(h / 32) * (w / 32)`, what its
    /// `<|image_pad|>` run in the prompt must be expanded to.
    pub fn vision_tokens(&self) -> usize {
        (self.width / (2 * 16)) * (self.height / (2 * 16))
    }
}

/// A decoded image: interleaved 8-bit RGBA, row-major.
pub struct Rgba {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}
