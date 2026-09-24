//! Image generation wire types - `POST /v1/images/generations` and
//! `POST /v1/images/edits`, the OpenAI Images API in its GPT-image-model form
//! (openai-python 2.53: images are always base64, sizes are arbitrary
//! `WIDTHxHEIGHT` strings, `output_format` / `background` / `quality` /
//! `output_compression` select the encoding, and `stream` returns
//! `image_generation.partial_image` / `image_generation.completed` events).
//! The DALL-E-only fields (`style`, `response_format: url`) are accepted on
//! the wire and refused by name in the handler, the way every other
//! unimplemented spec parameter is.
//!
//! Paddock extras ride the same object, absent from every OpenAI client and
//! harmless to them: `seed`, `steps`, `guidance`, `negative_prompt`, and
//! `width` / `height` as an alternative to `size`.
//!
//! Edits arrive as multipart (`image[]` parts + fields), so their request has
//! no serde form; [`ImageEditParams`] is what the handler assembles from the
//! parts.

use serde::{Deserialize, Serialize};

/// `background`: the alpha the caller wants. `auto` leaves it to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Background {
    Transparent,
    Opaque,
    Auto,
}

impl Background {
    pub fn as_str(self) -> &'static str {
        match self {
            Background::Transparent => "transparent",
            Background::Opaque => "opaque",
            Background::Auto => "auto",
        }
    }
}

/// `output_format`: the container the base64 carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Png,
    Jpeg,
    Webp,
}

impl OutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpeg",
            OutputFormat::Webp => "webp",
        }
    }

    /// PNG and WebP keep an alpha channel; JPEG has none.
    pub fn keeps_alpha(self) -> bool {
        !matches!(self, OutputFormat::Jpeg)
    }
}

/// `POST /v1/images/generations` body.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub background: Option<Background>,
    /// `low` | `auto` - accepted, nothing here to moderate
    #[serde(default)]
    pub moderation: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    /// 0-100, webp / jpeg only
    #[serde(default)]
    pub output_compression: Option<u32>,
    #[serde(default)]
    pub output_format: Option<OutputFormat>,
    /// 0-3 partial images while streaming
    #[serde(default)]
    pub partial_images: Option<u32>,
    /// `auto` | `low` | `medium` | `high` | `xhigh` | `max` (`standard` /
    /// `hd` are DALL-E's) - maps to the step count here
    #[serde(default)]
    pub quality: Option<String>,
    /// `b64_json` is the only form; `url` is refused
    #[serde(default)]
    pub response_format: Option<String>,
    /// `WIDTHxHEIGHT` or `auto`
    #[serde(default)]
    pub size: Option<String>,
    /// DALL-E 3 only - refused
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    // ---- Paddock extras ----
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub steps: Option<u32>,
    /// classifier-free guidance scale (`true_cfg_scale`); off at <= 1
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

/// `POST /v1/images/edits`, assembled from the multipart parts.
#[derive(Debug, Clone, Default)]
pub struct ImageEditParams {
    /// the `image` / `image[]` parts, in order
    pub images: Vec<Vec<u8>>,
    pub mask: Option<Vec<u8>>,
    pub prompt: String,
    pub model: Option<String>,
    pub background: Option<Background>,
    pub input_fidelity: Option<String>,
    pub n: Option<u32>,
    pub output_compression: Option<u32>,
    pub output_format: Option<OutputFormat>,
    pub partial_images: Option<u32>,
    pub quality: Option<String>,
    pub response_format: Option<String>,
    pub size: Option<String>,
    pub user: Option<String>,
    pub stream: Option<bool>,
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub negative_prompt: Option<String>,
}

/// Parse a `WIDTHxHEIGHT` size string.
pub fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// One generated image. GPT image models return base64 only; `url` and
/// `revised_prompt` exist for DALL-E callers reading the same object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageTokenDetails {
    pub image_tokens: u64,
    pub text_tokens: u64,
}

/// Token accounting in the images shape: input = prompt text (+ reference
/// image) tokens, output = the target's latent tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUsage {
    pub input_tokens: u64,
    pub input_tokens_details: ImageTokenDetails,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

/// The `/v1/images/*` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesResponse {
    /// unix seconds
    pub created: u64,
    pub data: Vec<ImageData>,
    /// `transparent` | `opaque` - what was produced, never `auto`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// `WIDTHxHEIGHT` as generated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageUsage>,
}

/// A streaming event: `image_generation.partial_image` /
/// `image_generation.completed`, and the `image_edit.*` twins. One struct,
/// `type` selects; `partial_image_index` only on partials, `usage` only on
/// completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageStreamEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub b64_json: String,
    pub background: String,
    pub created_at: u64,
    pub output_format: OutputFormat,
    pub quality: String,
    pub size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_image_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_request_reads_the_sdk_shape() {
        let r: ImageGenerationRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"qwen-image-2.1","size":"1536x864","output_format":"webp",
                "background":"transparent","n":2,"seed":42,"steps":20}"#,
        )
        .expect("parses");
        assert_eq!(parse_size(r.size.as_deref().unwrap()), Some((1536, 864)));
        assert_eq!(r.output_format, Some(OutputFormat::Webp));
        assert_eq!(r.background, Some(Background::Transparent));
        assert_eq!((r.n, r.seed, r.steps), (Some(2), Some(42), Some(20)));
    }

    #[test]
    fn unknown_fields_are_refused() {
        let e =
            serde_json::from_str::<ImageGenerationRequest>(r#"{"prompt":"x","sizes":"1024x1024"}"#)
                .expect_err("unknown field");
        assert!(e.to_string().contains("sizes"));
    }

    #[test]
    fn response_omits_absent_optionals() {
        let r = ImagesResponse {
            created: 1,
            data: vec![ImageData {
                b64_json: Some("AA==".into()),
                revised_prompt: None,
                url: None,
            }],
            background: Some("opaque".into()),
            output_format: Some(OutputFormat::Png),
            quality: None,
            size: Some("1024x1024".into()),
            usage: None,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["data"][0]["b64_json"], "AA==");
        assert!(v["data"][0].get("url").is_none());
        assert!(v.get("usage").is_none());
        assert_eq!(v["output_format"], "png");
    }
}
