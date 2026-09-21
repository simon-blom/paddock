//! Dense-prediction wire types - `POST /v1/segmentations`.
//!
//! A Paddock surface: no OpenAI or Anthropic API describes "image in, raster
//! out", so the envelope borrows the one shape every client here already
//! reads - the embeddings list (`object: "list"`, `data[]` with an `index`,
//! `model`, `usage`) - and the rasters ride it the way OpenAI ships compact
//! embeddings: base64 of the little-endian bytes. A 256x256 class raster is
//! 64 KB; as a JSON number array it would be half a megabyte of commas.
//!
//! The request has no JSON form at all. Chips are megabyte binary blocks, so
//! they arrive as multipart file parts (one or many `image` parts, each a
//! GeoTIFF or a raw HWC block) or as one raw `application/octet-stream` body
//! of concatenated blocks - see the handler in paddock-runner.

use serde::{Deserialize, Serialize};

/// Where an output raster sits on the ground - the input chip's own
/// georeferencing carried across the model, at the output's pixel size. Present
/// only when the chip arrived as a GeoTIFF that said where it was; a raw block
/// carries no position and gets none invented for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Georeference {
    /// EPSG code of the projected (or geographic) CRS, when the file names one
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epsg: Option<u32>,
    /// map X of the raster's top-left corner
    pub origin_x: f64,
    /// map Y of the raster's top-left corner
    pub origin_y: f64,
    /// map units per output pixel, east and south (both positive)
    pub pixel_size_x: f64,
    pub pixel_size_y: f64,
}

/// One chip's rasters. Every raster is `height` rows of `width` pixels,
/// row-major from the top-left, base64 of little-endian bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationData {
    pub object: String, // "segmentation"
    pub index: usize,
    pub width: usize,
    pub height: usize,
    /// u8 per pixel: the argmax class code, an index into `class_names`
    pub classes: String,
    /// f32 per pixel: the regression raster (metres for a canopy-height head).
    /// Unclamped - a negative value is the model being wrong, not a sentinel.
    pub regression: String,
    /// f16 per pixel per class, pixel-major (`[row][col][class]`): the class
    /// logits. Only when the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub georeference: Option<Georeference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationUsage {
    /// chips served - the unit this model is metered in
    pub chips: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationResponse {
    pub object: String, // "list"
    pub data: Vec<SegmentationData>,
    pub model: String,
    /// class code -> name, in code order
    pub class_names: Vec<String>,
    /// what the regression raster measures, e.g. "canopy_height_m"
    pub regression_name: String,
    /// ground metres per output pixel, as the checkpoint declares it
    pub pixel_size_m: f64,
    pub usage: SegmentationUsage,
}
