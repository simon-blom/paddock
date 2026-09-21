//! POST /v1/segmentations - dense prediction: image chips in, rasters out.
//!
//! A Paddock surface (nothing in the OpenAI or Anthropic APIs is shaped like
//! this), built for the way the model is actually used: batch over an area. A
//! property is thousands of chips and a county is millions, so a request
//! carries many chips, the engine coalesces chips across requests into full
//! passes, and the wire offers a form with no per-chip framing at all.
//!
//! Two ways in, told apart by Content-Type:
//!
//!   - `multipart/form-data`: one or more `image` parts. Each is either a TIFF
//!     (sniffed by magic, not by filename) - the form a real request carries,
//!     a georeferenced 4-band chip - or a raw block of exactly
//!     `side * side * bands` bytes, u8 HWC in the checkpoint's band order.
//!     Text fields: `response_format`, `logits`.
//!   - anything else (`application/octet-stream`): the body is N raw blocks
//!     back to back, options in the query string. No boundaries to scan, no
//!     copies: this is the bulk form.
//!
//! Two ways out (`response_format`):
//!
//!   - `json` (default): the embeddings-shaped list, rasters as base64 of
//!     little-endian bytes, plus class names and - for chips that arrived as
//!     GeoTIFFs - where each output raster sits on the ground.
//!   - `binary`: `application/octet-stream`, per chip the class raster (u8)
//!     then the regression raster (f32 LE) then, if asked for, the logits
//!     (f16 LE, pixel-major); the geometry rides `x-paddock-*` headers. What a
//!     pipeline that already knows where its chips are wants.
//!
//! Georeferencing is carried, never invented: a GeoTIFF's tiepoint and pixel
//! scale come back scaled to the output raster; a raw block has no position
//! and gets none. Nothing is reprojected or resampled here - a chip at the
//! wrong ground sample distance is refused by name, because the model would
//! serve it without complaint and be quietly wrong about every hectare.

use std::io::Cursor;
use std::sync::Arc;

use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

use paddock_api::ErrorBody;
use paddock_api::segmentation::{
    Georeference, SegmentationData, SegmentationResponse, SegmentationUsage,
};
use paddock_engine::segment::SegRequest;
use paddock_models::dinov3::Dinov3SegConfig;

use crate::routes::AppState;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn bad(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Json,
    Binary,
}

#[derive(Default)]
struct Options {
    format: Option<String>,
    logits: Option<String>,
}

impl Options {
    fn resolve(self) -> Result<(Format, bool), String> {
        let format = match self.format.as_deref().map(str::trim) {
            None | Some("") | Some("json") => Format::Json,
            Some("binary") => Format::Binary,
            Some(other) => {
                return Err(format!(
                    "response_format '{other}': expected 'json' or 'binary'"
                ));
            }
        };
        let logits = match self.logits.as_deref().map(str::trim) {
            None | Some("") | Some("false") | Some("0") => false,
            Some("true") | Some("1") => true,
            Some(other) => return Err(format!("logits '{other}': expected true or false")),
        };
        Ok((format, logits))
    }
}

/// One decoded chip: its pixels, and where it is if it said.
struct Chip {
    pixels: Vec<u8>,
    geo: Option<Georeference>,
}

pub async fn handle(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let Some(model) = state.segmenter.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no segmentation model is loaded (start paddock with a dense-prediction checkpoint \
             as `model`)",
        );
    };
    let cfg = &model.segmenter.info().config;
    let chip_bytes = cfg.image_size * cfg.image_size * cfg.channels;

    let is_multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.trim_start()
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        });

    let mut opts = Options::default();
    // chip-major, the layout the engine takes - filled in place so a bulk
    // request's hundred-plus megabytes are copied once, not once per stage
    let mut pixels: Vec<u8> = Vec::new();
    let mut geos: Vec<Option<Georeference>> = Vec::new();

    if is_multipart {
        let mut mp = match Multipart::from_request(req, &state).await {
            Ok(mp) => mp,
            Err(e) => return bad(e.to_string()),
        };
        loop {
            let field = match mp.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => return bad(e.to_string()),
            };
            // SDKs spell a repeated part `image[]`
            match field.name().unwrap_or_default().trim_end_matches("[]") {
                "image" | "file" => {
                    let bytes = match field.bytes().await {
                        Ok(b) => b,
                        Err(e) => return bad(e.to_string()),
                    };
                    let idx = geos.len();
                    match decode_chip(&bytes, cfg) {
                        Ok(c) => {
                            pixels.extend_from_slice(&c.pixels);
                            geos.push(c.geo);
                        }
                        Err(m) => return bad(format!("image {idx}: {m}")),
                    }
                }
                "response_format" => opts.format = field.text().await.ok(),
                "logits" => opts.logits = field.text().await.ok(),
                // `model` is accepted-and-ignored like every single-model server
                _ => {}
            }
        }
    } else {
        for (k, v) in req
            .uri()
            .query()
            .unwrap_or_default()
            .split('&')
            .filter_map(|kv| kv.split_once('='))
        {
            match k {
                "response_format" => opts.format = Some(v.to_owned()),
                "logits" => opts.logits = Some(v.to_owned()),
                _ => {}
            }
        }
        let body = match axum::body::Bytes::from_request(req, &state).await {
            Ok(b) => b,
            Err(e) => return bad(e.to_string()),
        };
        if body.is_empty() || body.len() % chip_bytes != 0 {
            return bad(format!(
                "raw body is {} bytes, not a whole number of {chip_bytes}-byte chips \
                 ({side}x{side}x{bands} u8, row-major, bands interleaved as {names}). Send \
                 multipart/form-data with `image` parts to upload TIFFs.",
                body.len(),
                side = cfg.image_size,
                bands = cfg.channels,
                names = cfg.band_names.join(","),
            ));
        }
        geos = vec![None; body.len() / chip_bytes];
        pixels = body.to_vec();
    }

    let (format, logits) = match opts.resolve() {
        Ok(o) => o,
        Err(m) => return bad(m),
    };
    if geos.is_empty() {
        return bad("no chips in the request - send at least one `image` part");
    }

    let n = geos.len();
    let out = match model
        .segmenter
        .segment(SegRequest {
            pixels,
            chips: n,
            want_logits: logits,
        })
        .await
    {
        Ok(o) => o,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
    };

    let px = out.size * out.size;
    let height_bytes = |i: usize| -> Vec<u8> {
        out.height[i * px..(i + 1) * px]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    let logit_bytes = |i: usize| -> Option<Vec<u8>> {
        out.logits.as_ref().map(|l| {
            l[i * px * out.n_classes..(i + 1) * px * out.n_classes]
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect()
        })
    };

    match format {
        Format::Binary => {
            let per = px + px * 4 + if logits { px * out.n_classes * 2 } else { 0 };
            let mut body = Vec::with_capacity(n * per);
            for i in 0..n {
                body.extend_from_slice(&out.classes[i * px..(i + 1) * px]);
                body.extend_from_slice(&height_bytes(i));
                if let Some(l) = logit_bytes(i) {
                    body.extend_from_slice(&l);
                }
            }
            (
                [
                    (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
                    (
                        header::HeaderName::from_static("x-paddock-chips"),
                        n.to_string(),
                    ),
                    (
                        header::HeaderName::from_static("x-paddock-raster-size"),
                        out.size.to_string(),
                    ),
                    (
                        header::HeaderName::from_static("x-paddock-classes"),
                        out.n_classes.to_string(),
                    ),
                    (
                        header::HeaderName::from_static("x-paddock-layout"),
                        if logits {
                            "classes:u8,regression:f32le,logits:f16le"
                        } else {
                            "classes:u8,regression:f32le"
                        }
                        .to_owned(),
                    ),
                ],
                body,
            )
                .into_response()
        }
        Format::Json => {
            let data = (0..n)
                .map(|i| SegmentationData {
                    object: "segmentation".to_owned(),
                    index: i,
                    width: out.size,
                    height: out.size,
                    classes: B64.encode(&out.classes[i * px..(i + 1) * px]),
                    regression: B64.encode(height_bytes(i)),
                    logits: logit_bytes(i).map(|l| B64.encode(l)),
                    georeference: geos[i].clone(),
                })
                .collect();
            Json(SegmentationResponse {
                object: "list".to_owned(),
                data,
                model: model.id.clone(),
                class_names: cfg.class_names.clone(),
                regression_name: cfg.regression_name.clone(),
                pixel_size_m: cfg.out_pixel_size_m,
                usage: SegmentationUsage { chips: n },
            })
            .into_response()
        }
    }
}

/// A part's bytes -> a chip. TIFF by magic (`II*\0` / `MM\0*`, classic only -
/// a 1 MB chip has no use for BigTIFF), otherwise a raw block of exactly the
/// model's size.
fn decode_chip(bytes: &[u8], cfg: &Dinov3SegConfig) -> Result<Chip, String> {
    let chip_bytes = cfg.image_size * cfg.image_size * cfg.channels;
    let is_tiff = bytes.len() >= 4 && (bytes[..4] == *b"II*\0" || bytes[..4] == *b"MM\0*");
    if !is_tiff {
        if bytes.len() != chip_bytes {
            return Err(format!(
                "{} bytes is neither a TIFF nor a raw {side}x{side}x{bands} u8 block ({chip_bytes} bytes)",
                bytes.len(),
                side = cfg.image_size,
                bands = cfg.channels,
            ));
        }
        return Ok(Chip {
            pixels: bytes.to_vec(),
            geo: None,
        });
    }
    decode_tiff(bytes, cfg)
}

fn decode_tiff(bytes: &[u8], cfg: &Dinov3SegConfig) -> Result<Chip, String> {
    use tiff::ColorType;
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;

    let mut dec = Decoder::new(Cursor::new(bytes)).map_err(|e| format!("TIFF: {e}"))?;
    let (w, h) = dec.dimensions().map_err(|e| format!("TIFF: {e}"))?;
    let side = cfg.image_size as u32;
    if (w, h) != (side, side) {
        return Err(format!(
            "TIFF is {w}x{h}; the model takes {side}x{side} chips and nothing here resamples - \
             a resized chip would be at the wrong ground sample distance"
        ));
    }
    // 8-bit, band-interleaved, as many bands as the checkpoint has. A
    // four-sample RGB file reads as RGBA whatever its extra sample is tagged,
    // which is exactly the order wanted: bands 1..4 as written.
    let ct = dec.colortype().map_err(|e| format!("TIFF: {e}"))?;
    let bands = match ct {
        ColorType::Gray(8) => 1,
        ColorType::GrayA(8) => 2,
        ColorType::RGB(8) => 3,
        ColorType::RGBA(8) => 4,
        ColorType::Multiband {
            bit_depth: 8,
            num_samples,
        } => num_samples as usize,
        other => return Err(format!("TIFF is {other:?}; the model takes 8-bit samples")),
    };
    if bands != cfg.channels {
        return Err(format!(
            "TIFF has {bands} band(s); the model takes {} ({}). An RGB export without the \
             near-infrared band is not a smaller input, it is a different model's input",
            cfg.channels,
            cfg.band_names.join(", ")
        ));
    }
    let pixels = match dec.read_image().map_err(|e| format!("TIFF: {e}"))? {
        DecodingResult::U8(v) => v,
        _ => return Err("TIFF did not decode to 8-bit samples".into()),
    };
    if pixels.len() != cfg.image_size * cfg.image_size * cfg.channels {
        // planar (band-sequential) files decode band by band; refuse rather
        // than interleave a guess
        return Err("TIFF is not band-interleaved (PlanarConfiguration must be 1)".into());
    }

    // ---- georeferencing: tiepoint + pixel scale, the plain north-up form ----
    let scale = dec.get_tag_f64_vec(Tag::ModelPixelScaleTag).ok();
    let tie = dec.get_tag_f64_vec(Tag::ModelTiepointTag).ok();
    let geo = match (scale, tie) {
        (Some(s), Some(t)) if s.len() >= 2 && t.len() >= 6 => {
            let (sx, sy) = (s[0], s[1]);
            // a chip at another resolution is a different question than the
            // one the model was trained to answer
            let want = cfg.pixel_size_m;
            if (sx - want).abs() > want * 1e-6 || (sy - want).abs() > want * 1e-6 {
                return Err(format!(
                    "GeoTIFF pixel size is {sx} x {sy}; the model was trained at {want} m and \
                     nothing here resamples"
                ));
            }
            // tiepoint (i, j, k, x, y, z): raster (i, j) sits at map (x, y).
            // Move it to the top-left corner, which is also the output's.
            let origin_x = t[3] - t[0] * sx;
            let origin_y = t[4] + t[1] * sy;
            let ratio = cfg.image_size as f64 / cfg.out_size as f64;
            Some(Georeference {
                epsg: dec
                    .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
                    .ok()
                    .and_then(|k| epsg_of(&k)),
                origin_x,
                origin_y,
                pixel_size_x: sx * ratio,
                pixel_size_y: sy * ratio,
            })
        }
        _ => None,
    };
    Ok(Chip { pixels, geo })
}

/// EPSG code out of a GeoKeyDirectory: ProjectedCSTypeGeoKey (3072) if the
/// file is projected, else GeographicTypeGeoKey (2048). Entries are
/// (key, tag location, count, value); location 0 means the value is inline.
/// 32767 is "user-defined" - not a code.
fn epsg_of(keys: &[u16]) -> Option<u32> {
    let n = *keys.get(3)? as usize;
    let entry = |id: u16| {
        keys[4..]
            .as_chunks::<4>()
            .0
            .iter()
            .take(n)
            .find(|e| e[0] == id && e[1] == 0 && e[3] != 32767)
            .map(|e| e[3] as u32)
    };
    entry(3072).or_else(|| entry(2048))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsg_prefers_the_projected_key() {
        // version header, then GTModelType, ProjectedCSType, GeographicType
        let keys = [
            1u16, 1, 0, 3, 1024, 0, 1, 1, 2048, 0, 1, 4619, 3072, 0, 1, 3006,
        ];
        assert_eq!(epsg_of(&keys), Some(3006));
        // user-defined projection: fall back to nothing, not to 32767
        let user = [1u16, 1, 0, 1, 3072, 0, 1, 32767];
        assert_eq!(epsg_of(&user), None);
        assert_eq!(epsg_of(&[]), None);
    }

    #[test]
    fn options_refuse_what_they_do_not_know() {
        let ok = Options {
            format: Some("binary".into()),
            logits: Some("true".into()),
        };
        assert!(matches!(ok.resolve(), Ok((Format::Binary, true))));
        assert!(Options::default().resolve().is_ok());
        assert!(
            Options {
                format: Some("geotiff".into()),
                logits: None
            }
            .resolve()
            .is_err()
        );
        assert!(
            Options {
                format: None,
                logits: Some("yes".into())
            }
            .resolve()
            .is_err()
        );
    }
}
