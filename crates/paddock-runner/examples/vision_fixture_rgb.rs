//! Image-codec-only helper for canonical GPU boundary diagnostics. It uses
//! the same image crate and EXIF orientation as the HTTP decoder. No resize,
//! normalization, tensor arithmetic, or CPU inference is performed here.
use image::ImageDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("expected encoded image and new RGB output path".into());
    }
    let mut decoder = image::ImageReader::open(&args[0])?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let rgb = image.to_rgb8();
    if rgb.width() == 0 || rgb.height() == 0 || rgb.width() > 4096 || rgb.height() > 4096 {
        return Err("boundary diagnostic accepts dimensions 1..4096".into());
    }
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    use std::io::Write;
    out.write_all(rgb.as_raw())?;
    println!("{} {}", rgb.width(), rgb.height());
    Ok(())
}
