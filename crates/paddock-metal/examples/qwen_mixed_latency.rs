//! Real-graph mixed-shape diagnostic, not a serving or reference parity gate.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires an M5 Mac");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::generator::Generator;
    let path = std::env::args().nth(1).ok_or("qwen_mixed_latency MODEL")?;
    let mut model = paddock_metal::Qwen35::load(std::path::Path::new(&path), 4096, 4, None)?;
    for slot in 0..3 {
        model.forward_prefill(
            slot,
            &(0..128)
                .map(|i| 100 + ((i + slot * 7) % 100) as u32)
                .collect::<Vec<_>>(),
        )?;
    }
    let mut position = 128;
    for rows in [32, 48, 64, 96, 128] {
        for repeat in 0..4 {
            model.prefill_begin(
                3,
                (0..1024)
                    .map(|i| 1000 + ((i + rows + repeat * 200) % 1000) as u32)
                    .collect(),
            )?;
            let started = std::time::Instant::now();
            model.forward_mixed(
                &[(0, 13, position), (1, 13, position), (2, 13, position)],
                rows - 3,
            )?;
            println!(
                "{}",
                serde_json::json!({"rows":rows,"repeat":repeat,
                "gpu_ms":model.last_gpu_seconds*1000.0,"wall_ms":started.elapsed().as_secs_f64()*1000.0})
            );
            model.prefill_abort(3);
            position += 1;
        }
    }
    Ok(())
}
