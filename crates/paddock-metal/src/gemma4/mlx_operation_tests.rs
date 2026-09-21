use super::*;
use paddock_models::safetensors::SafetensorsFile;

#[test]
#[ignore = "requires independent MLX GPU operation fixtures"]
fn gemma_muse_mlx_operation_boundaries() {
    let path = std::env::var("PADDOCK_MM_MLX_OPERATIONS").unwrap();
    let root = Path::new(&path);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let device = MetalDevice::new(None).unwrap();
    let source = paddock_models::safetensors::ShardedSafetensors::open_dir(Path::new(
        manifest["model"].as_str().unwrap(),
    ))
    .unwrap();
    let mut failures = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let fixture = SafetensorsFile::open(&root.join(case["file"].as_str().unwrap())).unwrap();
        let upload = |name| device.upload(fixture.bytes(name).unwrap().1).unwrap();
        let x = upload("x");
        let out = device.alloc(fixture.bytes("y").unwrap().1.len()).unwrap();
        let op = case["op"].as_str().unwrap();
        let n = case["width"].as_u64().unwrap() as usize;
        let rows = case["rows"].as_u64().unwrap() as usize;
        let cmd = device.begin().unwrap();
        let f = |name: &str| (case[name].as_f64().unwrap() as f32).to_bits();
        let result = match op {
            "sandwich" => {
                let delta = upload("delta");
                let w = upload("w");
                let next = upload("next");
                cmd.dispatch(
                    "gmlx_sandwich",
                    &[&x, &delta, &w, &next, &out],
                    &[
                        n as u32,
                        case["mode"].as_u64().unwrap() as u32,
                        f("eps"),
                        f("post_eps"),
                        1f32.to_bits(),
                    ],
                    [rows, 1, 1],
                    (n.div_ceil(128) * 32).min(1024),
                );
                &out
            }
            "dense" => {
                let nout = case["n"].as_u64().unwrap() as usize;
                let (_, bytes) = source.bytes(case["tensor"].as_str().unwrap()).unwrap();
                let weight = device.upload(bytes).unwrap();
                let bias = upload("bias");
                cmd.dispatch(
                    "gmlx_bmm",
                    &[&weight, &x, &out, &bias],
                    &[n as u32, nout as u32, rows as u32, 1],
                    [nout.div_ceil(32), rows.div_ceil(32), 1],
                    128,
                );
                &out
            }
            "attention" => {
                let hd = case["hd"].as_u64().unwrap() as usize;
                let kh = case["kh"].as_u64().unwrap() as usize;
                let k = upload("k_bf16");
                let v = upload("v_bf16");
                let meta = device
                    .upload(
                        &(0..rows)
                            .flat_map(|r| [0u32, r as u32])
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let pages = device
                    .upload(
                        &(0..rows.div_ceil(16) as u32)
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let tiles = device
                    .upload(
                        &(0..rows)
                            .step_by(32)
                            .flat_map(|r| [r as u32, (rows - r).min(32) as u32])
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                cmd.dispatch(
                    &format!("gmlx_prefill{hd}"),
                    &[&x, &k, &v, &meta, &pages, &out, &tiles],
                    &[32, kh as u32, rows.div_ceil(16) as u32, 0, 0],
                    [32, rows.div_ceil(32), 1],
                    128,
                );
                cmd.dispatch(
                    "gmlx_round",
                    &[&out],
                    &[(n * rows) as u32],
                    [(n * rows).div_ceil(256), 1, 1],
                    256,
                );
                &out
            }
            "projection" => {
                let nout = case["n"].as_u64().unwrap() as usize;
                let w = crate::affine::load(
                    &device,
                    &source,
                    case["tensor"].as_str().unwrap(),
                    n,
                    nout,
                )
                .unwrap();
                let workspace = device
                    .alloc(crate::affine::workspace_bytes(n, nout, rows))
                    .unwrap();
                crate::affine::project(&cmd, &[(&w, &out)], &x, rows, &workspace);
                &out
            }
            "norm" => {
                let w = upload("w");
                cmd.dispatch(
                    "gmlx_norm",
                    &[&x, &w, &out],
                    &[n as u32, case["mode"].as_u64().unwrap() as u32, f("eps")],
                    [rows, 1, 1],
                    (n.div_ceil(128) * 32).min(1024),
                );
                &out
            }
            "layernorm" => {
                let w = upload("w");
                let bias = upload("bias");
                cmd.dispatch(
                    "gmlx_layer_norm",
                    &[&x, &w, &bias, &out],
                    &[n as u32, f("eps")],
                    [rows, 1, 1],
                    (n.div_ceil(128) * 32).min(1024),
                );
                &out
            }
            "geglu" | "swiglu" => {
                let up = upload("up");
                cmd.dispatch(
                    if op == "geglu" {
                        "gmlx_geglu"
                    } else {
                        "mlx_swiglu"
                    },
                    &[&x, &up],
                    &[(n * rows) as u32],
                    [(n * rows).div_ceil(256), 1, 1],
                    256,
                );
                &x
            }
            "softcap" => {
                cmd.dispatch(
                    "gmlx_softcap",
                    &[&x],
                    &[(n * rows) as u32, f("cap"), f("scale")],
                    [(n * rows).div_ceil(256), 1, 1],
                    256,
                );
                &x
            }
            "erfgelu" => {
                cmd.dispatch(
                    "gmlx_erfgelu",
                    &[&x],
                    &[(n * rows) as u32],
                    [(n * rows).div_ceil(256), 1, 1],
                    256,
                );
                &x
            }
            "qnorm" => {
                let w = upload("w");
                let meta = device
                    .upload(
                        &(0..rows)
                            .flat_map(|r| [0u32, r as u32 + 9])
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let muse = case["muse"].as_bool().unwrap();
                let global = case["global_layer"].as_bool().unwrap();
                cmd.dispatch(
                    if muse { "mmlx_qnorm" } else { "gmlx_qnorm" },
                    &[&x, &w, &meta, &w],
                    &[
                        32,
                        case["hd"].as_u64().unwrap() as u32,
                        0,
                        if muse { 1e-5f32 } else { 1e-6f32 }.to_bits(),
                        if muse {
                            500000f32
                        } else if global {
                            1000000f32
                        } else {
                            10000f32
                        }
                        .to_bits(),
                        global as u32,
                    ],
                    [32, rows, 1],
                    32,
                );
                &x
            }
            _ => panic!("unknown operation {op}"),
        };
        cmd.finish().unwrap();
        let (_, bytes) = fixture.bytes("y").unwrap();
        // SAFETY: the GPU command completed; only compare its result with a GPU oracle.
        let actual = unsafe { result.read_f32(0, bytes.len() / 4) };
        let mut unequal = 0;
        let mut error = 0f32;
        for (i, (a, b)) in actual.iter().zip(bytes.chunks_exact(4)).enumerate() {
            let b = f32::from_le_bytes(b.try_into().unwrap());
            assert!(a.is_finite() && b.is_finite());
            if *a != b {
                unequal += 1;
                if unequal <= 3 {
                    eprintln!("{op}[{i}] {a} vs {b}");
                }
            }
            error = error.max((a - b).abs());
        }
        eprintln!(
            "{op} m={rows} unequal={unequal}/{} max={error}",
            actual.len()
        );
        if op == "attention" {
            let (_, bytes) = fixture.bytes("y_f32").unwrap();
            let max = actual
                .iter()
                .zip(bytes.chunks_exact(4))
                .map(|(a, b)| (a - f32::from_le_bytes(b.try_into().unwrap())).abs())
                .fold(0f32, f32::max);
            eprintln!("attention hd={} F32-reference max={max}", case["hd"]);
        }
        if unequal > 0 {
            failures.push(format!("{op} m={rows}: {unequal}, {error}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
