//! Bootstrap tests: config validation, framing, and the two-role handshake.
//! Host-only - no CUDA, no engine. The two-process path is exercised by the
//! coordinator/worker handshake tests here over localhost.

use paddock_dist::config::{DEFAULT_MASTER_PORT, ParallelConfig, ParallelConfigError, RankRole};
use paddock_dist::protocol::{ControlMessage, ProtocolError, receive_nccl_id, send_nccl_id};
use paddock_dist::worker::{BootstrapError, coordinate, shutdown_worker, work};
use std::io::Write as _;
use std::time::Duration;

fn resolved(rank: usize, port: u16) -> paddock_dist::config::Resolved {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(rank),
        master_addr: Some("127.0.0.1".into()),
        master_port: Some(port),
    };
    cfg.resolved(false).expect("valid").expect("tp2")
}

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .expect("bind probe")
        .local_addr()
        .expect("addr")
        .port()
}

// --- config validation ------------------------------------------------

#[test]
fn unset_parallel_is_the_historical_path() {
    let cfg = ParallelConfig::default();
    assert!(cfg.resolved(true).unwrap().is_none());
}

#[test]
fn tp1_collapses_to_the_historical_path() {
    let cfg = ParallelConfig {
        tp_size: Some(1),
        rank: None,
        master_addr: None,
        master_port: None,
    };
    assert!(cfg.resolved(true).unwrap().is_none());
    let cfg = ParallelConfig {
        tp_size: Some(1),
        rank: Some(0),
        master_addr: None,
        master_port: None,
    };
    assert!(cfg.resolved(true).unwrap().is_none());
}

#[test]
fn rank_without_tp_size_is_an_error() {
    let cfg = ParallelConfig {
        tp_size: None,
        rank: Some(1),
        master_addr: None,
        master_port: None,
    };
    assert!(matches!(
        cfg.resolved(true),
        Err(ParallelConfigError::RankOutOfRange { tp: 1, rank: 1 })
    ));
}

#[test]
fn tp3_is_rejected_not_downgraded() {
    let cfg = ParallelConfig {
        tp_size: Some(3),
        rank: Some(0),
        master_addr: None,
        master_port: None,
    };
    assert!(matches!(
        cfg.resolved(true),
        Err(ParallelConfigError::UnsupportedTpSize(3))
    ));
}

#[test]
fn rank_out_of_range_is_rejected() {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(2),
        master_addr: None,
        master_port: None,
    };
    assert!(matches!(
        cfg.resolved(true),
        Err(ParallelConfigError::RankOutOfRange { tp: 2, rank: 2 })
    ));
}

#[test]
fn rank1_may_not_serve_but_a_child_may_exist() {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(1),
        master_addr: Some("10.0.0.5".into()),
        master_port: None,
    };
    // A user-facing serving start is refused.
    assert!(matches!(
        cfg.resolved(true),
        Err(ParallelConfigError::WorkerMustNotServe)
    ));
    // The coordinator-spawned child (not a serving start) is accepted.
    let r = cfg.resolved(false).unwrap().unwrap();
    assert_eq!(r.role, RankRole::Worker);
    assert_eq!(r.master_port, DEFAULT_MASTER_PORT);
}

#[test]
fn rank1_without_master_addr_is_rejected() {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(1),
        master_addr: None,
        master_port: None,
    };
    assert!(matches!(
        cfg.resolved(false),
        Err(ParallelConfigError::EmptyMasterAddr)
    ));
}

#[test]
fn rank0_defaults_bind_wildcard_and_default_port() {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(0),
        master_addr: None,
        master_port: None,
    };
    let r = cfg.resolved(true).unwrap().unwrap();
    assert_eq!(r.master_addr, "0.0.0.0");
    assert_eq!(r.master_port, DEFAULT_MASTER_PORT);
    assert!(r.is_coordinator());
}

#[test]
fn worker_env_marks_the_child_and_sets_rank1() {
    let cfg = ParallelConfig {
        tp_size: Some(2),
        rank: Some(0),
        master_addr: Some("192.168.100.10".into()),
        master_port: Some(12345),
    };
    let resolved = cfg.resolved(true).unwrap().unwrap();
    let env: Vec<(String, String)> = resolved
        .worker_env()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let get = |k: &str| {
        env.iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    assert_eq!(get("PADDOCK_TP_SIZE"), "2");
    assert_eq!(get("PADDOCK_TP_RANK"), "1");
    assert_eq!(get("PADDOCK_TP_MASTER_ADDR"), "192.168.100.10");
    assert_eq!(get("PADDOCK_TP_MASTER_PORT"), "12345");
    assert_eq!(get("PADDOCK_TP_WORKER_CHILD"), "1");
}

#[test]
fn toml_table_parses_and_rejects_unknown_keys() {
    let v: toml::Value = toml::from_str("tp_size = 2\nrank = 0\nmaster_port = 11561").unwrap();
    let cfg = ParallelConfig::from_toml_value(&v).unwrap().unwrap();
    assert_eq!(cfg.tp_size, Some(2));
    assert_eq!(cfg.master_port, Some(11561));

    let bad: toml::Value = toml::from_str("frobnicate = 1").unwrap();
    assert!(ParallelConfig::from_toml_value(&bad).is_err());
}

// --- protocol framing ---------------------------------------------------

#[test]
fn frame_roundtrip_over_tcp() {
    let msg = ControlMessage::Hello {
        version: 1,
        tp_size: 2,
        who: "test".into(),
    };
    let frame = msg.to_frame().unwrap();
    assert_eq!(
        u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize + 4,
        frame.len()
    );
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let t = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        ControlMessage::from_stream(&mut s).unwrap()
    });
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    msg.to_stream(&mut client).unwrap();
    let got = t.join().unwrap();
    assert!(matches!(got, ControlMessage::Hello { tp_size: 2, .. }));
}

#[test]
fn oversized_frame_is_refused_not_read() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let t = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        ControlMessage::from_stream(&mut s).is_err()
    });
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    // Announce a frame larger than the cap without sending the payload.
    client.write_all(&(2u32 << 30).to_le_bytes()).unwrap();
    assert!(t.join().unwrap());
}

#[test]
fn nccl_id_roundtrip_and_malformed_length_rejected() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let t = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let id = receive_nccl_id(&mut stream).unwrap();
        assert_eq!(id, [0x9a; 128]);
        assert!(matches!(receive_nccl_id(&mut stream), Err(ProtocolError::BadNcclId(3))));
    });
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    send_nccl_id(&mut client, &[0x9a; 128]).unwrap();
    ControlMessage::NcclId { id: vec![1, 2, 3] }
        .to_stream(&mut client)
        .unwrap();
    t.join().unwrap();
}

// --- two-role handshake end to end ---------------------------------------

#[test]
fn worker_loop_exits_cleanly_on_graceful_shutdown() {
    let port = free_port();
    let coord = resolved(0, port);
    let work_cfg = resolved(1, port);

    let t = std::thread::spawn(move || coordinate(&coord, false));
    std::thread::sleep(Duration::from_millis(100));
    let w = std::thread::spawn(move || work(&work_cfg));

    let Ok((mut stream, session)) = t.join().unwrap() else {
        panic!("coordinate failed")
    };
    // Session ids are a process-global monotonic counter shared by every
    // test in this binary - assert "was assigned", not a specific value.
    assert!(session >= 1);
    shutdown_worker(&mut stream, true).unwrap();
    assert!(w.join().unwrap().is_ok());
}

#[test]
fn worker_loop_errors_on_non_graceful_shutdown() {
    let port = free_port();
    let coord = resolved(0, port);
    let work_cfg = resolved(1, port);

    let t = std::thread::spawn(move || coordinate(&coord, false));
    std::thread::sleep(Duration::from_millis(100));
    let w = std::thread::spawn(move || work(&work_cfg));

    let Ok((mut stream, _session)) = t.join().unwrap() else {
        panic!("coordinate failed")
    };
    shutdown_worker(&mut stream, false).unwrap();
    assert!(matches!(w.join().unwrap(), Err(BootstrapError::Aborted)));
}

#[test]
fn mismatched_world_size_is_rejected_and_coordinator_keeps_waiting() {
    let port = free_port();
    let coord = resolved(0, port);

    let t = std::thread::spawn(move || coordinate(&coord, false));
    std::thread::sleep(Duration::from_millis(100));

    // Dial with the wrong world size; expect a Reject frame.
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    ControlMessage::Hello {
        version: 1,
        tp_size: 8,
        who: "impostor".into(),
    }
    .to_stream(&mut c)
    .unwrap();
    let reply = ControlMessage::from_stream(&mut c).unwrap();
    assert!(matches!(reply, ControlMessage::Reject { .. }));
    drop(c);

    // The coordinator did not accept the impostor: it is still waiting, so a
    // correct worker can now join and complete the bootstrap.
    let w = std::thread::spawn(move || {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        paddock_dist::protocol::handshake(&mut c, 2, "real-worker").unwrap()
    });
    let session = w.join().unwrap();
    let Ok((mut stream, s2)) = t.join().unwrap() else {
        panic!("coordinate failed")
    };
    assert_eq!(s2, session);
    assert!(session >= 1);
    shutdown_worker(&mut stream, true).unwrap();
}

#[test]
fn protocol_version_mismatch_is_rejected() {
    let port = free_port();
    let coord = resolved(0, port);

    let t = std::thread::spawn(move || coordinate(&coord, false));
    std::thread::sleep(Duration::from_millis(100));

    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    ControlMessage::Hello {
        version: 999,
        tp_size: 2,
        who: "time-traveler".into(),
    }
    .to_stream(&mut c)
    .unwrap();
    let reply = ControlMessage::from_stream(&mut c).unwrap();
    assert!(matches!(reply, ControlMessage::Reject { .. }));
    drop(c);

    // A correct worker still completes bootstrap afterwards.
    let w = std::thread::spawn(move || {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        paddock_dist::protocol::handshake(&mut c, 2, "real-worker").unwrap()
    });
    let session = w.join().unwrap();
    let Ok((mut stream, s2)) = t.join().unwrap() else {
        panic!("coordinate failed")
    };
    assert_eq!(s2, session);
    shutdown_worker(&mut stream, true).unwrap();
}

#[test]
fn unexpected_message_shape_is_rejected() {
    let port = free_port();
    let coord = resolved(0, port);

    let t = std::thread::spawn(move || coordinate(&coord, false));
    std::thread::sleep(Duration::from_millis(100));

    // First frame is not a Hello.
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    ControlMessage::Shutdown { graceful: true }
        .to_stream(&mut c)
        .unwrap();
    let reply = ControlMessage::from_stream(&mut c).unwrap();
    assert!(matches!(reply, ControlMessage::Reject { .. }));
    drop(c);

    let w = std::thread::spawn(move || {
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        paddock_dist::protocol::handshake(&mut c, 2, "real-worker").unwrap()
    });
    let session = w.join().unwrap();
    let Ok((mut stream, s2)) = t.join().unwrap() else {
        panic!("coordinate failed")
    };
    assert_eq!(s2, session);
    shutdown_worker(&mut stream, true).unwrap();
}
