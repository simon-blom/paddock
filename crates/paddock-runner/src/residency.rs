//! Runner-local residency. The listener and model identity outlive GPU resources.
//! Loading and disposal run on blocking workers, never on an HTTP executor.
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, Semaphore};

pub use paddock_admin::residency::{Config, LoadPolicy, Phase, Snapshot};

struct State<T> {
    value: Option<Arc<T>>,
    snapshot: Snapshot,
    last_used: Instant,
    failed_at: Option<Instant>,
}
struct Inner<T> {
    state: Mutex<State<T>>,
    changed: Notify,
    admission: Arc<Semaphore>,
    build: Box<dyn Fn() -> Result<T, String> + Send + Sync>,
    dispose: Box<dyn Fn(T) + Send + Sync>,
}
pub struct Pool<T>(Arc<Inner<T>>);
impl<T> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// A realtime session retains this even while no speech is arriving.
pub struct Lease<T> {
    value: Option<Arc<T>>,
    owner: Arc<Inner<T>>,
}
impl<T> std::ops::Deref for Lease<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_deref().expect("live residency lease")
    }
}
impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        drop(self.value.take());
        let mut state = self.owner.state.lock().unwrap_or_else(|e| e.into_inner());
        state.snapshot.active_leases -= 1;
        state.last_used = Instant::now();
        self.owner.changed.notify_waiters();
    }
}
struct Waiting<T>(Arc<Inner<T>>);
impl<T> Drop for Waiting<T> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.snapshot.waiting_requests -= 1;
        state.last_used = Instant::now();
    }
}

impl<T: Send + Sync + 'static> Pool<T> {
    pub fn new(
        policy: Config,
        initial: Option<T>,
        build: impl Fn() -> Result<T, String> + Send + Sync + 'static,
        dispose: impl Fn(T) + Send + Sync + 'static,
    ) -> Self {
        let phase = if initial.is_some() {
            Phase::Loaded
        } else {
            Phase::Unloaded
        };
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                value: initial.map(Arc::new),
                last_used: Instant::now(),
                failed_at: None,
                snapshot: Snapshot {
                    phase,
                    active_leases: 0,
                    waiting_requests: 0,
                    loads: u64::from(phase == Phase::Loaded),
                    unloads: 0,
                    load_failures: 0,
                    last_load_ms: None,
                    last_error: None,
                    policy,
                },
            }),
            changed: Notify::new(),
            admission: Arc::new(Semaphore::new(32)),
            build: Box::new(build),
            dispose: Box::new(dispose),
        });
        let weak = Arc::downgrade(&inner);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let value = {
                    let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
                    let idle = state
                        .snapshot
                        .policy
                        .unload_after_idle_seconds
                        .is_some_and(|s| state.last_used.elapsed() >= Duration::from_secs(s));
                    if state.snapshot.phase == Phase::Loaded
                        && idle
                        && state.snapshot.active_leases == 0
                        && state.snapshot.waiting_requests == 0
                    {
                        state.snapshot.phase = Phase::Unloading;
                        state.value.take()
                    } else {
                        None
                    }
                };
                if let Some(value) = value {
                    let disposing = inner.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        let value = Arc::try_unwrap(value)
                            .unwrap_or_else(|_| panic!("untracked residency lease"));
                        (disposing.dispose)(value);
                    })
                    .await;
                    let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
                    if result.is_ok() {
                        state.snapshot.phase = Phase::Unloaded;
                        state.snapshot.unloads += 1;
                    } else {
                        state.snapshot.phase = Phase::Failed;
                        state.snapshot.last_error =
                            Some("Model disposal failed; restart the runner.".into());
                        // Never load again when disposal did not prove release.
                        inner.admission.close();
                    }
                    inner.changed.notify_waiters();
                }
            }
        });
        Self(inner)
    }

    pub fn snapshot(&self) -> Snapshot {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot
            .clone()
    }

    /// Permanently close admission and wait for leases/loaders before disposal.
    /// A timeout never advertises that memory has been freed.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        self.0.admission.close();
        self.0.changed.notify_waiters();
        tokio::time::timeout(timeout, async {
            loop {
                let changed = self.0.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                let value = {
                    let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
                    match state.snapshot.phase {
                        Phase::Unloaded => return true,
                        Phase::Failed => return false,
                        Phase::Loaded if state.snapshot.active_leases == 0 => {
                            state.snapshot.phase = Phase::Unloading;
                            state.value.take()
                        }
                        _ => None,
                    }
                };
                if let Some(value) = value {
                    let inner = self.0.clone();
                    // Detached finalizer: cancelling shutdown cannot strand
                    // Unloading or lose the confirmed-disposal notification.
                    tokio::spawn(async move {
                        let disposing = inner.clone();
                        let ok = tokio::task::spawn_blocking(move || {
                            let value = Arc::try_unwrap(value)
                                .unwrap_or_else(|_| panic!("untracked residency lease"));
                            (disposing.dispose)(value);
                        })
                        .await
                        .is_ok();
                        let mut state = inner.state.lock().unwrap_or_else(|e| e.into_inner());
                        state.snapshot.phase = if ok { Phase::Unloaded } else { Phase::Failed };
                        if ok {
                            state.snapshot.unloads += 1;
                        } else {
                            state.snapshot.last_error =
                                Some("Model disposal failed; restart the runner.".into());
                        }
                        inner.changed.notify_waiters();
                    });
                }
                changed.await;
            }
        })
        .await
        .unwrap_or(false)
    }

    pub fn set_policy(&self, policy: Config) -> Result<(), String> {
        policy.validate()?;
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.snapshot.policy = policy;
        state.last_used = Instant::now();
        Ok(())
    }

    /// The saved file remains authoritative. Only this typed policy is read;
    /// no model path, credential, or serving geometry changes apply through it.
    pub fn watch_policy(&self, path: std::path::PathBuf) {
        let weak = Arc::downgrade(&self.0);
        tokio::spawn(async move {
            let mut stamp = None;
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if weak.strong_count() == 0 {
                    return;
                }
                let Ok(meta) = tokio::fs::metadata(&path).await else {
                    continue;
                };
                let next = (meta.modified().ok(), meta.len());
                if stamp == Some(next) {
                    continue;
                }
                stamp = Some(next);
                if meta.len() > 1024 * 1024 {
                    continue;
                }
                let Ok(raw) = tokio::fs::read_to_string(&path).await else {
                    continue;
                };
                if raw.len() > 1024 * 1024 {
                    continue;
                }
                let policy = toml::from_str::<toml::Value>(&raw).ok().and_then(|value| {
                    value
                        .get("residency")
                        .cloned()
                        .map_or(Some(Config::default()), |v| v.try_into().ok())
                });
                let Some(inner) = weak.upgrade() else {
                    return;
                };
                let pool = Pool(inner);
                let Some(policy) = policy.filter(|p| p.validate().is_ok()) else {
                    tracing::warn!(
                        "invalid residency policy in saved configuration; keeping the applied policy"
                    );
                    continue;
                };
                if pool.snapshot().policy == policy {
                    continue;
                }
                let preload = policy.load == LoadPolicy::AtStartup;
                if pool.set_policy(policy).is_ok() && preload {
                    // A slow preload must not stall subsequent saved policy
                    // changes. Single-flight admission still owns the load.
                    tokio::spawn(async move {
                        if let Err(error) = pool.acquire().await {
                            tracing::warn!(%error, "residency policy applied, but model could not load");
                        }
                    });
                }
            }
        });
    }

    pub async fn acquire(&self) -> Result<Lease<T>, String> {
        let _admission = self.0.admission.clone().try_acquire_owned().map_err(|_| {
            "model_unavailable: residency queue is full or the worker failed".to_owned()
        })?;
        let timeout = {
            let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
            state.snapshot.waiting_requests += 1;
            Duration::from_secs(state.snapshot.policy.load_timeout_seconds)
        };
        let _waiting = Waiting(self.0.clone());
        tokio::time::timeout(timeout, async {
            loop {
                // Register before inspecting state: no lost load/unload notification.
                let changed = self.0.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                {
                    let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
                    if self.0.admission.is_closed() {
                        return Err("model_unavailable: residency is closing or failed".into());
                    }
                    match state.snapshot.phase {
                        Phase::Loaded => {
                            let value = state.value.clone().expect("loaded value");
                            state.snapshot.active_leases += 1;
                            return Ok(Lease {
                                value: Some(value),
                                owner: self.0.clone(),
                            });
                        }
                        Phase::Failed
                            if state
                                .failed_at
                                .is_some_and(|at| at.elapsed() < Duration::from_secs(2))
                                || self.0.admission.is_closed() =>
                        {
                            return Err(state
                                .snapshot
                                .last_error
                                .clone()
                                .unwrap_or_else(|| "model_unavailable".into()));
                        }
                        Phase::Unloaded | Phase::Failed => {
                            state.snapshot.phase = Phase::Loading;
                            let inner = self.0.clone();
                            // Detached from individual callers: cancellation cannot strand Loading.
                            tokio::spawn(async move {
                                let loading = inner.clone();
                                let start = Instant::now();
                                let result =
                                    tokio::task::spawn_blocking(move || (loading.build)()).await;
                                let mut state =
                                    inner.state.lock().unwrap_or_else(|e| e.into_inner());
                                state.last_used = Instant::now();
                                match result {
                                    Ok(Ok(value)) => {
                                        state.value = Some(Arc::new(value));
                                        state.snapshot.phase = Phase::Loaded;
                                        state.snapshot.loads += 1;
                                        state.snapshot.last_error = None;
                                        state.failed_at = None;
                                        state.snapshot.last_load_ms =
                                            Some(start.elapsed().as_millis() as u64);
                                    }
                                    other => {
                                        state.snapshot.phase = Phase::Failed;
                                        state.snapshot.load_failures += 1;
                                        state.failed_at = Some(Instant::now());
                                        if other.is_err() {
                                            inner.admission.close();
                                        }
                                        state.snapshot.last_error = Some(match other {
                                            Ok(Err(error)) => error,
                                            _ => "Model loader failed; restart the runner.".into(),
                                        });
                                    }
                                }
                                inner.changed.notify_waiters();
                            });
                        }
                        _ => {}
                    }
                }
                changed.await;
            }
        })
        .await
        .map_err(|_| "model_load_timeout: timed out waiting for model residency".to_owned())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    fn policy() -> Config {
        Config {
            load: LoadPolicy::OnDemand,
            unload_after_idle_seconds: Some(0),
            load_timeout_seconds: 1,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_waits_for_leases_and_rejects_further_admission() {
        let disposed = Arc::new(AtomicUsize::new(0));
        let done = disposed.clone();
        let pool = Pool::new(
            Config::default(),
            Some(42),
            || Ok(43),
            move |_| {
                done.fetch_add(1, SeqCst);
            },
        );
        let lease = pool.acquire().await.unwrap();
        let closing = pool.clone();
        let task = tokio::spawn(async move { closing.shutdown(Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(pool.acquire().await.is_err());
        assert_eq!(disposed.load(SeqCst), 0);
        drop(lease);
        assert!(task.await.unwrap());
        assert_eq!(disposed.load(SeqCst), 1);
        assert_eq!(pool.snapshot().phase, Phase::Unloaded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_timeout_keeps_disposal_owned_and_never_claims_release_early() {
        let pool = Pool::new(
            Config::default(),
            Some(42),
            || Ok(43),
            |_| std::thread::sleep(Duration::from_millis(150)),
        );
        assert!(!pool.shutdown(Duration::from_millis(20)).await);
        assert_eq!(pool.snapshot().phase, Phase::Unloading);
        phase(&pool, Phase::Unloaded).await;
        assert_eq!(pool.snapshot().unloads, 1);
        assert!(pool.acquire().await.is_err());
    }
    async fn phase<T: Send + Sync + 'static>(pool: &Pool<T>, expected: Phase) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.snapshot().phase != expected {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("residency transition deadline");
    }

    #[test]
    fn config_defaults_are_eager_and_explicit_policy_is_strict() {
        let default: Config = toml::from_str("").unwrap();
        assert!(!default.enabled());
        assert!(toml::from_str::<Config>("load = 'anything'").is_err());
        assert!(toml::from_str::<Config>("unload_after_idle_seconds = -1").is_err());
        assert!(toml::from_str::<Config>("unload_after_idel_seconds = 60").is_err());
        assert!(
            Config {
                load_timeout_seconds: 0,
                ..default
            }
            .validate()
            .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_load_can_recover_when_capacity_returns() {
        let capacity = Arc::new(AtomicUsize::new(0));
        let available = capacity.clone();
        let pool = Pool::new(
            policy(),
            None,
            move || {
                if available.load(SeqCst) == 0 {
                    Err("insufficient_memory".into())
                } else {
                    Ok(42)
                }
            },
            |_| {},
        );
        assert!(pool.acquire().await.is_err());
        capacity.store(1, SeqCst);
        tokio::time::sleep(Duration::from_millis(2100)).await;
        let lease = pool.acquire().await.unwrap();
        assert_eq!(*lease, 42);
        assert_eq!(pool.snapshot().load_failures, 1);
        assert_eq!(pool.snapshot().loads, 1);
        assert!(pool.snapshot().last_error.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn saved_policy_changes_live_and_invalid_edits_keep_applied_policy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runner.toml");
        std::fs::write(&path, "[residency]\nload='on_demand'\n").unwrap();
        let initial = Config {
            load: LoadPolicy::OnDemand,
            ..Default::default()
        };
        let pool = Pool::new(initial.clone(), None, || Ok(42), |_| {});
        pool.watch_policy(path.clone());
        std::fs::write(&path, "[residency]\nload_timeout_seconds=0\n").unwrap();
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert_eq!(pool.snapshot().policy, initial);
        assert_eq!(pool.snapshot().loads, 0);
        std::fs::write(&path, "[residency]\nload='at_startup'\n").unwrap();
        phase(&pool, Phase::Loaded).await;
        assert_eq!(pool.snapshot().policy, Config::default());
        let lease = pool.acquire().await.unwrap();
        std::fs::write(
            &path,
            "[residency]\nload='on_demand'\nunload_after_idle_seconds=0\n",
        )
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.snapshot().policy.unload_after_idle_seconds != Some(0) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(pool.snapshot().phase, Phase::Loaded);
        drop(lease);
        phase(&pool, Phase::Unloaded).await;
        assert_eq!(pool.snapshot().loads, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_arrivals_share_one_load_and_leases_prevent_unload() {
        let loads = Arc::new(AtomicUsize::new(0));
        let count = loads.clone();
        let disposed = Arc::new(AtomicUsize::new(0));
        let drops = disposed.clone();
        let pool = Pool::new(
            policy(),
            None,
            move || {
                count.fetch_add(1, SeqCst);
                std::thread::sleep(Duration::from_millis(40));
                Ok(42)
            },
            move |_| {
                drops.fetch_add(1, SeqCst);
            },
        );
        assert_eq!(loads.load(SeqCst), 0);
        let leases = futures::future::join_all((0..16).map(|_| pool.acquire())).await;
        assert!(leases.iter().all(|r| r.as_ref().is_ok_and(|v| **v == 42)));
        assert_eq!(loads.load(SeqCst), 1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(pool.snapshot().active_leases, 16);
        assert_eq!(disposed.load(SeqCst), 0);
        drop(leases);
        phase(&pool, Phase::Unloaded).await;
        assert_eq!(disposed.load(SeqCst), 1);
        let lease = pool.acquire().await.unwrap();
        assert_eq!(loads.load(SeqCst), 2);
        drop(lease);
        phase(&pool, Phase::Unloaded).await;
        assert_eq!(pool.snapshot().unloads, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_during_load_does_not_strand_loading_or_leak_resource() {
        let (release, wait) = std::sync::mpsc::channel();
        let wait = Mutex::new(wait);
        let pool = Pool::new(
            policy(),
            None,
            move || {
                wait.lock().unwrap().recv().unwrap();
                Ok(1)
            },
            |_| {},
        );
        let request = tokio::spawn({
            let pool = pool.clone();
            async move { pool.acquire().await }
        });
        phase(&pool, Phase::Loading).await;
        request.abort();
        let _ = request.await;
        assert_eq!(pool.snapshot().waiting_requests, 0);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.snapshot().unloads != 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(pool.snapshot().phase, Phase::Unloaded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn insufficient_memory_is_shared_and_does_not_retry_for_each_waiter() {
        let loads = Arc::new(AtomicUsize::new(0));
        let count = loads.clone();
        let pool: Pool<u64> = Pool::new(
            policy(),
            None,
            move || {
                count.fetch_add(1, SeqCst);
                std::thread::sleep(Duration::from_millis(40));
                Err("insufficient_memory: requires 100 bytes, available 50 bytes".into())
            },
            |_| {},
        );
        let replies = futures::future::join_all((0..16).map(|_| pool.acquire())).await;
        assert!(replies.iter().all(|r| {
            r.as_ref()
                .err()
                .is_some_and(|e| e.starts_with("insufficient_memory:"))
        }));
        assert_eq!(loads.load(SeqCst), 1);
        assert_eq!(pool.snapshot().phase, Phase::Failed);
        assert_eq!(pool.snapshot().waiting_requests, 0);
        assert!(pool.acquire().await.is_err());
        assert_eq!(loads.load(SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn arrivals_during_disposal_wait_until_memory_is_released() {
        let (release, wait) = std::sync::mpsc::channel();
        let wait = Mutex::new(wait);
        let loads = Arc::new(AtomicUsize::new(0));
        let count = loads.clone();
        let pool = Pool::new(
            policy(),
            Some(1),
            move || {
                count.fetch_add(1, SeqCst);
                Ok(2)
            },
            move |_| {
                wait.lock().unwrap().recv().unwrap();
            },
        );
        phase(&pool, Phase::Unloading).await;
        let request = tokio::spawn({
            let pool = pool.clone();
            async move { pool.acquire().await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(loads.load(SeqCst), 0);
        release.send(()).unwrap();
        let lease = request.await.unwrap().unwrap();
        assert_eq!(*lease, 2);
        pool.set_policy(Config::default()).unwrap();
        drop(lease);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_load_has_bounded_wait_and_bounded_queue() {
        let (release, wait) = std::sync::mpsc::channel();
        let wait = Mutex::new(wait);
        let pool = Pool::new(
            policy(),
            None,
            move || {
                wait.lock().unwrap().recv().unwrap();
                Ok(1)
            },
            |_| {},
        );
        let mut requests = Vec::new();
        for _ in 0..32 {
            requests.push(tokio::spawn({
                let pool = pool.clone();
                async move { pool.acquire().await }
            }));
        }
        tokio::time::timeout(Duration::from_secs(2), async {
            while pool.snapshot().waiting_requests < 32 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(
            pool.acquire()
                .await
                .err()
                .unwrap()
                .contains("queue is full")
        );
        for request in requests {
            assert!(
                request
                    .await
                    .unwrap()
                    .err()
                    .unwrap()
                    .starts_with("model_load_timeout:")
            );
        }
        release.send(()).unwrap();
        phase(&pool, Phase::Unloaded).await;
    }
}
