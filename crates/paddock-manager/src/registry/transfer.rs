//! Bounded range recovery and live progress, shared by native and web downloads.
use super::*;
use futures::StreamExt;
use std::time::Duration;

#[cfg(test)]
#[path = "transfer_tests.rs"]
mod tests;

const ATTEMPTS: usize = 4;
const HEADERS_TIMEOUT: Duration = Duration::from_secs(20);
const IDLE_TIMEOUT: Duration = Duration::from_secs(15);

/// Live bytes are not resume bits. Drop rolls an incomplete attempt back, even
/// on cancellation; only a fully written segment commits them permanently.
pub(super) struct RangeProgress {
    total: Arc<AtomicU64>,
    received: u64,
}
impl RangeProgress {
    fn new(total: &Arc<AtomicU64>) -> Self {
        Self {
            total: total.clone(),
            received: 0,
        }
    }
    fn add(&mut self, bytes: usize) {
        self.received += bytes as u64;
        self.total.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub(super) fn commit(mut self) {
        self.received = 0;
    }
}
impl Drop for RangeProgress {
    fn drop(&mut self) {
        self.total.fetch_sub(self.received, Ordering::Relaxed);
    }
}

fn transport(error: reqwest::Error) -> DlError {
    // Display alone hides the useful body/HTTP2/timeout cause. Do not retain
    // signed URLs from an origin in the user-visible error.
    let error = error.without_url();
    let mut detail = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        detail.push_str(": ");
        detail.push_str(&cause.to_string());
        source = cause.source();
    }
    DlError::Transport(detail)
}

fn retry_delay(error: &DlError, attempt: usize, start: u64) -> Option<Duration> {
    let backoff = Duration::from_millis((250 << attempt) + (start / SEGMENT % 8) * 31);
    match error {
        DlError::Transport(_) => Some(backoff),
        DlError::RetryableStatus { retry_after, .. } => {
            // Never shorten a server-requested delay. Long waits become an
            // actionable error instead of a silently frozen active download.
            match retry_after {
                Some(seconds) if *seconds > 30 => None,
                Some(seconds) => Some(backoff.max(Duration::from_secs(*seconds))),
                None => Some(backoff),
            }
        }
        _ => None, // authorization, missing files, protocol and integrity errors
    }
}

pub(super) async fn fetch_range(
    client: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
    size: u64,
    downloaded: &Arc<AtomicU64>,
) -> Result<(Vec<u8>, RangeProgress), DlError> {
    for attempt in 0..ATTEMPTS {
        let mut progress = RangeProgress::new(downloaded);
        match fetch_once(client, url, start, end, size, &mut progress).await {
            Ok(bytes) => return Ok((bytes, progress)),
            Err(error) => {
                drop(progress);
                if attempt + 1 < ATTEMPTS
                    && let Some(delay) = retry_delay(&error, attempt, start)
                {
                    tokio::time::sleep(delay).await;
                } else {
                    return Err(error);
                }
            }
        }
    }
    unreachable!("the last attempt always returns")
}

async fn fetch_once(
    client: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
    size: u64,
    progress: &mut RangeProgress,
) -> Result<Vec<u8>, DlError> {
    let response = tokio::time::timeout(
        HEADERS_TIMEOUT,
        client
            .get(url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
            .send(),
    )
    .await
    .map_err(|_| DlError::Transport("No response headers for 20 seconds".into()))?
    .map_err(transport)?;
    let status = response.status();
    if matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504) {
        return Err(DlError::RetryableStatus {
            status: status.as_u16(),
            retry_after: response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                // Date-form or malformed delays are not shortened into a
                // quick retry. Surface them for a later explicit Retry.
                .map(|h| {
                    h.to_str()
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(u64::MAX)
                }),
        });
    }
    if !status.is_success() {
        return Err(classify_status(status, url));
    }
    let expected_range = format!("bytes {start}-{end}/{size}");
    if status != reqwest::StatusCode::PARTIAL_CONTENT
        || response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok())
            != Some(&expected_range)
    {
        return Err(DlError::Http(
            "Origin returned an invalid Content-Range".into(),
        ));
    }
    let expected = (end - start + 1) as usize;
    let mut bytes = Vec::with_capacity(expected);
    let mut stream = response.bytes_stream();
    // Time out a silent stream, not a healthy slow transfer of a large range.
    while let Some(chunk) = tokio::time::timeout(IDLE_TIMEOUT, stream.next())
        .await
        .map_err(|_| DlError::Transport("No download data for 15 seconds".into()))?
    {
        let chunk = chunk.map_err(transport)?;
        if chunk.len() > expected.saturating_sub(bytes.len()) {
            return Err(DlError::Size {
                expected: expected as u64,
                got: (bytes.len() + chunk.len()) as u64,
            });
        }
        bytes.extend_from_slice(&chunk);
        progress.add(chunk.len());
    }
    if bytes.len() != expected {
        return Err(DlError::Transport(format!(
            "Incomplete range: received {} of {expected} bytes",
            bytes.len()
        )));
    }
    Ok(bytes)
}
