//! Opt-in local serving benchmarks. Synthetic prompts only. Token counts come
//! from server usage, never SSE-event counts. Streaming gaps are event gaps,
//! not claimed per-token GPU latency. This does not clear a user's cache.
use crate::routes::AppState;
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Serialize)]
struct Sample {
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: Option<u64>,
    ttft_ms: f64,
    duration_ms: f64,
    event_gap_ms: Vec<f64>,
    finish_reason: String,
}

#[derive(Default)]
struct StreamMetrics {
    total_bytes: usize,
    pending: Vec<u8>,
    first_ms: Option<f64>,
    last_ms: Option<f64>,
    gaps: Vec<f64>,
    usage: Option<Value>,
    finish: Option<String>,
    done: bool,
}
impl StreamMetrics {
    fn push(&mut self, bytes: &[u8], elapsed_ms: f64) -> Result<(), String> {
        self.total_bytes = self.total_bytes.saturating_add(bytes.len());
        if self.total_bytes > 16 * 1024 * 1024 || self.gaps.len() > 16_384 {
            return Err("Benchmark output exceeded its bounded fixture size.".into());
        }
        self.pending.extend_from_slice(bytes);
        if self.pending.len() > 1024 * 1024 {
            return Err("Benchmark stream exceeded its frame limit.".into());
        }
        while let Some(end) = self.pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=end).collect();
            let line = std::str::from_utf8(&line)
                .map_err(|_| "Invalid benchmark stream encoding.")?
                .trim();
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                self.done = true;
                continue;
            }
            let event: Value =
                serde_json::from_str(data).map_err(|_| "Invalid benchmark stream frame.")?;
            if event.get("error").is_some() {
                return Err(
                    "The model reported an error during the benchmark. Check instance logs.".into(),
                );
            }
            if event["usage"].is_object() {
                self.usage = Some(event["usage"].clone());
            }
            let choice = &event["choices"][0];
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish = Some(reason.into());
            }
            let delta = &choice["delta"];
            let has_output = ["content", "reasoning_content", "reasoning"]
                .iter()
                .any(|key| delta[key].as_str().is_some_and(|s| !s.is_empty()))
                || choice["text"].as_str().is_some_and(|s| !s.is_empty());
            if has_output {
                self.first_ms.get_or_insert(elapsed_ms);
                if let Some(last) = self.last_ms {
                    self.gaps.push((elapsed_ms - last).max(0.));
                }
                self.last_ms = Some(elapsed_ms);
            }
        }
        Ok(())
    }
    fn finish(self, duration_ms: f64) -> Result<Sample, String> {
        if !self.done {
            return Err("The benchmark stream disconnected before completion.".into());
        }
        let usage = self
            .usage
            .ok_or("The model did not report token usage; no throughput is estimated.")?;
        let output_tokens = usage["completion_tokens"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or("No output token usage was reported.")?;
        Ok(Sample {
            input_tokens: usage["prompt_tokens"]
                .as_u64()
                .ok_or("No input token usage was reported.")?,
            output_tokens,
            cached_tokens: usage["prompt_tokens_details"]["cached_tokens"].as_u64(),
            ttft_ms: self.first_ms.ok_or("No streamed output was observed.")?,
            duration_ms,
            event_gap_ms: self.gaps,
            finish_reason: self
                .finish
                .ok_or("No terminal finish reason was reported.")?,
        })
    }
}

async fn sample(
    client: &reqwest::Client,
    url: &str,
    key: Option<&str>,
    model: &str,
    words: usize,
    tokens: u32,
) -> Result<Sample, String> {
    // Unique first content per request prevents this fixture becoming a warm
    // repeated-prefix test. Model chat-template prefix reuse can still occur.
    let prompt = format!(
        "Benchmark {}. Read these notes: {}\nWrite a detailed implementation plan explaining every stage, in plain text.",
        uuid::Uuid::new_v4(),
        "The input buffer is valid and the function returns a checked value. ".repeat(words / 13)
    );
    let body = json!({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":tokens,"temperature":0.,"stream":true,"stream_options":{"include_usage":true}});
    let start = Instant::now();
    let mut request = client.post(url).json(&body);
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|_| "Could not reach the selected benchmark instance.")?;
    if !response.status().is_success() {
        return Err(format!(
            "The benchmark request failed with HTTP {}. Check the instance’s context limit and logs.",
            response.status()
        ));
    }
    let mut stream = response.bytes_stream();
    let mut metrics = StreamMetrics::default();
    while let Some(bytes) = stream.next().await {
        metrics.push(
            &bytes.map_err(|_| "Benchmark stream disconnected.")?,
            start.elapsed().as_secs_f64() * 1000.,
        )?;
        if metrics.done {
            break;
        }
    }
    metrics.finish(start.elapsed().as_secs_f64() * 1000.)
}

pub async fn run(
    state: Arc<AppState>,
    port: u16,
    pid: u32,
    concurrency: usize,
    long: bool,
) -> Result<Value, String> {
    if ![1, 4].contains(&concurrency) {
        return Err("Choose one or four concurrent requests.".into());
    }
    let runner = state
        .supervisor
        .list()
        .await
        .into_iter()
        .find(|r| r.port == port && r.pid == pid && r.status == "ok" && r.in_flight == Some(0))
        .ok_or("Select an idle, running instance. Other requests must finish first.")?;
    let model = runner
        .model
        .ok_or("The benchmark requires a text-generation model.")?;
    let key = state.supervisor.runner_key_checked(port, pid).await?;
    let config_revision = state.supervisor.config_file_hash(port);
    let settings = crate::native_endpoints::projection(&state.supervisor, port).ok();
    let hardware_before =
        serde_json::to_value(state.gpu.latest()).map_err(|_| "Could not record hardware state.")?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|_| "Could not create benchmark client.")?;
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    sample(&client, &url, key.as_deref(), &model, 128, 32).await?;
    let words = if long { 2048 } else { 128 };
    let mut samples = Vec::new();
    let mut wave_seconds = Vec::new();
    for _ in 0..3 {
        // Check process/config identity between trials. A user restart never
        // silently joins results from two configurations or checkpoints.
        state.supervisor.runner_key_checked(port, pid).await?;
        if state.supervisor.config_file_hash(port) != config_revision {
            return Err(
                "Instance settings changed during the benchmark. Results were discarded.".into(),
            );
        }
        let start = Instant::now();
        let trials =
            (0..concurrency).map(|_| sample(&client, &url, key.as_deref(), &model, words, 128));
        samples.extend(futures_util::future::try_join_all(trials).await?);
        wave_seconds.push(start.elapsed().as_secs_f64());
    }
    let wall_s: f64 = wave_seconds.iter().sum();
    let output: u64 = samples.iter().map(|s| s.output_tokens).sum();
    let ttft: Vec<f64> = samples.iter().map(|s| s.ttft_ms).collect();
    let gaps: Vec<f64> = samples
        .iter()
        .flat_map(|s| s.event_gap_ms.iter().copied())
        .collect();
    state.supervisor.runner_key_checked(port, pid).await?;
    if state.supervisor.config_file_hash(port) != config_revision {
        return Err(
            "Instance settings changed during the benchmark. Results were discarded.".into(),
        );
    }
    let report = json!({"schema":1,"id":uuid::Uuid::new_v4().to_string(),"created_at_ms":chrono::Utc::now().timestamp_millis(),"model":model,"port":port,"pid":pid,"runner_version":runner.version,
        "manager_version":env!("CARGO_PKG_VERSION"),"config_revision":config_revision,
        "max_ctx":settings.as_ref().and_then(|s|s["max_ctx"].as_u64()),"max_batch":settings.as_ref().and_then(|s|s["max_batch"].as_u64()),
        "concurrency":concurrency,"prompt_words":words,"output_limit":128,"trials":3,"warmups":1,
        "cache_policy":"Unique request prefixes; chat-template prefix reuse may remain. Cache not cleared.",
        "aggregate_output_tok_s":output as f64 / wall_s,"wall_seconds":wall_s,"output_tokens":output,
        "ttft_median_ms":percentile(&ttft,0.5),"stream_event_gap_p99_ms":percentile(&gaps,0.99),
        "spec":settings.as_ref().and_then(|s| s["settings"]["spec"].as_str()),
        "kv_cache_dtype":settings.as_ref().and_then(|s|s["settings"]["kv_cache_dtype"].as_str()),
        "samples":samples,"wave_seconds":wave_seconds,"hardware_before":hardware_before,"hardware_after":state.gpu.latest()});
    state
        .db
        .save_native_benchmark(&report)
        .map_err(|_| "The benchmark completed but could not be saved.")?;
    Ok(report)
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(
        sorted[((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn http_fixture_reports_usage_and_stops_at_done_without_waiting_for_eof() {
        use axum::{Json, Router, body::Body, http::HeaderMap, response::Response, routing::post};
        use std::convert::Infallible;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = Router::new().route("/v1/chat/completions", post(
            |headers: HeaderMap, Json(body): Json<Value>| async move {
                assert_eq!(headers["authorization"], "Bearer fixture-key");
                assert_eq!(body["model"], "fixture-model");
                assert_eq!(body["max_tokens"], 128);
                assert_eq!(body["stream_options"]["include_usage"], true);
                let chunks = [
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Thinking\"}}]}\r\n\r\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Answer å\"},\"finish_reason\":\"length\"}]}\n\n",
                    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":151,\"completion_tokens\":128,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n\n",
                    "data: [DONE]\n\n",
                ];
                let stream = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, Infallible>))
                    .chain(futures_util::stream::pending());
                Response::builder().header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream)).unwrap()
            }
        ));
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sample(
                &reqwest::Client::new(),
                &format!("http://{address}/v1/chat/completions"),
                Some("fixture-key"),
                "fixture-model",
                128,
                128,
            ),
        )
        .await;
        server.abort();
        let sample = result.unwrap().unwrap();
        assert_eq!(sample.input_tokens, 151);
        assert_eq!(sample.output_tokens, 128);
        assert_eq!(sample.cached_tokens, Some(7));
        assert_eq!(sample.event_gap_ms.len(), 1);
        assert!(sample.duration_ms >= sample.ttft_ms);
    }
    #[test]
    fn fragmented_utf8_reasoning_and_usage_not_event_counts() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"å\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":91,\"completion_tokens\":128}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut stream = StreamMetrics::default();
        for (i, byte) in raw.as_bytes().iter().enumerate() {
            stream.push(&[*byte], i as f64).unwrap();
        }
        let result = stream.finish(1000.).unwrap();
        assert_eq!(result.output_tokens, 128);
        assert_eq!(result.input_tokens, 91);
        assert_eq!(result.event_gap_ms.len(), 1);
        assert!(result.ttft_ms > 0.);
        assert_eq!(result.cached_tokens, None);
    }
    #[test]
    fn disconnected_or_unmetered_streams_are_not_benchmarks() {
        assert!(StreamMetrics::default().finish(10.).is_err());
        let mut stream = StreamMetrics::default();
        stream.push(b"data: [DONE]\n", 2.).unwrap();
        assert!(stream.finish(10.).is_err());
        assert_eq!(percentile(&[], 0.99), None);
        assert_eq!(percentile(&[3., 1., 2.], 0.5), Some(2.));
    }
}
