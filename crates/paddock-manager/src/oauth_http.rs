//! OAuth metadata is untrusted: pin validated DNS answers for each request,
//! refuse redirects/proxies and bound bodies. A remote server cannot use
//! discovery to reach loopback/private services. Explicit loopback MCP servers
//! may use loopback authorization servers for local development.
use serde_json::Value;
use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

fn loopback(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<IpAddr>()
                .is_ok_and(|a| a.is_loopback())
    })
}
fn public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => {
            !a.is_private()
                && !a.is_loopback()
                && !a.is_link_local()
                && !a.is_unspecified()
                && !a.is_multicast()
                && !a.is_broadcast()
                && !a.is_documentation()
                && a.octets()[0] != 0
                && a.octets()[0] < 240
                && !(a.octets()[0] == 100 && (64..128).contains(&a.octets()[1]))
                && !(a.octets()[0] == 198 && (18..=19).contains(&a.octets()[1]))
        }
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                public(IpAddr::V4(v4))
            } else {
                !a.is_loopback()
                    && !a.is_unspecified()
                    && !a.is_multicast()
                    && !a.is_unique_local()
                    && !a.is_unicast_link_local()
                    && (a.segments()[0] & 0xe000) == 0x2000
            }
        }
    }
}
pub(crate) async fn client(url: &str, resource: &str) -> Result<reqwest::Client, String> {
    let url = reqwest::Url::parse(&crate::integrations::safe_url(url)?)
        .map_err(|_| "Invalid authorization URL.")?;
    let resource = reqwest::Url::parse(resource).map_err(|_| "Invalid MCP resource URL.")?;
    let allow_local = loopback(&resource) && loopback(&url);
    let host = url
        .host_str()
        .ok_or("Authorization URL has no host.")?
        .trim_matches(['[', ']']);
    let port = url
        .port_or_known_default()
        .ok_or("Invalid authorization port.")?;
    let addresses: Vec<SocketAddr> = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::lookup_host((host, port)),
    )
    .await
    .map_err(|_| "Authorization DNS lookup timed out.")?
    .map_err(|_| "Authorization host could not be resolved.")?
    .take(32)
    .collect();
    if addresses.is_empty()
        || addresses
            .iter()
            .any(|a| !public(a.ip()) && !(allow_local && a.ip().is_loopback()))
    {
        return Err("Authorization metadata points to a private or unsupported address.".into());
    }
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(8))
        .timeout(Duration::from_secs(30))
        .resolve_to_addrs(host, &addresses)
        .build()
        .map_err(|_| "Authorization network client unavailable.".into())
}
pub(crate) async fn json(mut response: reqwest::Response) -> Result<Value, String> {
    if !response.status().is_success() {
        return Err("Authorization server refused the request.".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "Authorization response failed.")?
    {
        if bytes.len() + chunk.len() > 1024 * 1024 {
            return Err("Authorization response exceeds its limit.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid authorization response.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn non_public_addresses_are_refused() {
        for ip in [
            "127.0.0.1",
            "10.1.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "fe80::1",
        ] {
            assert!(!public(ip.parse().unwrap()), "{ip}");
        }
        assert!(public("8.8.8.8".parse().unwrap()));
    }
}
