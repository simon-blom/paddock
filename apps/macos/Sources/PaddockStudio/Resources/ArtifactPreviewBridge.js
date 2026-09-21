// Runs only in the host's isolated WKContentWorld, never in the artifact page.
// No native commands, credentials, attachments or conversation state exist here.
(() => {
  let frame, listener, timer, sent = false, latest;
  const strings = x => Array.isArray(x) ? x.filter(s => typeof s === 'string').slice(0, 32).map(s => s.slice(0, 300)) : [];
  const report = value => window.webkit.messageHandlers.artifactStatus.postMessage(value);
  window.paddockArtifactMount = (html, url, dark) => {
    if (listener) removeEventListener('message', listener);
    clearTimeout(timer); timer = 0; latest = undefined; sent = false;
    if (frame) frame.remove();
    frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-scripts');
    frame.setAttribute('allow', "camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'; display-capture 'none'");
    frame.referrerPolicy = 'no-referrer'; frame.title = 'Artifact preview';
    const current = frame;
    listener = e => {
      if (e.source !== current.contentWindow || e.origin !== 'null') return;
      const m = e.data?.paddockArtifactMissing;
      if (!m || typeof m !== 'object') return;
      latest = { blocked: strings(m.blocked), failed: strings(m.failed) };
      // Bounded telemetry, not a per-event native bridge; a hostile page
      // cannot flood SwiftUI with arbitrarily large reports.
      if (!timer) timer = setTimeout(() => { timer = 0; if (latest) report(latest); }, 250);
    };
    addEventListener('message', listener);
    current.addEventListener('load', () => {
      if (sent || current !== frame) return;
      sent = true;
      current.contentWindow.postMessage({type:'paddock:artifact', html, scrollbar:dark?'rgb(255 255 255 / 28%)':'rgb(0 0 0 / 28%)', minimalScrollbar:true}, '*');
      report({ready:true});
    });
    current.src = url;
    document.body.replaceChildren(current);
  };
})();
