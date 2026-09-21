import AppKit
import PaddockStudio
import WebKit

extension Checks {
  /// Render against the actual bundled CSS and actual reader. The tooltip and
  /// zoom menu use their real event handlers; latent authoring surfaces get
  /// inert DOM probes, never new product commands or annotation mutations.
  @MainActor func checkViewerOverlays(_ session: StudioWorkspace) async throws {
    try await check(
      "PDF painted before overlay appearance checks",
      """
      const canvas=[...document.querySelectorAll('.lector-page canvas')].find(c=>c.width>0&&c.height>0);
      if(!canvas)return false;
      // The presenter can own a bitmaprenderer context. Read a copy instead
      // of trying to claim its context as 2D (which returns null).
      const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;
      const ctx=copy.getContext('2d');ctx.drawImage(canvas,0,0);
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;
      let ink=0;for(let i=0;i<pixels.length;i+=16)if(pixels[i+3]>200&&Math.min(pixels[i],pixels[i+1],pixels[i+2])<180)ink++;
      if(ink<100)return false;
      window.viewerThemeCanvas=canvas;window.viewerThemePixels=canvas.toDataURL();return true;
      """)
    _ = try await session.webView.evaluateJavaScript(
      """
      window.viewerThemePage=document.querySelector('.lector-page');
      window.viewerThemeTipTarget=document.querySelector('.lector-toolbar .lector-btn:not([disabled])[aria-label]');
      document.querySelector('.lector-zoom__chevron').click();
      window.viewerThemeMenu=document.querySelector('.lector-zoom .lector-dropdown__menu--open');
      window.viewerThemeTipTarget.dispatchEvent(new MouseEvent('mouseenter'));
      """)
    try await check(
      "reader opens its real body-portalled tooltip and zoom menu",
      """
      const tip=document.querySelector('body > .lector-tooltip');
      if(!tip||getComputedStyle(tip).display==='none'||!tip.textContent||!window.viewerThemeMenu)return false;
      window.viewerThemeTip=tip;return true;
      """)
    for dark in [true, false, true] {
      await session.setDark(dark)
      try await check(
        "open tooltip/menu update in place in \(dark ? "dark" : "light")",
        Self.viewerThemeHelpers + """
          const tip=window.viewerThemeTip,menu=window.viewerThemeMenu;
          if(!tip.isConnected||!menu.isConnected||getComputedStyle(tip).display==='none'||!menu.classList.contains('lector-dropdown__menu--open'))return false;
          for(const el of [tip,menu])surface(el);
          assert(tip.parentElement===document.body,'tooltip is a real detached portal');
          assert(window.viewerThemeTipTarget.getAttribute('aria-describedby')===tip.id,'tooltip retains accessibility relationship');
          assert(getComputedStyle(menu.querySelector('button')).fontSize==='12px','native menu item type size');
          assert(window.viewerThemePage===document.querySelector('.lector-page'),'appearance remounted page');
          const canvas=window.viewerThemeCanvas;
          assert(canvas.isConnected&&canvas.toDataURL()===window.viewerThemePixels,'appearance changed PDF pixels');
          assert(getComputedStyle(canvas).filter==='none','PDF canvas was filtered');
          return true;
          """)
      try await check(
        "internal reader surfaces, controls and semantic colors in \(dark ? "dark" : "light")",
        Self.viewerThemeHelpers + Self.readerSurfaceProbes)
    }
    _ = try await session.webView.evaluateJavaScript(
      """
      window.viewerThemeTipTarget.dispatchEvent(new MouseEvent('mouseleave'));
      document.querySelector('.lector-zoom__chevron').click();
      for(const key of ['viewerThemePage','viewerThemeCanvas','viewerThemePixels','viewerThemeTip','viewerThemeTipTarget','viewerThemeMenu'])delete window[key];
      """)
    try await check(
      "reader overlay checks leave no popup or fixture behind",
      "!document.querySelector('[data-viewer-theme-probe]')&&!document.querySelector('.lector-zoom .lector-dropdown__menu--open')&&getComputedStyle(document.querySelector('body > .lector-tooltip')).display==='none'"
    )
  }

  @MainActor func checkMetadataTheme(_ session: StudioWorkspace) async throws {
    // Exercise a document-pane-sized WebView, not only a full-width window.
    let originalFrame = window?.frame
    window?.setContentSize(NSSize(width: 560, height: 820))
    defer {
      if let originalFrame { window?.setFrame(originalFrame, display: true) }
    }
    try await check(
      "real metadata rows loaded for appearance checks",
      "!!document.querySelector('.pv__content--info .fmp__v')")
    _ = try await session.webView.evaluateJavaScript(
      "window.viewerThemeDetails=document.querySelector('.pv__content--info');void 0")
    for dark in [true, false, true] {
      await session.setDark(dark)
      try await check(
        "metadata portal, bands, fields and warnings in \(dark ? "dark" : "light")",
        Self.viewerThemeHelpers + """
          const dialog=document.querySelector('.pv__content--info');
          assert(dialog===window.viewerThemeDetails,'metadata remounted during appearance change');
          surface(dialog);
          const rect=dialog.getBoundingClientRect();
          assert(rect.left>=0&&rect.right<=innerWidth&&rect.top>=0&&rect.bottom<=innerHeight,'metadata escapes document viewport');
          const overlay=getComputedStyle(document.querySelector('.pv__overlay'));
          assert(overlay.backgroundColor===color('--pk-bg-overlay'),'metadata overlay tint');
          assert(overlay.backdropFilter==='none','metadata backdrop blur');
          for(const selector of ['.pv__bar','.pv__tabs','.pv__body','.fmp__rows']) {
            assert(getComputedStyle(dialog.querySelector(selector)).backgroundColor===color('--pk-bg-popup'),selector+' should share popup ground');
          }
          const row=dialog.querySelector('.fmp__v');
          assert(getComputedStyle(row).backgroundColor==='rgba(0, 0, 0, 0)','metadata has no contrasting zebra bands');
          // Clone a real scoped-CSS row: no forged Vue scope IDs, no modifying
          // the fixture's metadata. The flag must keep its warning color/tint.
          const warning=row.cloneNode(true);warning.classList.add('fmp__v--flag');row.parentElement.append(warning);
          try {
            const s=getComputedStyle(warning);
            assert(s.color===color('--pk-status-warning'),'metadata security warning lost its color');
            assert(s.backgroundColor!=='rgba(0, 0, 0, 0)','metadata security warning lost its tint');
          } finally { warning.remove(); }
          assert(dialog.contains(document.activeElement),'metadata dialog lost focus containment');
          return true;
          """)
      if let path = ProcessInfo.processInfo.environment["PADDOCK_VIEWER_SNAPSHOTS"] {
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let snapshot = try await session.webView.takeSnapshot(configuration: nil)
        if let data = snapshot.tiffRepresentation,
          let png = NSBitmapImageRep(data: data)?.representation(using: .png, properties: [:])
        {
          try png.write(to: directory.appending(path: "metadata-\(dark ? "dark" : "light").png"))
        }
      }
    }
    _ = try await session.webView.evaluateJavaScript("delete window.viewerThemeDetails")
  }

  // Resolve CSS tokens through the browser, not a hand-copied palette or hex
  // parser. This also tests cascade and color serialization in real WebKit.
  private static let viewerThemeHelpers = """
    const assert=(ok,message)=>{if(!ok)throw new Error(message)};
    const token=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const color=name=>{
      const probe=document.createElement('span');probe.style.color='var('+name+')';document.body.append(probe);
      const value=getComputedStyle(probe).color;probe.remove();return value;
    };
    const surface=el=>{
      const s=getComputedStyle(el);
      assert(s.backgroundColor===color('--pk-bg-popup'),el.className+' background: '+s.backgroundColor);
      assert(s.color===color('--pk-text-primary'),el.className+' foreground: '+s.color);
      assert(s.borderRadius===token('--pk-radius-lg'),el.className+' radius: '+s.borderRadius);
      assert(s.borderTopColor===color('--pk-border-default'),el.className+' border');
      const probe=document.createElement('span');probe.style.boxShadow='var(--pk-shadow-md)';document.body.append(probe);
      const shadow=getComputedStyle(probe).boxShadow;probe.remove();
      assert(s.boxShadow===shadow,el.className+' shadow: '+s.boxShadow);
    };
    const contrast=(a,b)=>{
      const luminance=rgb=>rgb.match(/[\\d.]+/g).slice(0,3).map(Number).map(v=>v/255).map(v=>v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[0.2126,0.7152,0.0722][i],0);
      const x=luminance(a),y=luminance(b);return (Math.max(x,y)+0.05)/(Math.min(x,y)+0.05);
    };
    """

  private static let readerSurfaceProbes = """
    const host=document.createElement('div');host.dataset.viewerThemeProbe='';
    document.querySelector('.lector-workspace').append(host);
    try {
      const classes=['lector-context-menu','lector-tool-group__panel','lector-annot-popover',
        'lector-sig-status-popover','lector-cmt-thread__props-dropdown','lector-cmt-mention-dropdown',
        'lector-modal','lector-poly-tooltip','lector-toast','lector-capture-action-bar',
        'lector-text-sel-toolbar','lector-multi-select-bar','lector-search-bar'];
      for(const cls of classes){const el=document.createElement('div');el.className=cls;host.append(el);surface(el);}
      const modal=host.querySelector('.lector-modal');
      modal.innerHTML='<button class="lector-modal__btn lector-modal__btn--primary">Confirm</button><button class="lector-modal__btn lector-modal__btn--primary" disabled>Disabled</button><button class="lector-modal__btn lector-modal__btn--danger">Delete</button><input class="lector-modal__input" aria-label="Fixture input"><input type="checkbox" checked><div class="lector-signature-card__detail--warn">Warning</div>';
      const primary=modal.querySelector('.lector-modal__btn--primary'),p=getComputedStyle(primary);
      assert(p.color===color('--pk-text-inverse'),'primary action foreground');
      assert(p.backgroundColor===color('--pk-accent'),'primary action background');
      assert(contrast(p.color,p.backgroundColor)>=4.5,'primary action contrast is below 4.5:1');
      const disabled=modal.querySelector(':disabled');
      assert(disabled.disabled&&Number(getComputedStyle(disabled).opacity)<1,'disabled affordance lost');
      const input=modal.querySelector('.lector-modal__input'),s=getComputedStyle(input);
      assert(s.backgroundColor===color('--pk-bg-base')&&s.borderRadius===token('--pk-radius-md'),'input surface/radius');
      input.focus();
      assert(document.activeElement===input&&getComputedStyle(input).outlineColor===color('--pk-border-focus')&&getComputedStyle(input).outlineWidth==='2px','input keyboard focus ring');
      const danger=getComputedStyle(modal.querySelector('.lector-modal__btn--danger'));
      assert(danger.backgroundColor===color('--pk-status-error'),'destructive action lost its red');
      assert(getComputedStyle(modal.querySelector('.lector-signature-card__detail--warn')).color===color('--pk-status-warning'),'signature warning lost its color');
      for(const cls of ['lector-modal-overlay','lector-sidebar-backdrop']) {
        const el=document.createElement('div');el.className=cls;host.append(el);
        assert(getComputedStyle(el).backgroundColor===color('--pk-bg-overlay'),cls+' tint');
      }
      const selection=host.querySelector('.lector-text-sel-toolbar');
      selection.innerHTML='<button class="lector-text-sel-toolbar__btn">Copy</button>';
      assert(getComputedStyle(selection.firstElementChild).color===color('--pk-text-primary'),'selection toolbar action contrast');
      const swatch=document.createElement('button');swatch.className='lector-annot-popover__color';swatch.style.backgroundColor='rgb(219, 37, 173)';host.querySelector('.lector-annot-popover').append(swatch);
      assert(getComputedStyle(swatch).backgroundColor==='rgb(219, 37, 173)','annotation color changed');
      const search=document.createElement('span');search.className='lector-search-highlight';host.append(search);
      assert(getComputedStyle(search).backgroundColor==='rgba(250, 204, 21, 0.4)','search match lost its highlight');
      return true;
    } finally { host.remove(); }
    """
}
