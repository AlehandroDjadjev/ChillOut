/* ChillOut WRF playback — animates the real per-frame model grids embedded in the result.
   Data comes from postprocessor._build_animation: quantized uint8 frames (base64) for the
   baseline and candidate runs, plus a shared value scale and geographic bounds. The Δ layer
   is derived here (candidate − baseline). No simulation runs in the browser; we only render
   frames WRF actually produced. Exposes window.WrfPlayer with load(animation) / reset(). */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  var FRAME_MS = 750; // wall-clock time each real frame is held during playback
  var ICON_PLAY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var ICON_PAUSE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';

  function b64ToBytes(s) {
    var bin = atob(s), n = bin.length, arr = new Uint8Array(n);
    for (var i = 0; i < n; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }

  // Absolute field ramp (cool → amber → hot), warm-theme thermal.
  function absColor(t) {
    t = clamp(t, 0, 1);
    if (t < 0.5) {
      var k = t / 0.5; // cool blue → amber
      return [lerp(58, 255, k) | 0, lerp(110, 177, k) | 0, lerp(165, 82, k) | 0];
    }
    var k2 = (t - 0.5) / 0.5; // amber → terracotta
    return [lerp(255, 226, k2) | 0, lerp(177, 85, k2) | 0, lerp(82, 58, k2) | 0];
  }

  // Diverging Δ ramp centred on 0: cooling = blue, warming = terracotta.
  function deltaColor(d) {
    d = clamp(d, -1, 1);
    if (d < 0) {
      var k = -d; // 0 → cream, 1 → cool blue
      return [lerp(245, 58, k) | 0, lerp(236, 110, k) | 0, lerp(225, 165, k) | 0];
    }
    var k2 = d; // 0 → cream, 1 → terracotta
    return [lerp(245, 226, k2) | 0, lerp(236, 85, k2) | 0, lerp(225, 58, k2) | 0];
  }

  function Player() {
    var canvas = $("playerCanvas");
    if (!canvas) return null;
    var ctx = canvas.getContext("2d");
    var off = document.createElement("canvas");
    var offctx = off.getContext("2d");

    var anim = null;        // loaded animation payload
    var base = [], cand = []; // decoded Uint8Array frames
    var nx = 0, ny = 0, nframes = 0;
    var vmin = 0, span = 1, dmaxByte = 1;
    var layer = "delta";
    var cur = 0;
    var playing = false;
    var raf = null, lastTs = 0, acc = 0;

    function decode() {
      base = anim.frames.baseline.map(b64ToBytes);
      cand = anim.frames.candidate.map(b64ToBytes);
      nframes = Math.min(base.length, cand.length);
      nx = anim.nx; ny = anim.ny;
      vmin = anim.scale.vmin; span = (anim.scale.vmax - anim.scale.vmin) || 1;
      off.width = nx; off.height = ny;
      // Δ scale: largest absolute baseline↔candidate byte difference across all frames.
      var m = 1;
      for (var f = 0; f < nframes; f++) {
        var a = base[f], c = cand[f];
        for (var i = 0; i < a.length; i++) {
          var d = Math.abs(c[i] - a[i]);
          if (d > m) m = d;
        }
      }
      dmaxByte = m;
    }

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = canvas.clientWidth || 480;
      var h = canvas.clientHeight || Math.round(w * (ny / nx || 1));
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function renderCell(img, idx, rgb) {
      img.data[idx] = rgb[0];
      img.data[idx + 1] = rgb[1];
      img.data[idx + 2] = rgb[2];
      img.data[idx + 3] = 255;
    }

    function drawFrame() {
      if (!anim) return;
      var img = offctx.createImageData(nx, ny);
      var a = base[cur], c = cand[cur];
      for (var j = 0; j < ny; j++) {
        // WRF row 0 is south; canvas y=0 is top → flip vertically.
        var row = (ny - 1 - j) * nx;
        for (var i = 0; i < nx; i++) {
          var src = j * nx + i;
          var rgb;
          if (layer === "delta") {
            rgb = deltaColor((c[src] - a[src]) / dmaxByte);
          } else {
            var byte = (layer === "candidate" ? c[src] : a[src]);
            rgb = absColor(byte / 255);
          }
          renderCell(img, (row + i) * 4, rgb);
        }
      }
      offctx.putImageData(img, 0, 0);

      var W = canvas.clientWidth, H = canvas.clientHeight;
      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, nx, ny, 0, 0, W, H);
      drawPolygon(W, H);
      updateReadout();
    }

    function drawPolygon(W, H) {
      // Targets can be many shapes: prefer anim.polygons (list of rings), fall back to the
      // single anim.polygon for back-compat with older completed jobs.
      var rings = anim.polygons || (anim.polygon ? [anim.polygon] : []);
      if (!rings.length) return;
      var b = anim.bounds;
      var lonSpan = (b.lon1 - b.lon0) || 1e-6;
      var latSpan = (b.lat1 - b.lat0) || 1e-6;
      ctx.strokeStyle = "rgba(42,33,28,0.85)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 4]);
      rings.forEach(function (ring) {
        if (!ring || ring.length < 3) return;
        ctx.beginPath();
        for (var i = 0; i < ring.length; i++) {
          var x = (ring[i][0] - b.lon0) / lonSpan * W;
          var y = (1 - (ring[i][1] - b.lat0) / latSpan) * H;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }

    function updateReadout() {
      var t = $("playerTime"); if (t) t.textContent = "+" + anim.times_h[cur].toFixed(1) + " h";
      var tag = $("playerLayerTag");
      if (tag) tag.textContent = layer === "delta" ? "Δ " + anim.label
        : (layer === "candidate" ? "Candidate " : "Baseline ") + anim.label;
      var s = $("playerScrub"); if (s) s.value = String(cur);
    }

    function updateLegend() {
      var lo = $("playerLegendLo"), hi = $("playerLegendHi");
      var bar = $("playerLegendBar"), unit = $("playerLegendUnit");
      if (unit) unit.textContent = anim.unit || "";
      if (layer === "delta") {
        var dmax = dmaxByte / 255 * span;
        if (lo) lo.textContent = (-dmax).toFixed(2);
        if (hi) hi.textContent = "+" + dmax.toFixed(2);
        if (bar) bar.style.background =
          "linear-gradient(90deg, rgb(58,110,165), rgb(245,236,225), rgb(226,85,58))";
      } else {
        if (lo) lo.textContent = vmin.toFixed(1);
        if (hi) hi.textContent = (vmin + span).toFixed(1);
        if (bar) bar.style.background =
          "linear-gradient(90deg, rgb(58,110,165), rgb(255,177,82), rgb(226,85,58))";
      }
    }

    function setLayer(l) {
      layer = l;
      Array.prototype.forEach.call(document.querySelectorAll(".player__layer"), function (b) {
        b.classList.toggle("is-active", b.dataset.layer === l);
      });
      updateLegend();
      drawFrame();
    }

    function setPlaying(on) {
      playing = on;
      var btn = $("playerPlay");
      if (btn) { btn.classList.toggle("is-playing", on); btn.innerHTML = on ? ICON_PAUSE : ICON_PLAY; }
      if (on && !raf) { lastTs = 0; acc = 0; raf = requestAnimationFrame(tick); }
      if (!on && raf) { cancelAnimationFrame(raf); raf = null; }
    }

    function tick(ts) {
      if (!playing) { raf = null; return; }
      if (!lastTs) lastTs = ts;
      acc += ts - lastTs; lastTs = ts;
      if (acc >= FRAME_MS) {
        acc = 0;
        cur = (cur + 1) % nframes;
        drawFrame();
      }
      raf = requestAnimationFrame(tick);
    }

    // wire controls (once)
    var playBtn = $("playerPlay");
    if (playBtn) playBtn.addEventListener("click", function () { setPlaying(!playing); });
    var scrub = $("playerScrub");
    if (scrub) scrub.addEventListener("input", function () {
      setPlaying(false); cur = clamp(parseInt(this.value, 10) || 0, 0, nframes - 1); drawFrame();
    });
    Array.prototype.forEach.call(document.querySelectorAll(".player__layer"), function (b) {
      b.addEventListener("click", function () { setLayer(this.dataset.layer); });
    });
    var rt;
    window.addEventListener("resize", function () {
      clearTimeout(rt); rt = setTimeout(function () { if (anim) { resize(); drawFrame(); } }, 150);
    });

    return {
      load: function (animation) {
        if (!animation || !animation.frames) { this.reset(); return; }
        anim = animation;
        decode();
        if (nframes < 2) { this.reset(); return; }
        cur = 0;
        $("player").hidden = false;
        var scr = $("playerScrub"); if (scr) { scr.max = String(nframes - 1); scr.value = "0"; }
        setLayer("delta");
        // Defer the first measure+draw to the next frame so the stage has a non-zero
        // clientWidth (it was just un-hidden); otherwise the canvas would draw at 0px.
        requestAnimationFrame(function () {
          resize();
          drawFrame();
          var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
          setPlaying(!reduce);
        });
      },
      reset: function () {
        setPlaying(false);
        anim = null; cur = 0;
        var p = $("player");
        if (p) p.hidden = true;
      },
      // Re-measure + repaint after a layout change (e.g. window resize).
      refresh: function () { if (anim) { resize(); drawFrame(); } },
    };
  }

  var instance = null;
  document.addEventListener("DOMContentLoaded", function () { instance = Player(); });
  window.WrfPlayer = {
    load: function (a) { if (instance) instance.load(a); },
    reset: function () { if (instance) instance.reset(); },
    refresh: function () { if (instance) instance.refresh(); },
  };
})();
