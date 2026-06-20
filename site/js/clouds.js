/* ============================================================
   Hero background — seeded clouds releasing rain + snow.
   Pure canvas, no dependencies. Cool-dominant, premium.
   ============================================================ */
(function () {
  "use strict";

  var canvas = document.getElementById("cloudCanvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var W = 0, H = 0;
  var clouds = [];
  var drops = [];      // rain streaks
  var flakes = [];     // snow dots
  var glows = [];      // soft moisture pockets behind everything

  function rand(a, b) { return a + Math.random() * (b - a); }

  function resize() {
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function makeCloud() {
    var r = rand(70, 200);
    return {
      x: rand(-0.1, 1.1) * W,
      y: rand(0.06, 0.5) * H,     // clouds sit in the upper band
      r: r,
      vx: rand(0.05, 0.22) * (r / 120),
      blobs: Math.round(rand(4, 7)),
      seed: rand(0, Math.PI * 2),
      alpha: rand(0.06, 0.16),
      seeds: rand(0.4, 1)          // how actively this cloud "rains"
    };
  }

  function makeDrop() {
    var fast = rand(5, 10);
    return {
      x: rand(0, 1) * W,
      y: rand(-0.1, 1) * H,
      len: rand(10, 22),
      v: fast,
      a: rand(0.10, 0.30)
    };
  }

  function makeFlake() {
    return {
      x: rand(0, 1) * W,
      y: rand(-0.1, 1) * H,
      r: rand(0.8, 2.2),
      v: rand(0.5, 1.4),
      drift: rand(-0.4, 0.4),
      ph: rand(0, Math.PI * 2),
      a: rand(0.25, 0.7)
    };
  }

  function makeGlow() {
    return {
      x: rand(0, 1) * W,
      y: rand(0.3, 1) * H,
      r: rand(140, 320),
      phase: rand(0, Math.PI * 2),
      speed: rand(0.0006, 0.0015),
      dry: Math.random() < 0.35     // a few dry (terracotta) pockets, most wet (green)
    };
  }

  function build() {
    clouds = []; drops = []; flakes = []; glows = [];
    var nc = Math.max(4, Math.round(W / 260));
    for (var i = 0; i < nc; i++) clouds.push(makeCloud());

    var nd = Math.max(40, Math.round(W / 7));
    for (var d = 0; d < nd; d++) drops.push(makeDrop());

    var nf = Math.max(22, Math.round(W / 18));
    for (var f = 0; f < nf; f++) flakes.push(makeFlake());

    var ng = Math.max(3, Math.round(W / 360));
    for (var g = 0; g < ng; g++) glows.push(makeGlow());
  }

  function drawGlow(h, time) {
    var pulse = 0.5 + 0.5 * Math.sin(time * h.speed + h.phase);
    var r = h.r * (0.85 + pulse * 0.25);
    var grd = ctx.createRadialGradient(h.x, h.y, 0, h.x, h.y, r);
    if (h.dry) {
      grd.addColorStop(0, "rgba(226, 85, 58, " + (0.10 + pulse * 0.08) + ")");
      grd.addColorStop(1, "rgba(226, 85, 58, 0)");
    } else {
      grd.addColorStop(0, "rgba(58, 157, 93, " + (0.10 + pulse * 0.09) + ")");
      grd.addColorStop(1, "rgba(58, 157, 93, 0)");
    }
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(h.x, h.y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawCloud(c) {
    for (var i = 0; i < c.blobs; i++) {
      var t = (i / c.blobs) * Math.PI * 2 + c.seed;
      var bx = c.x + Math.cos(t) * c.r * 0.55 * (0.5 + (i % 3) * 0.2);
      var by = c.y + Math.sin(t) * c.r * 0.24;
      var br = c.r * rand(0.42, 0.7);
      var g = ctx.createRadialGradient(bx, by, 0, bx, by, br);
      g.addColorStop(0, "rgba(255, 255, 255, " + c.alpha + ")");
      g.addColorStop(0.5, "rgba(255, 250, 243, " + c.alpha * 0.55 + ")");
      g.addColorStop(1, "rgba(255, 250, 243, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawDrop(p) {
    ctx.strokeStyle = "rgba(90, 140, 170, " + p.a * 0.7 + ")";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x - 1.2, p.y + p.len);   // slight slant
    ctx.stroke();
  }

  function drawFlake(p) {
    ctx.fillStyle = "rgba(150, 180, 205, " + p.a * 0.6 + ")";
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fill();
  }

  function frame(time) {
    ctx.clearRect(0, 0, W, H);

    // moisture glows underneath
    ctx.globalCompositeOperation = "multiply";
    for (var i = 0; i < glows.length; i++) drawGlow(glows[i], time);
    ctx.globalCompositeOperation = "source-over";

    // rain
    for (var d = 0; d < drops.length; d++) {
      var p = drops[d];
      p.y += p.v;
      p.x -= 0.6;
      if (p.y > H + p.len) { p.y = -p.len; p.x = rand(0, 1) * W; }
      drawDrop(p);
    }

    // snow
    for (var f = 0; f < flakes.length; f++) {
      var s = flakes[f];
      s.y += s.v;
      s.ph += 0.02;
      s.x += Math.sin(s.ph) * 0.4 + s.drift;
      if (s.y > H + 4) { s.y = -4; s.x = rand(0, 1) * W; }
      drawFlake(s);
    }

    // clouds on top
    ctx.globalCompositeOperation = "source-over";
    for (var c = 0; c < clouds.length; c++) {
      var cl = clouds[c];
      cl.x += cl.vx;
      if (cl.x - cl.r * 1.6 > W) { cl.x = -cl.r * 1.6; cl.y = rand(0.06, 0.5) * H; }
      drawCloud(cl);
    }
    ctx.globalCompositeOperation = "source-over";

    if (!reduceMotion) requestAnimationFrame(frame);
  }

  function init() {
    resize();
    build();
    frame(0);
  }

  var rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () { resize(); build(); if (reduceMotion) frame(0); }, 160);
  });

  init();
})();
