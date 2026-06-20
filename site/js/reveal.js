/* ============================================================
   Scroll reveals, nav scroll state, and stat count-ups.
   ============================================================ */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- nav solidify on scroll -----------------------------------
  var nav = document.getElementById("nav");
  function onScroll() {
    if (!nav) return;
    nav.classList.toggle("is-scrolled", window.scrollY > 24);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---- apply per-element reveal delay ---------------------------
  var revealEls = Array.prototype.slice.call(document.querySelectorAll("[data-reveal]"));
  revealEls.forEach(function (el) {
    var d = parseInt(el.getAttribute("data-delay") || "0", 10);
    el.style.setProperty("--reveal-delay", d);
  });

  // ---- count-up for stats --------------------------------------
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var prefix = el.getAttribute("data-prefix") || "";
    var suffix = el.getAttribute("data-suffix") || "";
    var decimals = (String(target).split(".")[1] || "").length;
    if (reduceMotion) {
      el.textContent = prefix + target.toFixed(decimals) + suffix;
      return;
    }
    var start = performance.now();
    var dur = 1400;
    function step(now) {
      var p = Math.min((now - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---- intersection observer -----------------------------------
  if (!("IntersectionObserver" in window) || reduceMotion) {
    revealEls.forEach(function (el) { el.classList.add("is-visible"); });
    document.querySelectorAll("[data-count]").forEach(countUp);
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var el = entry.target;
      el.classList.add("is-visible");
      el.querySelectorAll && el.querySelectorAll("[data-count]").forEach(function (c) {
        if (!c.__counted) { c.__counted = true; countUp(c); }
      });
      io.unobserve(el);
    });
  }, { threshold: 0.18, rootMargin: "0px 0px -8% 0px" });

  revealEls.forEach(function (el) { io.observe(el); });

  // observe stat numbers that aren't inside a [data-reveal] wrapper too
  document.querySelectorAll("[data-count]").forEach(function (c) {
    io.observe(c.closest("[data-reveal]") || c);
  });
})();
