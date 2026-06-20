# ChillOut — Climate Intelligence Landing Page

## Context

ChillOut is a climate-tech hackathon startup. We need a single, high-impact marketing landing page that instantly communicates one idea: *"We help identify where cooling effects would matter most."* The platform's concept is temperature optimization via cloud placement — but the page must sell the vision visually, **not** explain ML models, RL, or technical architecture.

The current `site/` holds only a placeholder `index.html` (purple gradient starter). We replace it with a premium, dark, animated landing page. Everything is **client-side with mock data** — no backend, no Postgres, no API calls. The existing `functions/api` is left untouched.

## Aesthetic direction

- **Tone:** premium, refined, editorial climate-tech. Dark mode. Subtle glassmorphism, smooth motion, generous spacing.
- **Color (CSS variables):** deep near-black base (`#070a0c` / layered charcoals), a single sharp **cool accent** (ice cyan/teal) as the dominant brand color, with a restrained **warm "heat" accent** (amber→coral) used *only* for heatmap/data moments. The heat→cool contrast is the brand story, which justifies the one gradient we use. **No purple→pink, no gradient text on every headline, no blue→cyan-on-white cliché.**
- **Typography:** distinctive serif display (Instrument Serif / Fraunces via Google Fonts) + clean grotesque body (e.g. Schibsted Grotesk / Geist — a non-Inter grotesque). Loaded via `<link>` preconnect.
- **Icons:** inline SVG (Lucide-style line icons). No emoji as UI.
- **Motion:** one orchestrated load with staggered reveals; scroll-triggered fade/slide via `IntersectionObserver`; animated cloud-particle + heatmap canvas in the hero; interactive canvas region for solution/demo.

## File structure (under `./site/`)

Split for craft (not one giant inline blob):

```
site/
  index.html        ← semantic sections, SEO meta, font links
  css/
    styles.css      ← design tokens (CSS vars), layout, glass, components, responsive
    animations.css  ← keyframes + reveal/transition classes
  js/
    clouds.js       ← hero canvas: drifting cloud particles + subtle heatmap blobs
    map.js          ← shared stylized region canvas renderer (cooling zones, heat gradient)
    demo.js         ← interactive simulator: region/slider/target → mock recommendation
    reveal.js       ← IntersectionObserver scroll reveals + nav scroll state
  assets/
    (favicon / og image — generated via `openkbs image` if time permits)
```

Plain HTML (best for SEO + simplicity per skill guidance). No build step, no framework.

## Sections (single page, anchor nav)

1. **Sticky nav** — ChillOut wordmark, anchor links (Problem / Solution / Demo / Use Cases / Vision), subtle "Try Demo" button. Glass background that solidifies on scroll.
2. **Hero** — headline *"Cooling the planet, one cloud at a time."*, subhead, two CTAs (Try Demo → scrolls to demo; Learn More → scrolls to problem). Full-bleed `clouds.js` canvas background (drifting clouds + faint heatmap shimmer). Staggered text reveal on load.
3. **Problem** — headline *"Heat is becoming one of the world's biggest challenges."* + 4 stat cards (Rising heat waves, Increasing wildfire risk, Agricultural stress, Urban heat islands) with mock numbers, icons, and count-up-on-reveal. Infographic-style asymmetric grid (not 4 identical boxes — vary sizes/emphasis).
4. **Solution** — headline *"Find the coolest possible outcome."* + description, beside an interactive stylized region canvas (`map.js`) showing highlighted cooling zones over a heat field.
5. **Interactive Demo** — *"Explore a cooling scenario."* Controls: region selector (dropdown), temperature slider, cooling-target selector. On **Simulate** → animates the region canvas and reveals: recommended zone, expected temperature reduction (°C), and a confidence indicator (animated ring/bar). All computed from mock logic in `demo.js`.
6. **Use Cases** — 3 cards: Wildfire Prevention, Agriculture, Climate Research (copy from brief), line-icon + hover lift.
7. **Vision** — headline *"A future of intelligent climate adaptation."* + short data-driven-decision-support copy, calm full-width band.
8. **Footer** — ChillOut brand + tagline *"Temperature Intelligence for a Warmer World."*, minimal nav links, copyright. No fake testimonials/logos.

## Mock interactive demo logic (`demo.js`)

- Hardcoded list of ~5 regions, each with a base temp and a few candidate zones (lat/long-ish grid coords + cooling potential).
- On simulate: pick the highest-cooling zone given the slider/target, compute expected reduction (deterministic mock formula + slight variance), derive a confidence % from inputs. Animate the canvas to highlight the chosen zone; animate numbers counting up.
- Purely front-end; no network.

## Files to create / modify

- **Replace** `site/index.html` (full rewrite).
- **Create** `site/css/styles.css`, `site/css/animations.css`.
- **Create** `site/js/clouds.js`, `site/js/map.js`, `site/js/demo.js`, `site/js/reveal.js`.
- Optionally generate `site/assets/og.png` + favicon via `openkbs image`.
- No changes to `openkbs.json` or `functions/`.

## Verification

1. Serve locally (`python3 -m http.server` in `site/`) and open in a browser.
2. Use `agent-browser` (load its skill first) to: load the page, screenshot hero, scroll through every section confirming reveals fire, run the demo (select region, move slider, pick target, click Simulate) and confirm the recommendation + confidence render and the canvas updates.
3. Check responsive layout at mobile width (≤480px) and tablet — no overflow, nav collapses gracefully.
4. Confirm no console errors and all 4 JS files load.
5. Confirm constraints honored: no ML/RL/architecture explanations anywhere in copy; no cliché purple→pink gradients; no emoji UI controls.
6. (Build-only) Report results and a preview screenshot; do **not** deploy unless asked.
