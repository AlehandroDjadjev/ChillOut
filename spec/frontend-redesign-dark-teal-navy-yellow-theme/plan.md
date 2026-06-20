# Frontend redesign — dark teal-navy + yellow theme

## Context
The ChillOut WRF app is now consolidated and deployed from `AlehandroDjadjev/ChillOut`
(project `30dd4f8523d0`). The current frontend is a **light** cream theme (`--bg:#fbf4ec`,
terracotta `--primary:#e2553a`, amber accent) across the landing page and the simulation
studio. The user finds the heavy white/cream "busy for the eyes" and wants a **dark**,
professional, Apple-clean restyle:

- Dark theme: very dark greenish-blue / teal-navy backgrounds + **yellow** accent + the
  transition tones between them (amber/gold → teal).
- Much less white; eye-friendly dark surfaces.
- Keep the canvas cloud animation (re-tint it for dark, don't remove it).
- Cleaner Apple-style typography/layout.
- **Remove "Precision Cloud-Seeding" copy in all 4 places** (user confirmed): hero eyebrow
  badge, `<title>`, `og:description`, footer tagline.

The whole theme is driven by CSS custom properties in `:root` (`site/css/styles.css`), so the
core change is a token rewrite, plus fixing a small set of **hardcoded light values** that
don't read from tokens (in `styles.css`, `js/clouds.js`, `simulation/sim.css`).

## Palette (new `:root` tokens in `site/css/styles.css`)
Deep teal-navy surfaces, off-white ink (not pure white), yellow primary, teal secondary, amber transition.

| Token | New value | Role |
|---|---|---|
| `--bg` | `#081317` | very dark teal-navy base |
| `--bg-2` | `#0b1c22` | secondary bg |
| `--surface` | `#0f242b` | cards/nav |
| `--surface-2` | `#16323b` | inputs/insets |
| `--line` | `rgba(220,240,238,0.10)` | hairlines |
| `--line-strong` | `rgba(220,240,238,0.20)` | stronger borders |
| `--ink` | `#e9f1ef` | primary text (soft white) |
| `--ink-soft` | `#a3bcbd` | secondary text |
| `--ink-mute` | `#6e8a8d` | muted text |
| `--primary`/`-deep`/`-glow` | `#f5c542` / `#d9a521` / `rgba(245,197,66,0.28)` | yellow CTA/highlight |
| `--accent` | `#f0a830` | amber/gold transition tone |
| `--cool`/`-deep`/`-glow` | `#34d3c0` / `#1fa896` / `rgba(52,211,192,0.25)` | teal secondary accent |
| `--success`/`-glow` | `#34d399` / `rgba(52,211,153,0.28)` | done/success |
| `--heat`/`-deep`/`-glow` | `#f0613f` / `#e2492a` / `rgba(240,97,63,0.24)` | errors only |
| `--glass-bg`/`-brd` | `rgba(15,36,43,0.55)` / `rgba(220,240,238,0.12)` | glass panels |

Note: tokens currently alias `--cool: var(--primary)` etc. — that aliasing will be **unset** so
teal is its own accent. `--heat-*` stays warm (used for error states only).

## Files & exact changes

### 1. `site/css/styles.css` (token rewrite + hardcoded-light fixes)
- Rewrite the `:root` block per the palette table above.
- `::selection` → `background: var(--primary); color: #081317;` (dark text on yellow).
- `body::before` atmosphere radial — swap `rgba(226,85,58…)/(255,177,82…)/(58,157,93…)` for
  yellow `rgba(245,197,66,…)`, amber `rgba(240,168,48,…)`, teal `rgba(52,211,192,…)` at low alpha.
- `.grain` — `mix-blend-mode: multiply` is invisible on dark → change to `overlay` (or `soft-light`),
  opacity ~`0.05`.
- `.nav.is-scrolled` background `rgba(255,250,243,0.72)` → `rgba(8,19,23,0.72)`.
- `.hero__veil` gradient `rgba(251,244,236,…)` → dark `rgba(8,19,23,…)` stops.
- `.stat__icon` / `.case__icon` hardcoded `rgba(226,85,58,…)` backgrounds → token-tinted
  (yellow/teal at low alpha).
- `.btn--primary` `color:#fff5ee` → `color:#081317` (dark on yellow).
- Apple-clean typographic polish (measured, low-risk): tighten `.hero__title` letter-spacing
  (~`-0.02em`), nudge nav/hero spacing for more negative space, ensure body line-height comfortable.

### 2. `site/js/clouds.js` (re-tint canvas for dark)
- Glow composite op `multiply` (line ~147) → `screen` / `lighter` so glows show on dark.
- `drawGlow` dry pocket `rgba(226,85,58,…)` → yellow `rgba(245,197,66,…)`; wet pocket
  `rgba(58,157,93,…)` → teal `rgba(52,211,192,…)`.
- Clouds (`rgba(255,255,255…)`/`rgba(255,250,243…)`) — keep luminous but slightly cooler/lower alpha
  so they glow softly rather than wash out the dark hero.
- Rain `rgba(90,140,170…)` → brighter `rgba(120,185,205…)`; snow `rgba(150,180,205…)` fine (maybe +alpha).

### 3. `site/simulation/sim.css` (studio dark fixes)
- `.dot--target` `rgba(226,85,58,0.30)` + heat border → teal/amber token tint.
- `.errors` / `.result__error` `rgba(194,59,34,…)` → keep error-red but dark-friendly (use `--heat` glow).
- `.player__readout` background `rgba(255,250,243,0.82)` → `rgba(8,19,23,0.72)` (dark glass).
- `.toast` shadow `rgba(42,33,28,0.25)` → deeper dark shadow.
- Pipeline/player "on-primary" text `#fff5ee` → `#081317` (dark text on yellow `.is-active`/play/layer).
- Leaflet tweaks reference tokens already — fine.

### 4. `site/index.html` (copy + meta)
- **Line 53**: remove the eyebrow badge "Precision Cloud-Seeding Platform" entirely
  (default: drop the `<p class="eyebrow">…</p>` for a cleaner hero).
- **Line 6** `<title>`, **line 203** footer `.footer__tag`, **line 11** `og:description`:
  rewrite to drop "Precision Cloud Seeding" phrasing (e.g. title → "ChillOut — Seed the clouds,
  shape the weather."; footer → a non-"precision-seeding" tagline).
- **Line 8** `theme-color` `#fbf4ec` → `#081317`.
- **Line 17** favicon SVG: bg `#fbf4ec` → `#081317`, stroke `#e2553a` → `#f5c542`.
- **Lines 20-21** bump `?v=10` → `?v=11` on styles.css + animations.css.
- Lines 217-218 (clouds.js / reveal.js) — also bump `?v=10` → `?v=11`.

### 5. `site/simulation/index.html` (meta + cache-bust)
- **Line 8** `theme-color` → `#081317`; **line 13** favicon SVG colors → dark/yellow.
- **Lines 19-20** bump `?v=10` → `?v=11` (styles.css, sim.css); bump any `player.js`/`sim.js` `?v=` too.

> animations.css keyframes use `rgba(226,85,58,0)` only as **transparent** endpoints (alpha 0,
> RGB irrelevant) and otherwise read from tokens — no change needed there.

## Verification
1. Local: open `site/index.html` and `site/simulation/index.html` in the browser (agent-browser
   or manual). Confirm: dark teal-navy theme, yellow CTAs with dark text, teal secondary accents,
   cloud animation still runs and glows are visible on dark, no light/cream flashes, grain subtle,
   nav glass dark on scroll. No "Precision Cloud-Seeding" text anywhere (grep `site/` → only the
   sim.js status string "Seeding the target column" remains, which is fine).
2. Simulation studio: map readout, panels, pipeline chips, player controls, error/toast states all
   legible on dark; yellow `.is-active` chips show dark text.
3. Deploy: commit `style: dark teal-navy + yellow redesign; drop precision-seeding copy` (bulleted
   body), push via SSH to `origin main`, then `openkbs site deploy`.
4. Hard-refresh the live CloudFront `/` and `/simulation/` (the `?v=11` bump defeats the 1-year
   immutable asset cache) and confirm the new theme is live.

## Out of scope
- No functional/API/worker changes. No layout restructuring beyond spacing/typography polish.
- ML↔WRF deployment unification (still deferred).
