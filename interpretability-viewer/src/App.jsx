import { createContext, Fragment, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AtlasMark } from './AtlasMark';
import { useAtlasFavicon } from './atlas-mark';
import './App.css';
import { FACT_KEYS, lookupWiki } from './wiki';

const BASE_URL = import.meta.env.BASE_URL ?? '/';
const USE_LOCAL_ASSETS = import.meta.env.VITE_ASSET_SOURCE === 'local';

// Overlay display preference (heatmap composited over the original), shared by all views.
const OverlayContext = createContext({ enabled: false, opacity: 0.5 });
const useOverlay = () => useContext(OverlayContext);

// Aiming the context card at a subject is available anywhere a name is shown.
// There is one reference surface on the page and this is how anything else
// points at it.
const WikiContext = createContext({ open: () => {} });
const useWiki = () => useContext(WikiContext);

const VS_KEYS = ['mode', 'model', 'dataset', 'classId', 'imageId', 'methods', 'models'];

function readStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return Object.fromEntries(VS_KEYS.map((k) => [k, params.get(k)]).filter(([, v]) => v != null));
}

function writeStateToUrl(vs) {
  const params = new URLSearchParams();
  for (const k of VS_KEYS) if (vs[k] != null) params.set(k, vs[k]);
  const query = params.toString();
  window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname);
}

function initialTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return 'light';
}

// Apply once at module load — before first render, no flash, no effect.
document.documentElement.dataset.theme = initialTheme();

function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme);
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    setTheme(next);
  };
  return (
    <button
      type="button" className="theme-toggle" onClick={toggle}
      title="Toggle theme" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >{theme === 'dark' ? '☀' : '☾'}</button>
  );
}

const METHOD_CATEGORIES = {
  gradient: {
    label: 'Gradient',
    methods: [
      'Saliency', 'InputXGradient',
      'IntegratedGradients', 'LayerIntegratedGradients',
      'DeepLift', 'DeepLiftShap', 'GradientShap',
      'GuidedBackprop', 'Deconvolution',
      'LayerGradCam', 'GuidedGradCam',
    ],
  },
  perturbation: {
    label: 'Perturbation',
    methods: [
      'CB-RISE', 'RISE',
      'Occlusion',
      'Lime',
      'KernelShap', 'ShapleyValueSampling',
      'FeatureAblation', 'FeaturePermutation',
    ],
  },
};

/* Torchvision names carry their family as a prefix, so the rail can group
   twelve models the way it groups methods: one checkbox for all the ResNets.
   Order here is the order on screen — architectural lineage, not alphabet. */
const MODEL_FAMILIES = [
  { key: 'alexnet', label: 'AlexNet', prefixes: ['alexnet'] },
  { key: 'vgg', label: 'VGG', prefixes: ['vgg'] },
  { key: 'inception', label: 'Inception', prefixes: ['inception', 'googlenet'] },
  { key: 'resnet', label: 'ResNet', prefixes: ['resnet', 'resnext', 'wide_resnet'] },
  { key: 'densenet', label: 'DenseNet', prefixes: ['densenet'] },
  { key: 'mobilenet', label: 'MobileNet', prefixes: ['mobilenet'] },
  { key: 'efficientnet', label: 'EfficientNet', prefixes: ['efficientnet'] },
  { key: 'regnet', label: 'RegNet', prefixes: ['regnet'] },
  { key: 'convnext', label: 'ConvNeXt', prefixes: ['convnext'] },
  { key: 'vit', label: 'ViT', prefixes: ['vit_', 'vit'] },
  { key: 'swin', label: 'Swin', prefixes: ['swin'] },
];

function categorizeModel(name) {
  const norm = normalize(name);
  return MODEL_FAMILIES.find((f) => f.prefixes.some((p) => norm.startsWith(p)))?.key ?? 'other';
}

// Within a family, resnet18 before resnet101: the depth suffix is a number,
// and a plain string sort would read it as text and put 101 first.
function compareModelNames(a, b) {
  const [an, bn] = [a, b].map((v) => Number(String(v).match(/\d+/)?.[0] ?? NaN));
  if (Number.isFinite(an) && Number.isFinite(bn) && an !== bn) return an - bn;
  return String(a).localeCompare(String(b));
}

// Groups for the rail, in family order, skipping families this dataset has no
// run for. Same shape FacetFilter takes for methods, so one component serves both.
function groupModels(models) {
  const grouped = {};
  for (const m of models) (grouped[categorizeModel(m)] ??= []).push(m);
  return [...MODEL_FAMILIES, { key: 'other', label: 'Other' }]
    .filter((f) => grouped[f.key]?.length)
    .map((f) => ({ key: f.key, label: f.label, methods: grouped[f.key].sort(compareModelNames) }));
}

function normalize(v) {
  return String(v ?? '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').trim();
}

function compareMixedIds(a, b) {
  const an = Number(a), bn = Number(b);
  return Number.isFinite(an) && Number.isFinite(bn) ? an - bn : String(a).localeCompare(String(b));
}

function categorizeMethod(name) {
  const norm = normalize(name);
  for (const [cat, { methods }] of Object.entries(METHOD_CATEGORIES)) {
    if (methods.some((m) => norm.startsWith(normalize(m)))) return cat;
  }
  return 'other';
}

function compareMethods(category, a, b) {
  const order = METHOD_CATEGORIES[category]?.methods ?? [];
  const rank = (name) => order.findIndex((method) => normalize(name).startsWith(normalize(method)));
  const aRank = rank(a), bRank = rank(b);
  if (aRank !== bRank) return (aRank < 0 ? order.length : aRank) - (bRank < 0 ? order.length : bRank);
  return String(a).localeCompare(String(b));
}

// The checked methods, in the order the sidebar lists them, minus the ones
// this particular image has no output for.
function resolveMethodEntries(methods, outputs = {}) {
  return (methods ?? []).filter((m) => outputs?.[m]).map((m) => [m, outputs[m]]);
}

// Same list, but for the whole row of models at once: a method earns a row in
// the comparison matrix if any model on screen produced it. The models that
// did not get a hole in that row rather than a shorter column, so what sits
// side by side is always the same method.
function compareMethodRows(methods, cells) {
  return (methods ?? []).filter((m) => cells.some((c) => c.record?.outputs?.[m]));
}

// Heavy image binaries live on Hugging Face datasets; JSON metadata stays local (in git).
// Set VITE_ASSET_SOURCE=local to serve everything from public/.
const HF_DATASET = 'https://huggingface.co/datasets/Matgc04';
// imagenet-pico images live in a private HF dataset. Its small JSON metadata stays
// public and local, so only files below val/ go through the authenticated Worker.
const WORKER_ORIGIN = import.meta.env.VITE_WORKER_URL;
const HF_ROUTES = USE_LOCAL_ASSETS ? [] : [
  { test: /^imagenet-pico-ai\/val\//, base: `${HF_DATASET}/neuralatlas-imagenet-pico-ai/resolve/main/`, strip: /^imagenet-pico-ai\// },
  // Compatibility for runs created before manifest schema v2 added base_url.
  { test: /^outputs\/images\//, base: `${HF_DATASET}/neuralatlas-attributions/resolve/main/`, strip: /^outputs\// },
  ...(WORKER_ORIGIN ? [{ test: /^imagenet-pico\/val\//, base: `${WORKER_ORIGIN}/hf/`, strip: /^imagenet-pico\// }] : []),
];

function resolveAssetUrl(path) {
  if (!path) return null;
  const value = String(path);
  if (/^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('data:')) return value;
  const rel = value.replace(/^\/+/, '');
  for (const route of HF_ROUTES) {
    if (route.test.test(rel)) return `${route.base}${rel.replace(route.strip, '')}`;
  }
  return `${BASE_URL}${rel}`;
}

async function fetchJson(path, { retryCount = 0, retryDelayMs = 400, signal, ...options } = {}) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const response = await fetch(resolveAssetUrl(path), { ...options, signal });
      if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError' || attempt >= retryCount) throw error;
      const jitter = 0.75 + Math.random() * 0.5;
      await new Promise((resolve, reject) => {
        const onAbort = () => {
          clearTimeout(timeoutId);
          reject(signal.reason);
        };
        const timeoutId = setTimeout(() => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        }, retryDelayMs * (2 ** attempt) * jitter);
        if (signal?.aborted) onAbort();
        else signal?.addEventListener('abort', onAbort, { once: true });
      });
    }
  }
}

function resolveRunOutputUrl(baseUrl, classId, outputPath) {
  if (!outputPath || !baseUrl || USE_LOCAL_ASSETS) return outputPath;
  const value = String(outputPath);
  if (/^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('data:')) return value;
  const filename = value.split('/').pop();
  return `${String(baseUrl).replace(/\/+$/, '')}/${encodeURIComponent(classId)}/${encodeURIComponent(filename)}`;
}

function filenameFromUrl(url) {
  if (!url) return null;
  return decodeURIComponent(String(url).split(/[?#]/)[0].split('/').pop());
}

function buildAtlasData(manifest, runPayloads) {
  const models = {};
  const records = [];
  for (const model of manifest?.models ?? []) {
    // Keep metadata for empty runs so their model and dataset remain selectable.
    models[model] = {};
    for (const dataset of manifest?.datasets_by_model?.[model] ?? []) {
      const run = runPayloads[`${model}::${dataset}`];
      models[model][dataset] = run?.summary?.metrics ?? null;
      const images = new Map();
      for (const image of run?.images?.images ?? []) {
        const classId = String(image.class_id), imageId = String(image.image_id);
        const originalUrl = image.original_url ?? null;
        images.set(JSON.stringify([classId, imageId]), {
          model, dataset, classId, imageId, originalUrl,
          filename: filenameFromUrl(originalUrl),
          outputs: Object.fromEntries(Object.entries(image.outputs ?? {}).map(([method, path]) => [
            method, resolveRunOutputUrl(run.baseUrl, classId, path),
          ])),
          prediction: image.prediction ?? null,
          interpretabilityMetrics: image.interpretability_metrics ?? {},
        });
      }
      records.push(...images.values());
    }
  }
  records.sort((a, b) =>
    a.dataset.localeCompare(b.dataset) || compareMixedIds(a.classId, b.classId) ||
    compareMixedIds(a.imageId, b.imageId) || a.model.localeCompare(b.model)
  );
  return { models, records };
}

// `models` is an allow-list: the columns the rail left checked. Rows are built
// from those records only, so hiding a model that had an image the others lack
// drops the row with it instead of leaving a line of empty cells.
function getClassCompareMatrix(records, { dataset, classId, models: allowed }) {
  if (!dataset || !classId) return { models: [], rows: [] };

  const allow = allowed ? new Set(allowed) : null;
  const scoped = records.filter((r) =>
    r.dataset === dataset && r.classId === classId && (!allow || allow.has(r.model))
  );
  const models = [...new Set(scoped.map((r) => r.model))].sort(compareModelNames);

  // Image IDs are run-local and can point at different source files between
  // models. Use the source filename as the comparison identity so a shared
  // original is only shown when the models refer to the same dataset image.
  // Without source metadata, keep models separate: the index proves no identity.
  const sourceKeyFor = (record) => JSON.stringify([
    record.classId,
    record.filename || `${record.model}:image:${record.imageId}`,
  ]);
  const bySource = new Map();
  for (const r of scoped) {
    const sourceKey = sourceKeyFor(r);
    if (!bySource.has(sourceKey)) bySource.set(sourceKey, new Map());
    bySource.get(sourceKey).set(r.model, r);
  }

  const rows = [...bySource.entries()].map(([sourceKey, rowMap]) => {
    const exemplar = rowMap.values().next().value;
    return {
      sourceKey,
      imageId: exemplar?.imageId ?? null,
      filename: exemplar?.filename ?? null,
      classId,
      classLabel: exemplar?.classLabel ?? classId,
      cells: models.map((model) => ({ model, record: rowMap.get(model) ?? null })),
    };
  }).sort((a, b) =>
    compareMixedIds(a.imageId ?? '', b.imageId ?? '') ||
    String(a.filename ?? '').localeCompare(String(b.filename ?? ''))
  );

  return { models, rows };
}

function formatMetricPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

/* ── Reusable UI Components ─────────────────────────────────── */

const JET_LUT = (() => {
  function pw(t, stops) {
    for (let i = 0; i < stops.length - 1; i++) {
      const [x0, y0] = stops[i], [x1, y1] = stops[i + 1];
      if (t <= x1) return y0 + (y1 - y0) * (t - x0) / (x1 - x0);
    }
    return stops.at(-1)[1];
  }
  return Array.from({ length: 256 }, (_, i) => {
    const t = i / 255;
    return [
      pw(t, [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]]),
      pw(t, [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]]),
      pw(t, [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]]),
    ].map(v => Math.round(Math.max(0, Math.min(1, v)) * 255));
  });
})();

function applyJet(img, canvas) {
  if (!canvas || !img.naturalWidth || !img.naturalHeight) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    const gray = px[i];
    const [r, g, b] = JET_LUT[gray];
    px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = 255;
  }
  ctx.putImageData(d, 0, 0);
}

function JetCanvas({ src, className, alt, opacity, onPainted }) {
  const canvasRef = useRef(null);
  const paintedRef = useRef(onPainted);
  paintedRef.current = onPainted;
  useEffect(() => {
    if (!src) return undefined;
    const img = new Image();
    let cancelled = false;

    // The canvas is reused across src changes, so its old pixels survive until
    // the next image loads — and forever if that one 404s. Wipe it first: a
    // blank tile is honest, a stale heatmap from the previous image is not.
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    // Needed so the canvas can read pixels for the jet colormap when the heatmap is
    // served cross-origin (Hugging Face). HF reflects the request Origin in its CORS
    // header, so an anonymous request is allowed and the canvas stays untainted.
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => {
      if (!cancelled) {
        applyJet(img, canvasRef.current);
        paintedRef.current?.(src);
      }
    };
    img.src = src;

    return () => {
      cancelled = true;
    };
  }, [src]);

  if (!src) return null;
  return (
    <canvas
      ref={canvasRef} className={className} role="img" aria-label={alt}
      style={opacity == null ? undefined : { opacity }}
    />
  );
}

// Heatmap, optionally composited over the model-view crop of the original.
function Attribution({ src, originalSrc, alt, className }) {
  const { enabled, opacity } = useOverlay();
  // The heatmap is decoded and repainted asynchronously, and the base <img>
  // keeps showing the *previous* photo until the new one decodes. Left alone
  // that plays as a slideshow — old original, new original, then the map. So
  // the stack tracks which src it has actually painted and stays invisible
  // until that is the src it was asked for: no bare photo posing as a map.
  const [paintedSrc, setPaintedSrc] = useState(null);
  const [paintedBase, setPaintedBase] = useState(null);
  if (!originalSrc) return <JetCanvas className={className} src={src} alt={alt} />;
  const ready = paintedSrc === src && paintedBase === originalSrc;
  return (
    <div className={`${className} overlay-stack${ready ? ' is-ready' : ''}`}>
      <img
        className="overlay-stack__base" src={originalSrc} alt="" loading="lazy"
        // A base that 404s still counts as settled: the map is the point, and
        // it reads fine over the bare bed.
        onLoad={() => setPaintedBase(originalSrc)}
        onError={() => setPaintedBase(originalSrc)}
      />
      <JetCanvas
        className="overlay-stack__heat" src={src} alt={alt}
        opacity={enabled ? opacity : 1}
        onPainted={setPaintedSrc}
      />
    </div>
  );
}

// Original shown cropped to the model's view (Resize 256 -> CenterCrop 224);
// click toggles to the full untouched image inside the same box.
function OriginalImage({ src, alt, className }) {
  const [expanded, setExpanded] = useState(false);
  if (!src) return null;
  // The zoom is a transform, which paints outside the element's own box —
  // it needs a frame to be clipped by, the same way .overlay-stack does.
  return (
    <div
      className={`${className} original-frame`}
      role="button" tabIndex={0} aria-pressed={expanded}
      onClick={() => setExpanded((v) => !v)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded((v) => !v); }
      }}
      title={expanded ? 'Full image — click to crop to model view' : 'Cropped to model view (224) — click for full image'}
    >
      <img className={`original-crop${expanded ? ' is-expanded' : ''}`} src={src} alt={alt} loading="lazy" />
    </div>
  );
}

function formatMetricBadgeValue(value) {
  return Math.abs(value) >= 1000
    ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumSignificantDigits: 3 }).format(value)
    : value.toFixed(2);
}

// Display order only; each metric is described once, in the wiki.
function metricTitle(name) {
  const entry = lookupWiki('metric', name);
  return entry ? `${entry.title} — ${entry.summary}` : name;
}

const METRIC_ORDER = ['lif', 'morph', 'segment', 'fidelity', 'fidelity_superpixel'];

function MetricBadges({ metrics }) {
  const items = METRIC_ORDER
    .map((name) => ({ name, title: metricTitle(name), rawValue: metrics?.[name] }))
    .filter(({ rawValue }) => rawValue != null && rawValue !== '' && Number.isFinite(Number(rawValue)))
    .map(({ name, title, rawValue }) => ({ name, title, value: Number(rawValue) }));

  if (!items.length) return null;
  return (
    <dl className="metric-badges" aria-label="Interpretability metrics">
      {items.map(({ name, title, value }) => (
        <div key={name} className={`metric-badge${name === 'fidelity_superpixel' ? ' metric-badge--wide' : ''}`} title={title}>
          <dt>{name}</dt>
          <dd>{formatMetricBadgeValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function MiniImage({ caption, captionActions, src, alt, missingText = 'Not available', variant = 'attribution', originalSrc, metrics, wikiKind }) {
  const resolvedSrc = resolveAssetUrl(src);
  // Name first, then the picture. In a wall of twenty heatmaps you scan for a
  // method by name and then look at what it produced, so the label leads.
  return (
    <figure className="mini-image">
      {/* No caption when the name lives on the axis instead of on the cell —
          the comparison matrix labels its method rows once, in the gutter. */}
      {(caption || captionActions) && (
        <figcaption>
          {caption && <span className="mini-image__name">{caption}</span>}
          {caption && wikiKind && <InfoDot kind={wikiKind} id={caption} label={caption} />}
          {captionActions}
        </figcaption>
      )}
      {!resolvedSrc
        ? <div className="mini-image__missing">{missingText}</div>
        : variant === 'original'
          ? <OriginalImage className="mini-image__asset mini-image__asset--original" src={resolvedSrc} alt={alt} />
          : <Attribution className="mini-image__asset mini-image__asset--attribution" src={resolvedSrc} originalSrc={resolveAssetUrl(originalSrc)} alt={alt} />}
      {resolvedSrc && variant !== 'original' && <MetricBadges metrics={metrics} />}
    </figure>
  );
}

// A switch, not a button: overlay is a state the maps are in, and a switch
// shows which state that is without having to read the label for a tense.
function OverlayControl({ enabled, opacity, onToggle, onOpacity }) {
  return (
    <div className="overlay-control">
      <button
        type="button" className={`switch${enabled ? ' is-on' : ''}`}
        onClick={onToggle} role="switch" aria-checked={enabled}
      >
        <span className="switch__track"><span className="switch__knob" /></span>
        <span className="switch__label">Overlay heatmap</span>
      </button>
      {enabled && (
        <label className="overlay-control__opacity">
          Opacity
          <span aria-hidden="true">0</span>
          <input
            type="range" min="0" max="1" step="0.05" value={opacity}
            onChange={(e) => onOpacity(Number(e.target.value))}
          />
          <span aria-hidden="true">1</span>
        </label>
      )}
    </div>
  );
}

// Horizontal, and it rides with the images instead of sitting in the sidebar:
// the scale is a property of the maps being read, not of the selection.
function ColorbarLegend() {
  return (
    <div className="colorbar-legend">
      <span className="colorbar-legend__title">Attribution</span>
      <span className="colorbar-legend__tick" aria-hidden="true">0</span>
      <span className="colorbar-legend__bar" role="img" aria-label="Attribution color scale from 0 to 1" />
      <span className="colorbar-legend__tick" aria-hidden="true">1</span>
      <span className="colorbar-legend__note">normalized per image</span>
    </div>
  );
}

// Render settings sit above the maps they change, sticky under the top bar.
function RenderBar({ overlay, opacity, onToggle, onOpacity }) {
  const ref = useRef(null);

  // The compare header sticks flush under this bar, so its offset *is* this
  // bar's height. Published from a measurement rather than declared as a
  // constant: the bar is content-sized and wraps at narrow widths, and a
  // number guessed in the stylesheet leaves a transparent seam that the maps
  // scroll through.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const publish = () => {
      const { height } = el.getBoundingClientRect();
      document.documentElement.style.setProperty('--renderbar-h', `${height}px`);
    };
    publish();
    const ro = new ResizeObserver(publish);
    ro.observe(el);
    return () => {
      ro.disconnect();
      document.documentElement.style.removeProperty('--renderbar-h');
    };
  }, []);

  return (
    <div className="render-bar" ref={ref}>
      <div className="render-bar__controls">
        <OverlayControl enabled={overlay} opacity={opacity} onToggle={onToggle} onOpacity={onOpacity} />
        <ColorbarLegend />
      </div>
    </div>
  );
}

/* ── Reference layer ────────────────────────────────────────────
   Every model, dataset, method and metric has a glossary entry. The dot
   shows the one-liner on hover, and on click aims the context card at the
   top of the page at that entry. A finger has no hover, so the tap that would
   have hovered is the first one: it shows the tip, and the next one opens the
   card. What broke that on a phone was the dismissal, not the two steps —
   listening for the tap that closes the tip from inside the tap that opened
   it. It is armed a frame later now. */

function InfoDot({ kind, id, label }) {
  const entry = lookupWiki(kind, id);
  const { open } = useWiki();
  const rootRef = useRef(null);
  const anchorRef = useRef(null);
  const tipRef = useRef(null);
  const pointer = useRef(null);
  const [anchor, setAnchor] = useState(null);
  const [pos, setPos] = useState(null);
  // Which input is driving, read off the event rather than off a media query:
  // a laptop with a touchscreen is both, and what counts is the one in hand.
  const [byTouch, setByTouch] = useState(false);

  const show = () => {
    const r = anchorRef.current?.getBoundingClientRect();
    if (r) setAnchor(r);
  };
  const hide = () => { setAnchor(null); setPos(null); setByTouch(false); };

  // Placed after the tip exists, because deciding whether it fits below the dot
  // needs its measured height. useLayoutEffect, so the first painted frame is
  // already in the right place.
  useLayoutEffect(() => {
    if (!anchor || !tipRef.current) return;
    const { width, height } = tipRef.current.getBoundingClientRect();
    const below = anchor.bottom + 8;
    setPos({
      left: Math.max(8, Math.min(anchor.left, window.innerWidth - width - 8)),
      // Flip above the dot rather than run off the bottom of the window.
      top: below + height > window.innerHeight - 8
        ? Math.max(8, anchor.top - 8 - height)
        : below,
    });
  }, [anchor]);

  // A tip a finger opened has no hover to end it: the next touch anywhere else
  // closes it, and so does the page moving under it — the position is measured
  // once, in viewport coordinates, so a scroll would strand it mid-air. The
  // away-listener waits a frame, or the tap that opened the tip closes it.
  useEffect(() => {
    if (!anchor || !byTouch) return;
    const away = (e) => { if (!rootRef.current?.contains(e.target)) hide(); };
    const arm = requestAnimationFrame(() => document.addEventListener('pointerdown', away));
    window.addEventListener('scroll', hide, { passive: true, capture: true });
    return () => {
      cancelAnimationFrame(arm);
      document.removeEventListener('pointerdown', away);
      window.removeEventListener('scroll', hide, { capture: true });
    };
  }, [anchor, byTouch]);

  if (!entry) return null;

  // First tap reads, second tap acts. A pointerdown that never reported its
  // type falls back to asking whether this screen can hover at all — a phone
  // that only fires click must not be treated as a mouse.
  //
  // The gate is byTouch and not "is the tip up", because tapping the button
  // focuses it, onFocus shows the tip, and the click that follows would find
  // its own tooltip already standing and read as the second tap.
  const act = (e) => {
    e.stopPropagation();
    const reported = pointer.current;
    pointer.current = null;
    const finger = reported ? reported !== 'mouse' : !window.matchMedia?.('(hover: hover)').matches;
    if (finger && !byTouch) {
      setByTouch(true);
      show();
      return;
    }
    hide();
    open(kind, entry.id);
  };

  return (
    <span
      ref={rootRef} className="info-dot"
      /* Filtered by pointer type rather than left as onMouseEnter: touch fires
         a compatibility mouseenter before the click, which would open the tip
         and make the first tap look like the second. */
      onPointerEnter={(e) => { if (e.pointerType === 'mouse') show(); }}
      onPointerLeave={(e) => { if (e.pointerType === 'mouse') hide(); }}
    >
      <button
        ref={anchorRef} type="button" className="info-dot__btn"
        aria-label={`About ${label ?? entry.title}`}
        onFocus={show} onBlur={hide}
        onPointerDown={(e) => { pointer.current = e.pointerType; }}
        onClick={act}
      >i</button>
      {/* Portalled to the body. The tip is positioned in viewport coordinates,
          and any transformed ancestor — every image card carries one from its
          entry animation — would otherwise become the origin those coordinates
          resolve against and throw the tip across the page. */}
      {anchor && createPortal(
        <span
          ref={tipRef} className="info-tip" role="tooltip"
          style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
        >
          <span className="info-tip__title">{entry.title}</span>
          {entry.tags?.length > 0 && <span className="info-tip__tags">{entry.tags.join(' · ')}</span>}
          <span className="info-tip__summary">{entry.summary}</span>
          <span className="info-tip__cta">{byTouch ? 'Tap again to open it at the top' : 'Click to open it at the top'}</span>
        </span>,
        document.body,
      )}
    </span>
  );
}

const factOf = (entry, label) => entry.facts?.find(([k]) => k === label)?.[1] ?? '—';

// Segmentation variants (for example "Lime (SLIC)" and "Lime (KMeans)")
// share one glossary entry. Keep the real method id for selection, but show
// only one chip in the reference card.
function uniqueWikiMethods(methodOptions, activeMethod) {
  const grouped = new Map();
  for (const methodId of methodOptions) {
    const entry = lookupWiki('method', methodId);
    const key = entry?.id?.toLowerCase() ?? String(methodId).toLowerCase();
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        id: methodId,
        label: entry?.title ?? methodId,
      });
    } else if (methodId === activeMethod) {
      // Keep the active segmentation as the target if it is not the first
      // variant in the selected list.
      existing.id = methodId;
    }
  }
  return [...grouped.values()];
}

// Models have one entry each, so the chip list is just the documented ones.
function wikiEntryOptions(kind, ids) {
  return ids
    .map((id) => [id, lookupWiki(kind, id)])
    .filter(([, entry]) => entry)
    .map(([id, entry]) => ({ id, label: entry.title ?? id }));
}

/* The subject of the screen, stated above the maps: what dataset, what model,
   what method. It replaces the info dots that used to hang off the dataset and
   model crumbs — the answer is now on the page instead of one hover away, and
   the dots stay only where there is no room for a card (the method captions).
   Full-bleed across the content column, like the render strip: it captions
   everything under it, so it is as wide as what it captions. "Read more" grows
   the card downwards rather than opening the side panel — the maps below stay
   in place, they only move down. */

const CONTEXT_TABS = [
  { kind: 'dataset', label: 'Dataset' },
  { kind: 'model', label: 'Model' },
  { kind: 'method', label: 'Method' },
];

function ContextCard({
  dataset, model, method, methodOptions = [], onMethod,
  modelOptions = [], onModel,
  tab, onTab, expanded, onExpand,
}) {
  const ids = { dataset, model, method };
  // Methods and models ask the same question of the card — "which of the ones
  // on screen am I reading?" — so both answer it with the same chip row.
  const pickers = {
    method: { options: uniqueWikiMethods(methodOptions, method), active: method, onPick: onMethod },
    model: { options: wikiEntryOptions('model', modelOptions), active: model, onPick: onModel },
  };

  // A tab exists only when its subject is documented: a dataset with no entry
  // would open an empty card.
  const tabs = CONTEXT_TABS.filter((t) => lookupWiki(t.kind, ids[t.kind]));
  if (!tabs.length) return null;

  const active = tabs.find((t) => t.kind === tab) ?? tabs[0];
  const entry = lookupWiki(active.kind, ids[active.kind]);
  const picker = pickers[active.kind];

  return (
    <section className="context-card" aria-label="Current selection reference">
      <nav className="context-card__tabs" role="tablist" aria-label="Reference subject">
        {tabs.map((t) => (
          <button
            key={t.kind} type="button" role="tab"
            aria-selected={t.kind === active.kind}
            className={`context-card__tab${t.kind === active.kind ? ' is-on' : ''}`}
            onClick={() => onTab(t.kind)}
          >{t.label}</button>
        ))}
      </nav>

      <div className="context-card__body">
        {/* Which one of them the card is reading. Only when more than one is
            checked — with a single one the tab label already says it. */}
        {picker?.onPick && picker.options.length > 1 && (
          <div className="context-card__picker">
            {picker.options.map(({ id, label }) => (
              <button
                key={id} type="button"
                className={`wiki-chip${id === picker.active ? ' is-on' : ''}`}
                onClick={() => picker.onPick(id)}
              >{label}</button>
            ))}
          </div>
        )}

        {/* Identity on the left, the sentence in the middle, the control on
            the right — one line across the full width instead of a stack in a
            narrow column. */}
        <div className="context-card__lede">
          <div className="context-card__ident">
            <h2 className="context-card__title">{entry.title}</h2>
            <p className="context-card__tags">
              {entry.tags?.map((t) => <span key={t} className="wiki-tag">{t}</span>)}
            </p>
          </div>
          <p className="context-card__summary">{entry.summary}</p>
          <button
            type="button" className="context-card__more"
            aria-expanded={expanded}
            onClick={() => onExpand(!expanded)}
          >
            {expanded ? 'Show less' : 'Read more'}
            <svg className="context-card__chev" viewBox="0 0 10 6" aria-hidden="true">
              <path d="M1 1.5 5 5 9 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        {expanded && (
          <div className="context-card__detail">
            <div className="context-card__slot">
              <span className="context-card__slot-label">What makes it different</span>
              <p className="context-card__slot-text">{entry.differs}</p>
            </div>

            {/* Same fixed dials as the reference panel, laid out across the
                width instead of down a rail. */}
            <dl className="context-card__facts">
              {FACT_KEYS[active.kind].map((label) => (
                <div key={label} className="context-card__fact">
                  <dt>{label}</dt>
                  <dd>{factOf(entry, label)}</dd>
                </div>
              ))}
            </dl>

            <p className="context-card__links">
              {entry.links?.length
                ? entry.links.map((l) => (
                  <a key={l.href} href={l.href} target="_blank" rel="noreferrer noopener">{l.label}</a>
                ))
                : <span className="context-card__nolink">No external source</span>}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ── Facet filter ───────────────────────────────────────────────
   Checkboxes, not a single-choice dropdown: the useful question is
   almost always "these four, without that one". One component serves
   both facets — methods grouped by family, models grouped by
   architecture — because the question is identical in both. */

// `indeterminate` is a DOM property with no HTML attribute, so it can only be
// set imperatively.
function TriCheckbox({ checked, indeterminate, ...rest }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.indeterminate = Boolean(indeterminate); }, [indeterminate]);
  return <input ref={ref} type="checkbox" checked={checked} {...rest} />;
}

/* Two facets, one rail. Tabs instead of a second stacked list: the panel keeps
   one height, and the count on the idle tab reports its state without opening
   it. Only shown when there is a second facet to reach — see ControlsPanel. */
function RailTabs({ items, value, onChange }) {
  return (
    <div className="rail-tabs" role="tablist" aria-label="Filter facet">
      {items.map((item) => (
        <button
          key={item.value} type="button" role="tab"
          aria-selected={value === item.value}
          className={`rail-tabs__tab${value === item.value ? ' rail-tabs__tab--active' : ''}`}
          onClick={() => onChange(item.value)}
        >
          {item.label}
          <span className="rail-tabs__count">{item.selected}/{item.total}</span>
        </button>
      ))}
    </div>
  );
}

function FacetFilter({
  groups, selected, onChange, disabled, onCollapse,
  label = 'Methods', noun = 'method', wikiKind = 'method', emptyText, tabs = null,
}) {
  const [query, setQuery] = useState('');
  const all = useMemo(() => groups.flatMap((g) => g.methods), [groups]);
  const chosen = useMemo(() => new Set(selected), [selected]);

  const q = normalize(query);
  const visible = groups
    .map((g) => ({ ...g, methods: g.methods.filter((m) => !q || normalize(m).includes(q)) }))
    .filter((g) => g.methods.length);

  const setFrom = (next) => onChange(all.filter((m) => next.has(m)));
  const toggle = (m) => {
    const next = new Set(chosen);
    if (next.has(m)) next.delete(m); else next.add(m);
    setFrom(next);
  };
  const toggleGroup = (methods, on) => {
    const next = new Set(chosen);
    for (const m of methods) if (on) next.add(m); else next.delete(m);
    setFrom(next);
  };

  return (
    <div className="facet-filter">
      <div className="facet-filter__head">
        {/* The tabs carry the facet's name and count, so the plain label only
            appears when this is the only list in the rail. */}
        {tabs ?? (
          <>
            <span className="tiny-form__label">{label}</span>
            <span className="facet-filter__count">{selected.length}/{all.length}</span>
          </>
        )}
        {onCollapse && (
          <button
            type="button" className="panel-collapse-btn" onClick={onCollapse}
            title={`Collapse ${label.toLowerCase()}`} aria-label={`Collapse ${label.toLowerCase()}`}
          >
            {/* On a phone the rail folds up into a bar rather than sideways, so
                the arrow turns to match and picks up the word it is hiding —
                a lone glyph in the corner read as part of the crumb above it. */}
            <span className="panel-collapse-btn__icon" aria-hidden="true">&#8592;</span>
            <span className="panel-collapse-btn__label">Hide {label.toLowerCase()}</span>
          </button>
        )}
      </div>

      {disabled || all.length === 0 ? (
        <p className="status-message">{emptyText ?? `Choose a dataset to list its ${noun}s.`}</p>
      ) : (
        <>
          <input
            className="combo__input" value={query} placeholder={`Filter ${noun}s`}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="facet-filter__actions">
            <button type="button" onClick={() => onChange(all)}>All</button>
            <button type="button" onClick={() => onChange([])}>None</button>
          </div>
          <div className="facet-filter__list">
            {visible.length === 0 && <p className="combo__empty">No matches — try a shorter query.</p>}
            {visible.map((group) => {
              const allOn = group.methods.every((m) => chosen.has(m));
              // A family with one member whose name is the family name would
              // print itself twice — ALEXNET over alexnet. The row alone says it.
              const sameName = group.methods.length === 1 && normalize(group.methods[0]) === normalize(group.label);
              return (
                <div key={group.key} className="facet-filter__group">
                  {/* One checkbox for the whole family. Partly-checked shows
                      the dash, so the group's state is readable at a glance
                      without counting its children. */}
                  {!sameName && (
                  <label className="facet-filter__group-head">
                    <TriCheckbox
                      checked={allOn}
                      indeterminate={!allOn && group.methods.some((m) => chosen.has(m))}
                      onChange={() => toggleGroup(group.methods, !allOn)}
                      aria-label={`All ${group.label} ${noun}s`}
                    />
                    <span className="facet-filter__group-label">{group.label}</span>
                  </label>
                  )}
                  {/* The info dot sits outside the <label>, or clicking it
                      would toggle the checkbox on the way through. */}
                  {group.methods.map((m) => (
                    <div key={m} className="facet-check">
                      <label className="facet-check__label">
                        <input type="checkbox" checked={chosen.has(m)} onChange={() => toggle(m)} />
                        <span className="facet-check__name">{m}</span>
                      </label>
                      <InfoDot kind={wikiKind} id={m} label={m} />
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function MethodFigures({ methods, outputs, imageId, originalSrc, interpretabilityMetrics }) {
  const entries = resolveMethodEntries(methods, outputs);
  if (!entries.length) return <MiniImage caption="Method" missingText="No method checked in the sidebar." />;
  return entries.map(([name, url]) => (
    <MiniImage
      key={name} caption={name} src={url} originalSrc={originalSrc} wikiKind="method"
      alt={`${name} for image ${imageId}`} metrics={interpretabilityMetrics?.[name]}
    />
  ));
}

function PredictionBadge({ prediction, classId, labels }) {
  if (!prediction) return null;
  const predId = prediction.predicted_class_id;
  const predLabel = labels?.[predId] ?? `Class ${predId}`;
  const isCorrect = String(predId) === String(classId);
  const confidencePct = prediction.confidence == null
    ? null
    : `${(Number(prediction.confidence) * 100).toFixed(1)}%`;
  // A tinted pill, because a bare line of text floated loose under the image
  // with nothing tying it to the picture. It states a fact and is never a
  // button: no fill saturated enough to read as one, and no hover state.
  return (
    <p
      className={`prediction prediction--${isCorrect ? 'correct' : 'incorrect'}`}
      title={`Predicted class ${predId} \u2014 ${isCorrect ? 'matches' : 'does not match'} class ${classId}`}
    >
      <span
        className="prediction__icon" role="img"
        aria-label={isCorrect ? 'Correct prediction' : 'Incorrect prediction'}
      >
        {isCorrect ? '\u2713' : '\u2717'}
      </span>
      <span className="prediction__label">{predLabel}</span>
      {confidencePct && <span className="prediction__confidence">{confidencePct}</span>}
    </p>
  );
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state-shell">
      <div className="empty-state">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

// Loading and failure are inline sentences that name the next action —
// no spinner, no skeleton, no toast.
function AppStatus({ children }) {
  return (
    <div className="app-status">
      <p className="status-message" role="status" aria-live="polite">{children}</p>
    </div>
  );
}

// A label and a hairline. Replaces the heading-plus-blurb block that used to
// sit above every section and say nothing the reader didn't already know.
function SectionRule({ label, children }) {
  return (
    <div className="section-rule">
      <span className="section-rule__label">{label}</span>
      <span className="section-rule__line" aria-hidden="true" />
      {children}
    </div>
  );
}

// Stats read left to right on hairlines, not as six floating cards.
function ModelStatsRail({ model, dataset, stats }) {
  if (!model || !dataset || !stats) return null;

  const items = [
    { label: 'Samples', value: stats.total },
    { label: 'Correct', value: stats.correct },
    { label: 'Accuracy', value: formatMetricPercent(stats.accuracy) },
    { label: 'Precision', value: formatMetricPercent(stats.macroPrecision) },
    { label: 'Recall', value: formatMetricPercent(stats.macroRecall) },
    { label: 'F1', value: formatMetricPercent(stats.macroF1) },
  ];

  return (
    <section className="stat-rail" aria-label="Selected model metrics">
      <div className="stat-rail__run">
        <span className="stat-rail__label">Run</span>
        <strong className="stat-rail__run-name">
          {model}<span> / {dataset}</span>
        </strong>
      </div>
      <dl className="stat-rail__stats">
        {items.map((item) => (
          <div key={item.label} className="stat-rail__stat">
            <dt className="stat-rail__label">{item.label}</dt>
            <dd className="stat-rail__value">{item.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ModeSwitcher({ value, onChange }) {
  const items = [
    { value: 'single', label: 'Single' },
    { value: 'model_grid', label: 'By Model' },
    { value: 'class_compare', label: 'Across Models' },
  ];
  return (
    <div className="mode-switcher" role="tablist" aria-label="Viewer mode">
      {items.map((item) => (
        <button
          key={item.value} type="button" role="tab"
          aria-selected={value === item.value}
          className={`mode-switcher__tab${value === item.value ? ' mode-switcher__tab--active' : ''}`}
          onClick={() => onChange(item.value)}
        >{item.label}</button>
      ))}
    </div>
  );
}

// Mode is navigation, not a filter — it belongs in the chrome with the
// wordmark and the live readout, not stacked on top of the dropdowns.
function TopBar({ mode, onModeChange, readout, wikiOn, onToggleWiki }) {
  return (
    <header className="topbar">
      <span className="topbar__brand">
        <AtlasMark size={22} />
        <span className="topbar__name">NeuralAtlas</span>
      </span>
      <ModeSwitcher value={mode} onChange={onModeChange} />
      <p className="topbar__readout" role="status" aria-live="polite">{readout}</p>
      <button
        type="button" className={`topbar__wiki${wikiOn ? ' is-on' : ''}`}
        onClick={onToggleWiki} aria-pressed={wikiOn}
        title={wikiOn ? 'Hide reference' : 'Show reference'}
        aria-label={wikiOn ? 'Hide reference' : 'Show reference'}
      >
        {/* An open book, not the word: it sits next to the theme glyph and
            reads as a way out of the page at a glance. */}
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10 5.6C8.6 4.3 6.6 3.8 3.9 4a1 1 0 0 0-.9 1v8.7a1 1 0 0 0 1 1c2.6-.2 4.5.3 6 1.5 1.5-1.2 3.4-1.7 6-1.5a1 1 0 0 0 1-1V5a1 1 0 0 0-.9-1c-2.7-.2-4.7.3-6.1 1.6Z" />
          <path d="M10 5.6v10.6" />
        </svg>
      </button>
      <ThemeToggle />
    </header>
  );
}

/* One step of the selection path. Closed it is a label and its current value —
   what is on screen, stated in one line. Open it is the same searchable list
   that used to live in the rail, just brought to where the answer is read. */
function CrumbSelect({ label, value, items, onSelect, placeholder, disabled }) {
  const inputId = useId();
  const list = (items ?? []).map((i) => typeof i === 'string' ? { value: i, label: i } : i);
  const selectedLabel = value == null ? '' : list.find((i) => i.value === value)?.label ?? String(value);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const rootRef = useRef(null);
  const inputRef = useRef(null);

  const q = normalize(query);
  const filtered = q ? list.filter((i) => !i.isHeader && normalize(i.label).includes(q)) : list;

  const commit = (v) => { onSelect(v); setOpen(false); setQuery(''); };

  // Outside-click rather than blur: the trigger is a button, and blur-to-close
  // would fight the click that is trying to toggle it shut.
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (!rootRef.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  useEffect(() => { if (open) inputRef.current?.focus(); }, [open]);

  return (
    <div className={`crumb${disabled ? ' is-disabled' : ''}`} ref={rootRef}>
      <button
        type="button" id={inputId}
        className={`crumb__trigger${open ? ' is-open' : ''}`}
        disabled={disabled}
        aria-haspopup="listbox" aria-expanded={open}
        onClick={() => { setOpen((o) => !o); setQuery(''); }}
      >
        <span className="crumb__label">{label}</span>
        <span className="crumb__line">
          <span className="crumb__value">{selectedLabel || placeholder}</span>
          {/* The caret is the only thing saying "this opens". Without it the
              crumb reads as a printed caption. */}
          <svg className="crumb__caret" viewBox="0 0 10 6" aria-hidden="true">
            <path d="M1 1.5 5 5 9 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {open && !disabled && (
        <div className="crumb__pop">
          <input
            ref={inputRef} className="combo__input" value={query}
            placeholder={placeholder} aria-label={`Search ${label}`}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { const f = filtered.find((i) => !i.isHeader); if (f) commit(f.value); }
            }}
          />
          <div className="crumb__list" role="listbox">
            {filtered.length === 0 ? (
              <div className="combo__empty">No matches — try a shorter query.</div>
            ) : filtered.slice(0, 1000).map((item) =>
              item.isHeader ? (
                <div key={item.value} className="combo__group-header">{item.label}</div>
              ) : (
                <button
                  type="button" key={item.value} role="option"
                  aria-selected={item.value === value}
                  className={`combo__option${item.value === value ? ' is-on' : ''}`}
                  onClick={() => commit(item.value)}
                >{item.label}</button>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* The selection — model, dataset, class, image — in the chrome, where it reads
   as the caption for the whole screen instead of four dropdowns in a rail. */
function SelectionBar({ children }) {
  return <div className="selection-bar">{children}</div>;
}

/* ── View Components ────────────────────────────────────────── */

function SingleImageGallery({ imageData, labels }) {
  if (!imageData?.original) {
    return <EmptyState title="No image selected yet" description="Choose a model, dataset, class, and image to inspect." />;
  }

  const outputs = Object.entries(imageData.outputs ?? {});
  return (
    <section className="image-gallery">
      <SectionRule label={`Original + ${outputs.length} attribution${outputs.length === 1 ? '' : 's'}`} />
      <div className="gallery-grid">
        <div className="image-card" style={{ '--delay': '80ms' }}>
          <div className="image-card__header image-card__header--original">
            <h3>Image</h3>
            <PredictionBadge prediction={imageData.prediction} classId={imageData.classId} labels={labels} />
          </div>
          <OriginalImage className="image-card__image image-card__image--original" src={resolveAssetUrl(imageData.original)} alt="Original selection" />
        </div>
        {outputs.map(([method, url]) => (
          <div key={method} className="image-card">
            <div className="image-card__header">
              <h3>{method}</h3>
              <InfoDot kind="method" id={method} label={method} />
            </div>
            <Attribution className="image-card__image image-card__image--attribution" src={resolveAssetUrl(url)} originalSrc={resolveAssetUrl(imageData.original)} alt={`${method} explanation`} />
            <MetricBadges metrics={imageData.interpretabilityMetrics?.[method]} />
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelGridView({ records, methods, ready, labels }) {
  if (!ready) return <EmptyState title="Choose model and dataset" description="Select a model and dataset to browse many images at once." />;
  if (!records.length) return <EmptyState title="No images match your filters" description="Try a different class selection to widen the result set." />;

  return (
    <section className="model-grid" aria-label="Model image grid">
      {records.map((record) => (
        <article key={`${record.model}__${record.dataset}__${record.classId}__${record.imageId}`} className="model-grid-card">
          <header className="model-grid-card__meta">
            <h3>Class {record.classId} - {record.classLabel}</h3>
            <p>Image {record.imageId}{record.filename ? ` - ${record.filename}` : ''}</p>
            <PredictionBadge prediction={record.prediction} classId={record.classId} labels={labels} />
          </header>
          <div className="model-grid-card__images">
            <MiniImage caption="Original" src={record.originalUrl} alt={`Original ${record.imageId}`} missingText="Original unavailable" variant="original" />
            <MethodFigures
              methods={methods} outputs={record.outputs} imageId={record.imageId}
              originalSrc={record.originalUrl} interpretabilityMetrics={record.interpretabilityMetrics}
            />
          </div>
        </article>
      ))}
    </section>
  );
}

function ClassCompareView({ matrix, methods, ready, labels, onHideModel, totalModels }) {
  const rowRefs = useRef([]);
  const activeRowRef = useRef(null);
  // How far position: sticky is holding the last row below its place in the
  // flow right now. scrollToRow has to take it back out of the row's rect.
  const heldRef = useRef(0);

  // The last row has no next row to scroll into: the page runs out with the
  // original still parked under the header, a screen above the maps that just
  // arrived. So the maps scroll through first, like in any other row, and the
  // scroll that is left after that belongs to the original. The matrix grows a
  // tail exactly as long as the trip the original still has ahead of it
  // (--compare-tail), the row holds still while that tail goes by (sticky at
  // --compare-pin-top, see App.css), and the original comes down by however
  // far the row is being held. One thing moves at a time, and since the held
  // row covers the tail, the page never shows a blank stretch.
  // useLayoutEffect, because the tail and the pin are layout: a new matrix
  // should not paint one frame with the previous one's numbers.
  useLayoutEffect(() => {
    const row = rowRefs.current[matrix.rows.length - 1];
    const section = row?.parentElement;
    const original = row?.querySelector('.compare-original__content');
    const maps = row?.querySelectorAll('.compare-cell--map');
    const lastMap = maps?.[maps.length - 1];
    if (!original || !lastMap) return;

    const root = document.documentElement;
    let tail = 0;
    let inset = 0;

    const clear = () => {
      tail = 0;
      heldRef.current = 0;
      section.style.removeProperty('--compare-tail');
      section.style.removeProperty('--compare-pin-top');
      original.style.translate = '';
    };

    // Geometry. It changes with the viewport and the content, never with the
    // scroll position, so none of this runs per frame.
    const measure = () => {
      const style = getComputedStyle(original);
      // One-column list: nothing sticks, so there is nothing to line up.
      if (style.position !== 'sticky') { clear(); return; }
      const image = original.querySelector('.mini-image__asset, .mini-image__missing') ?? original;
      const target = lastMap.querySelector('.mini-image__asset, .mini-image__missing') ?? lastMap;
      const sectionStyle = getComputedStyle(section);
      inset = parseFloat(sectionStyle.paddingBottom) + parseFloat(sectionStyle.borderBottomWidth);

      const rowRect = row.getBoundingClientRect();
      const contentRect = original.getBoundingClientRect();
      // The matrix is the last child of the content column. Use its bottom
      // padding, not the page's scrollHeight: the tail changes scrollHeight
      // and would feed its own measurement back into the pin position.
      const below = parseFloat(getComputedStyle(section.parentElement).paddingBottom);
      // Where the row's top edge sits once its bottom edge has reached the fold.
      const pinTop = root.clientHeight - below - rowRect.height;
      const stickyTop = parseFloat(style.top) || 0;
      // Where the original rests at that moment: under the header, inside its cell.
      const rest = Math.min(Math.max(stickyTop, pinTop), pinTop + rowRect.height - contentRect.height);
      // Both offsets are internal to the row, so scroll and translate cancel out.
      const imageInContent = image.getBoundingClientRect().top - contentRect.top;
      const targetInRow = target.getBoundingClientRect().top - rowRect.top;
      const travel = pinTop + targetInRow - (rest + imageInContent);

      // A row that fits under the header shows its original and its maps together
      // already. Keep the added scroll distance on whole pixels.
      tail = pinTop < stickyTop ? Math.max(0, Math.round(travel)) : 0;
      if (!tail) { clear(); return; }
      section.style.setProperty('--compare-tail', `${tail}px`);
      section.style.setProperty('--compare-pin-top', `${pinTop}px`);
    };

    // Per scroll frame: the original comes down exactly as far as sticky is
    // holding the row. Reads first, one write after, so no forced reflow.
    const update = () => {
      if (!tail) return;
      const slack = section.getBoundingClientRect().bottom - inset - row.getBoundingClientRect().bottom;
      const held = Math.max(0, Math.min(tail, tail - slack));
      heldRef.current = held;
      original.style.translate = held ? `0 ${held}px` : '';
    };

    let frame = 0;
    let stale = false;
    const run = () => {
      frame = 0;
      if (stale) { stale = false; measure(); }
      update();
    };
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(run); };
    const onResize = () => { stale = true; onScroll(); };
    const observer = new ResizeObserver(onResize);
    observer.observe(row);
    observer.observe(original);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    measure();
    update();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
      clear();
    };
  }, [matrix, methods, ready]);

  const scrollToRow = (index) => {
    const row = rowRefs.current[index];
    if (!row) return;
    activeRowRef.current = index;
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    // A pinned last row reports where sticky is holding it, not where it lives.
    const held = row.matches(':last-child') ? heldRef.current : 0;
    const top = row.getBoundingClientRect().top - held + window.scrollY
      - parseFloat(getComputedStyle(row).scrollMarginTop);
    window.scrollTo({ left: window.scrollX, top, behavior: reduceMotion ? 'auto' : 'smooth' });
  };

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName))) return;

      const plainKey = !event.metaKey && !event.ctrlKey && !event.altKey;
      // Two ways to walk the rows: J/K for the keyboard-first reader, and the
      // arrows — left/right because the rows read as a filmstrip of images,
      // shift+up/down because the page itself scrolls vertically.
      const forward = ['j', 'arrowright'];
      const backward = ['k', 'arrowleft'];
      const key = event.key.toLowerCase();
      const delta = plainKey && !event.shiftKey && forward.includes(key)
        ? 1
        : plainKey && !event.shiftKey && backward.includes(key)
          ? -1
          : plainKey && event.shiftKey && event.key === 'ArrowDown'
            ? 1
            : plainKey && event.shiftKey && event.key === 'ArrowUp'
              ? -1
              : 0;
      if (!delta) return;

      const rows = rowRefs.current.filter(Boolean);
      const stickyBottoms = ['.topbar', '.selection-bar', '.render-bar', '.compare-header', '.viewer-layout--collapsed .controls-panel']
        .map((selector) => document.querySelector(selector)?.getBoundingClientRect())
        .filter((rect) => rect?.height > 0 && rect.bottom > 0)
        .map((rect) => rect.bottom);
      const anchor = Math.max(0, ...stickyBottoms) + 8;
      const visible = rows.findIndex((row) => {
        const rect = row.getBoundingClientRect();
        return rect.top <= anchor && rect.bottom > anchor;
      });
      const fallback = rows.reduce((best, row, index) => {
        const distance = Math.abs(row.getBoundingClientRect().top - anchor);
        return distance < best.distance ? { index, distance } : best;
      }, { index: 0, distance: Infinity }).index;
      const current = activeRowRef.current ?? (visible < 0 ? fallback : visible);
      const next = Math.max(0, Math.min(rows.length - 1, current + delta));
      if (next === current) return;
      event.preventDefault();
      scrollToRow(next);
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [matrix.rows]);

  if (!ready) return <EmptyState title="Choose dataset and class" description="Pick a dataset and class to align the same image IDs across models." />;
  if (totalModels > 0 && !matrix.models.length) return <EmptyState title="No models checked" description="Check at least one model in the rail to raise a column." />;
  if (!matrix.rows.length) return <EmptyState title="No aligned rows found" description="No images are available for this class across the selected models." />;

  const style = { '--compare-columns': matrix.models.length + 1 };
  return (
    <section className="compare-matrix" style={style} aria-label="Class comparison matrix">
      <div className="compare-header">
        <div className="compare-header__gutter" aria-hidden="true" />
        <div className="compare-header__cell compare-header__cell--original">Original</div>
        {matrix.models.map((m) => (
          <div key={m} className="compare-header__cell">
            <span className="compare-header__name">{m}</span>
            <InfoDot kind="model" id={m} label={m} />
            {/* Dropping a column is a thought you have while reading the
                column, so the control lives on it. Same act as unchecking it
                in the rail; it appears on hover and never shifts the header. */}
            {onHideModel && (
              <button
                type="button" className="compare-header__hide"
                onClick={() => onHideModel(m)}
                title={`Hide ${m}`} aria-label={`Hide ${m} column`}
              >&#215;</button>
            )}
          </div>
        ))}
      </div>
      {matrix.rows.map((row, rowIndex) => {
        const origUrl = row.cells.find((c) => c.record?.originalUrl)?.record?.originalUrl ?? null;
        const methodRows = compareMethodRows(methods, row.cells);
        const rowLabel = `Image ${row.imageId}`;
        // The original spans every method row, so the grid needs those rows to
        // be explicit — an implicit grid has nothing for `1 / -1` to reach.
        // Two rows per method: the name, then the band of maps it names.
        const rowStyle = { '--compare-rows': methodRows.length * 2 + 1 };
        return (
          <div
            key={row.sourceKey} className="compare-row" style={rowStyle}
            ref={(node) => { rowRefs.current[rowIndex] = node; }}
          >
            <article className="compare-cell compare-cell--original">
              <div className="compare-original__content">
                <MiniImage
                  caption={rowLabel} src={origUrl} alt={`Original ${rowLabel.toLowerCase()}`}
                  missingText="Original unavailable" variant="original"
                  captionActions={(
                    <nav className="compare-nav" aria-label={`Image ${rowIndex + 1} of ${matrix.rows.length}`}>
                      <span className="compare-nav__count">{rowIndex + 1} / {matrix.rows.length}</span>
                      <button
                        type="button" className="compare-nav__button"
                        disabled={rowIndex === 0} onClick={() => scrollToRow(rowIndex - 1)}
                        aria-label="Previous image" title="Previous image · K or ←"
                      >&#8249;</button>
                      <button
                        type="button" className="compare-nav__button"
                        disabled={rowIndex === matrix.rows.length - 1} onClick={() => scrollToRow(rowIndex + 1)}
                        aria-label="Next image" title="Next image · J or →"
                      >&#8250;</button>
                    </nav>
                  )}
                />
              </div>
            </article>
            {/* Row 1 of the matrix: what each model called this image. The
                gutter above the method names stays empty on purpose. */}
            <div className="compare-row__gutter" aria-hidden="true" />
            {row.cells.map((cell) => (
              <div key={`prediction__${cell.model}`} className="compare-cell compare-cell--prediction" data-model={cell.model}>
                <PredictionBadge prediction={cell.record?.prediction} classId={row.classId} labels={labels} />
              </div>
            ))}
            {!methodRows.length && (
              <p className="compare-row__empty">No method checked in the sidebar.</p>
            )}
            {methodRows.map((method) => (
              <Fragment key={method}>
                <h4 className="compare-row__method">
                  <span className="compare-row__method-label">
                    <span className="compare-row__method-name">{method}</span>
                    <InfoDot kind="method" id={method} label={method} />
                  </span>
                </h4>
                {row.cells.map((cell) => (
                  <div key={`${method}__${cell.model}`} className="compare-cell compare-cell--map" data-model={cell.model}>
                    <MiniImage
                      src={cell.record?.outputs?.[method]}
                      originalSrc={cell.record?.originalUrl}
                      alt={`${method} for image ${row.imageId} on ${cell.model}`}
                      missingText={`No ${method} map`}
                      metrics={cell.record?.interpretabilityMetrics?.[method]}
                    />
                  </div>
                ))}
              </Fragment>
            ))}
          </div>
        );
      })}
    </section>
  );
}

function resolveSelection(options, currentValue) {
  if (!options.length) return null;
  if (currentValue && options.some((o) => (o.value ?? o) === currentValue)) return currentValue;
  return options[0].value ?? options[0];
}

/* ── Main Form ──────────────────────────────────────────────── */

function ModelForm({ atlas, runs }) {
  const { models, records } = atlas;

  const [vs, setVs] = useState(() => ({
    mode: 'single', model: null, dataset: null, classId: null, imageId: null, methods: null,
    ...readStateFromUrl(),
  }));
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const panelRef = useRef(null);
  const [facet, setFacet] = useState('method');
  const [overlay, setOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [lblCache, setLblCache] = useState({});
  const [dsStatus, setDsStatus] = useState({});

  const patch = (update) => { const next = { ...vs, ...update }; setVs(next); writeStateToUrl(next); };

  // Where the rail lies down across the top of the content, the bar that opens
  // it is sticky but the panel it opens is not: tapping it three screens down
  // expanded a list nobody could see. Opening it goes to it.
  const isPhone = () => window.matchMedia?.('(max-width: 760px)').matches ?? false;
  const scrollIntoViewSoon = (target, block = 'start') => {
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    setTimeout(() => {
      const node = typeof target === 'string' ? document.querySelector(target) : target;
      node?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block });
    }, 0);
  };
  // The card is pinned to the top of the content column and the dot that aims
  // it can be several screens below, so opening it goes to it — on any width,
  // not just where the rail lies down. scrollIntoView on its own would tuck the
  // top of the card under the sticky bars, so the landing point clears them.
  const scrollContextCardIntoView = () => {
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    setTimeout(() => {
      const card = document.querySelector('.context-card');
      if (!card) return;
      const offset = ['.topbar', '.selection-bar']
        .map((selector) => document.querySelector(selector)?.getBoundingClientRect())
        .filter((rect) => rect?.height > 0)
        .reduce((sum, rect) => sum + rect.height, 0) + 8;
      const box = card.getBoundingClientRect();
      // Already reading it: a nudge of a few pixels is worse than staying put.
      if (box.top >= offset && box.bottom <= window.innerHeight) return;
      window.scrollTo({
        top: Math.max(0, box.top + window.scrollY - offset),
        behavior: reduceMotion ? 'auto' : 'smooth',
      });
    }, 0);
  };

  const expandPanel = () => {
    setPanelCollapsed(false);
    if (isPhone()) scrollIntoViewSoon(panelRef.current);
  };

  const modelOptions = useMemo(() => Object.keys(models).sort(), [models]);

  const effectiveModel = resolveSelection(modelOptions, vs.model);

  const datasetOptions = useMemo(() => {
    if (vs.mode === 'class_compare') {
      return [...new Set(
        Object.values(models).flatMap((datasets) => Object.keys(datasets))
      )].sort(compareMixedIds);
    }
    if (!effectiveModel) return [];
    return Object.keys(models[effectiveModel] ?? {}).sort(compareMixedIds);
  }, [vs.mode, effectiveModel, models]);

  const effectiveDataset = resolveSelection(datasetOptions, vs.dataset);

  // Load dataset metadata (images structure + labels)
  useEffect(() => {
    if (!effectiveDataset) return;
    const controller = new AbortController();
    const { signal } = controller;
    const ds = effectiveDataset;
    const updStatus = (fields) =>
      !signal.aborted && setDsStatus((p) => ({ ...p, [ds]: { ...p[ds], ...fields } }));

    if (!lblCache[ds]) {
      updStatus({ labelsLoading: true, labelsError: null });
      fetchJson('imagenet-mini/imagenet-1k-id2label.json', { signal, retryCount: 2 })
        .then((data) => { if (!signal.aborted) setLblCache((p) => ({ ...p, [ds]: data })); })
        .catch((e) => {
          if (e.name !== 'AbortError') {
            updStatus({ labelsError: 'Failed to load.' });
          }
        })
        .finally(() => updStatus({ labelsLoading: false }));
    }

    return () => controller.abort();
  }, [effectiveDataset, lblCache]);

  const imageRecords = useMemo(
    () => records.map((r) => ({ ...r, classLabel: lblCache[r.dataset]?.[r.classId] ?? r.classId })),
    [records, lblCache]
  );

  const classOptions = useMemo(() => {
    if (!effectiveDataset) return [];
    const labels = lblCache[effectiveDataset] ?? {};
    const classes = imageRecords.filter((r) => r.dataset === effectiveDataset &&
      (vs.mode === 'class_compare' || r.model === effectiveModel)).map((r) => r.classId);
    return [...new Set(classes)].sort(compareMixedIds)
      .map((id) => ({ value: id, label: `${id} - ${labels[id] ?? id}` }));
  }, [vs.mode, effectiveDataset, effectiveModel, lblCache, imageRecords]);

  const effectiveClassId = resolveSelection(classOptions, vs.classId);

  const modelGridRecords = useMemo(() => imageRecords.filter((r) =>
    r.model === effectiveModel && r.dataset === effectiveDataset &&
    (!effectiveClassId || r.classId === effectiveClassId)
  ), [imageRecords, effectiveModel, effectiveDataset, effectiveClassId]);

  const imageOptions = useMemo(() => modelGridRecords.map((r) => ({
    value: r.imageId,
    label: r.originalUrl ? `${r.imageId} - ${r.filename}` : r.imageId,
  })), [modelGridRecords]);

  const effectiveImageId = vs.mode === 'single' ? resolveSelection(imageOptions, vs.imageId) : null;

  const methodGroups = useMemo(() => {
    if (!effectiveDataset) return [];
    const methods = new Set();
    const comparedModels = vs.mode === 'class_compare' && vs.models
      ? new Set(vs.models.split(',').filter(Boolean))
      : null;
    for (const r of imageRecords) {
      if (r.dataset !== effectiveDataset) continue;
      if (vs.mode !== 'class_compare' && r.model !== effectiveModel) continue;
      if (comparedModels && !comparedModels.has(r.model)) continue;
      if (effectiveClassId && r.classId !== effectiveClassId) continue;
      if (vs.mode === 'single' && effectiveImageId && r.imageId !== effectiveImageId) continue;
      Object.keys(r.outputs).forEach((method) => methods.add(method));
    }
    const grouped = {};
    for (const m of methods) { const c = categorizeMethod(m); (grouped[c] ??= []).push(m); }
    return [...Object.keys(METHOD_CATEGORIES), 'other']
      .filter((c) => grouped[c]?.length)
      .map((c) => ({
        key: c,
        label: METHOD_CATEGORIES[c]?.label ?? 'Other',
        methods: grouped[c].sort((a, b) => compareMethods(c, a, b)),
      }));
  }, [imageRecords, vs.mode, vs.models, effectiveModel, effectiveDataset, effectiveClassId, effectiveImageId]);

  const availableMethods = useMemo(() => methodGroups.flatMap((g) => g.methods), [methodGroups]);

  // Every model with a run for this dataset and class — the full set of columns
  // the compare view could raise, before the rail has its say.
  const modelGroups = useMemo(() => {
    if (!effectiveDataset || !effectiveClassId) return [];
    const names = new Set();
    for (const r of imageRecords) {
      if (r.dataset === effectiveDataset && r.classId === effectiveClassId) names.add(r.model);
    }
    return groupModels([...names]);
  }, [imageRecords, effectiveDataset, effectiveClassId]);

  const availableModels = useMemo(() => modelGroups.flatMap((g) => g.methods), [modelGroups]);

  // Absent from the URL means "all of them", so a link keeps working when the
  // dataset gains a method. An explicit list is intersected with what exists.
  const selectedMethods = useMemo(() => {
    if (vs.methods == null) return availableMethods;
    const wanted = new Set(vs.methods.split(',').filter(Boolean));
    return availableMethods.filter((m) => wanted.has(m));
  }, [vs.methods, availableMethods]);

  // Same convention as methods: absent from the URL means "all of them", so a
  // shared link keeps working after the sweep adds convnext.
  const selectedModels = useMemo(() => {
    if (vs.models == null) return availableModels;
    const wanted = new Set(vs.models.split(',').filter(Boolean));
    return availableModels.filter((m) => wanted.has(m));
  }, [vs.models, availableModels]);

  // The context card is the only reference surface on the page, so its state
  // lives here: which subject it reads, whether the full entry is open, and
  // whether it is on screen at all (the book in the top bar).
  const [contextTab, setContextTab] = useState('model');
  const [contextOpen, setContextOpen] = useState(false);
  const [contextShown, setContextShown] = useState(false);
  const [pickedMethod, setPickedMethod] = useState(null);
  const [pickedModel, setPickedModel] = useState(null);
  // A method named from the sidebar may not be checked, so it is enough that
  // it be documented — otherwise fall back to the first one on screen.
  const contextMethod = pickedMethod && selectedMethods.includes(pickedMethod)
    && lookupWiki('method', pickedMethod)
    ? pickedMethod
    : selectedMethods[0] ?? null;

  // Anything that names a subject — an info dot on a method caption, a
  // checkbox in the rail — aims the card at it and opens the full entry.
  const wikiApi = useMemo(() => ({
    open: (kind, id) => {
      setContextShown(true);
      setContextOpen(true);
      if (kind) setContextTab(kind);
      if (kind === 'method' && id) setPickedMethod(id);
      if (kind === 'model' && id) setPickedModel(id);
      scrollContextCardIntoView();
    },
  }), []);

  // Without a nudge the card would keep reading whatever was last opened while
  // the selection moved on underneath it, so it follows the last thing touched.
  // Checking a whole family at once is left alone — twenty added methods name
  // no single subject.
  const lastSelection = useRef({ model: null, dataset: null, methods: [] });
  useEffect(() => {
    const prev = lastSelection.current;
    lastSelection.current = { model: effectiveModel, dataset: effectiveDataset, methods: selectedMethods };
    if (prev.model == null && prev.dataset == null) return; // first pass, nothing was touched
    if (effectiveModel !== prev.model) setContextTab('model');
    else if (effectiveDataset !== prev.dataset) { setContextTab('dataset'); setPickedMethod(null); setPickedModel(null); }
    else {
      const added = selectedMethods.filter((m) => !prev.methods.includes(m));
      if (added.length === 1) { setContextTab('method'); setPickedMethod(added[0]); }
    }
  }, [effectiveModel, effectiveDataset, selectedMethods]);

  const setSelectedMethods = (list) => {
    const next = availableMethods.filter((m) => list.includes(m));
    patch({ methods: next.length === availableMethods.length ? null : next.join(',') });
  };

  const setSelectedModels = (list) => {
    const next = availableModels.filter((m) => list.includes(m));
    patch({ models: next.length === availableModels.length ? null : next.join(',') });
  };

  const singleImageData = useMemo(() => {
    if (vs.mode !== 'single' || !effectiveModel || !effectiveDataset || !effectiveClassId || !effectiveImageId) return null;
    const r = imageRecords.find((i) =>
      i.model === effectiveModel && i.dataset === effectiveDataset && i.classId === effectiveClassId && i.imageId === effectiveImageId
    );
    if (!r) return null;
    return {
      original: r.originalUrl,
      outputs: Object.fromEntries(resolveMethodEntries(selectedMethods, r.outputs)),
      interpretabilityMetrics: r.interpretabilityMetrics,
      prediction: r.prediction,
      classId: r.classId,
    };
  }, [imageRecords, vs.mode, effectiveModel, effectiveDataset, effectiveClassId, effectiveImageId, selectedMethods]);

  const classCompareMatrix = useMemo(
    () => getClassCompareMatrix(imageRecords, {
      dataset: effectiveDataset, classId: effectiveClassId, models: selectedModels,
    }),
    [imageRecords, effectiveDataset, effectiveClassId, selectedModels]
  );

  const selectedModelStats = effectiveModel && effectiveDataset
    ? models[effectiveModel]?.[effectiveDataset] ?? null
    : null;

  const hasContent =
    (vs.mode === 'single' && Boolean(singleImageData?.original)) ||
    (vs.mode === 'model_grid' && modelGridRecords.length > 0) ||
    (vs.mode === 'class_compare' && classCompareMatrix.rows.length > 0);

  const dsInfo = effectiveDataset ? dsStatus[effectiveDataset] ?? {} : {};
  const isLoading = dsInfo.labelsLoading;

  const handleModeChange = (mode) => {
    const next = { ...vs, mode };
    if (mode !== 'single') next.imageId = null;
    setVs(next);
    writeStateToUrl(next);
  };

  const methodLabel = (() => {
    if (!availableMethods.length) return 'no methods';
    if (!selectedMethods.length) return 'no method checked';
    if (selectedMethods.length === 1) return selectedMethods[0];
    if (selectedMethods.length === availableMethods.length) return `all ${availableMethods.length} methods`;
    return `${selectedMethods.length} of ${availableMethods.length} methods`;
  })();

  // The readout sits in the top bar next to the mode tabs, so it never has to
  // name the mode — only what the current selection resolves to.
  const summaryText = (() => {
    if (vs.mode === 'single') {
      if (!singleImageData?.original) return 'Complete the selection to begin';
      return `1 image · ${Object.keys(singleImageData.outputs ?? {}).length} attributions`;
    }
    if (vs.mode === 'model_grid') {
      if (!effectiveModel || !effectiveDataset) return 'Select model and dataset';
      const cls = effectiveClassId ? `class ${effectiveClassId}` : 'all classes';
      return `${modelGridRecords.length} images · ${cls} · ${methodLabel}`;
    }
    if (!effectiveDataset || !effectiveClassId) return 'Select dataset and class';
    const modelLabel = classCompareMatrix.models.length === availableModels.length
      ? `${availableModels.length} models`
      : `${classCompareMatrix.models.length} of ${availableModels.length} models`;
    return `${classCompareMatrix.rows.length} rows · ${modelLabel} · ${methodLabel}`;
  })();

  // Models are only a filter where they are columns. Everywhere else the model
  // is a single choice and lives in the selection bar, so the tab would be a
  // second control for a question already answered.
  const showModelFacet = vs.mode === 'class_compare';

  // Where the model is a column rather than a single choice, the card reads the
  // one that was last named — a column header dot, a chip, the rail — and
  // otherwise the first column on screen.
  const contextModel = showModelFacet
    ? (pickedModel && selectedModels.includes(pickedModel) ? pickedModel : selectedModels[0] ?? null)
    : effectiveModel;
  const railTabs = showModelFacet ? (
    <RailTabs
      value={facet} onChange={setFacet}
      items={[
        { value: 'method', label: 'Methods', selected: selectedMethods.length, total: availableMethods.length },
        { value: 'model', label: 'Models', selected: selectedModels.length, total: availableModels.length },
      ]}
    />
  ) : null;

  // Wait only for the active selection, including models still loading that
  // cannot yet appear in the record-derived comparison filters.
  const requiredModels = vs.mode === 'class_compare'
    ? modelOptions.filter((model) => effectiveDataset in models[model]
      && (vs.models == null || vs.models.split(',').includes(model)))
    : effectiveModel && effectiveDataset ? [effectiveModel] : [];
  const requiredRuns = requiredModels.map((model) => runs[`${model}::${effectiveDataset}`]);
  if (requiredRuns.some((run) => run?.error)) {
    return <AppStatus>Could not load the selected run. Reload to try again.</AppStatus>;
  }
  if (requiredRuns.some((run) => !run)) {
    return <AppStatus>Loading the selected view.</AppStatus>;
  }

  return (
    <OverlayContext.Provider value={{ enabled: overlay, opacity: overlayOpacity }}>
    <WikiContext.Provider value={wikiApi}>
    <TopBar
      mode={vs.mode} onModeChange={handleModeChange} readout={summaryText}
      wikiOn={contextShown} onToggleWiki={() => setContextShown((v) => !v)}
    />
    <SelectionBar>
      {vs.mode !== 'class_compare' && (
        <CrumbSelect
          label="Model" value={effectiveModel} items={modelOptions}
          onSelect={(v) => patch({ model: v, classId: null, imageId: null })}
          placeholder="Search model" disabled={!modelOptions.length}
        />
      )}

      <CrumbSelect
        label="Dataset" value={effectiveDataset}
        items={datasetOptions}
        onSelect={(v) => patch({ dataset: v })}
        placeholder="Search dataset"
        disabled={!datasetOptions.length}
      />

      <CrumbSelect
        label="Class" value={effectiveClassId} items={classOptions}
        onSelect={(v) => patch({ classId: v, imageId: null })}
        placeholder={isLoading ? 'Loading class metadata...' : 'Search class'}
        disabled={!effectiveDataset || !classOptions.length}
      />

      {vs.mode === 'single' && (
        <CrumbSelect
          label="Image" value={effectiveImageId} items={imageOptions}
          onSelect={(v) => patch({ imageId: v })}
          placeholder="Search image id / filename"
          disabled={!effectiveClassId}
        />
      )}
    </SelectionBar>
    <div className={`viewer-layout${panelCollapsed ? ' viewer-layout--collapsed' : ''}`}>
      <aside ref={panelRef} className={`controls-panel${panelCollapsed ? ' controls-panel--collapsed' : ''}`}>
        {panelCollapsed ? (
          <button className="panel-toggle" onClick={expandPanel} title="Expand controls">
            <span className="panel-toggle__icon">&#8594;</span>
            <span className="panel-toggle__label">{showModelFacet && facet === 'model' ? 'Models' : 'Methods'}</span>
          </button>
        ) : showModelFacet && facet === 'model' ? (
          <FacetFilter
            groups={modelGroups} selected={selectedModels} onChange={setSelectedModels}
            label="Models" noun="model" wikiKind="model"
            emptyText="Choose a dataset and class to list the models that ran on it."
            tabs={railTabs}
            onCollapse={hasContent ? () => setPanelCollapsed(true) : null}
          />
        ) : (
          <FacetFilter
            groups={methodGroups} selected={selectedMethods}
            onChange={setSelectedMethods} disabled={!effectiveDataset}
            tabs={showModelFacet ? railTabs : null}
            onCollapse={hasContent ? () => setPanelCollapsed(true) : null}
          />
        )}
      </aside>

      <main className="viewer-content">
        {dsInfo.labelsError && (
          <div className="selection-status">
            <p className="status-message" role="status">Some dataset metadata failed to load.</p>
          </div>
        )}
        {/* The card names what is on screen; the render strip sets how it is
            drawn. Reading order follows: subject first, then controls, then
            the maps they act on. */}
        {contextShown && (
          <ContextCard
            dataset={effectiveDataset} model={contextModel} method={contextMethod}
            methodOptions={selectedMethods} onMethod={setPickedMethod}
            modelOptions={showModelFacet ? selectedModels : []}
            onModel={showModelFacet ? setPickedModel : null}
            tab={contextTab} onTab={setContextTab}
            expanded={contextOpen} onExpand={setContextOpen}
          />
        )}

        <RenderBar
          overlay={overlay} opacity={overlayOpacity}
          onToggle={() => setOverlay((v) => !v)} onOpacity={setOverlayOpacity}
        />

        {vs.mode === 'model_grid' && (
          <ModelStatsRail model={effectiveModel} dataset={effectiveDataset} stats={selectedModelStats} />
        )}

        {vs.mode === 'single' && <SingleImageGallery imageData={singleImageData} labels={lblCache[effectiveDataset]} />}
        {vs.mode === 'model_grid' && (
          <ModelGridView records={modelGridRecords} methods={selectedMethods} ready={Boolean(effectiveModel && effectiveDataset)} labels={lblCache[effectiveDataset]} />
        )}
        {vs.mode === 'class_compare' && (
          <ClassCompareView matrix={classCompareMatrix} methods={selectedMethods}
            ready={Boolean(effectiveDataset && effectiveClassId)} labels={lblCache[effectiveDataset]}
            totalModels={availableModels.length}
            onHideModel={(m) => setSelectedModels(selectedModels.filter((x) => x !== m))} />
        )}
      </main>
    </div>
    </WikiContext.Provider>
    </OverlayContext.Provider>
  );
}

function App() {
  const [manifest, setManifest] = useState(null);
  const [runs, setRuns] = useState({});
  const [error, setError] = useState(null);

  useAtlasFavicon();

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    fetchJson('outputs/manifest.json', { signal })
      .then((data) => {
        if (signal.aborted) return;
        setManifest(data);
        for (const [model, datasets] of Object.entries(data.runs ?? {})) {
          for (const [dataset, paths] of Object.entries(datasets)) {
            const key = `${model}::${dataset}`;
            Promise.all([
              fetchJson(paths.images, { signal }),
              fetchJson(paths.summary, { signal }),
            ]).then(([images, summary]) => {
              if (signal.aborted) return;
              // Normalize each run once as it arrives; other runs never gate it.
              const run = buildAtlasData({
                models: [model], datasets_by_model: { [model]: [dataset] },
              }, { [key]: { images, summary, baseUrl: paths.base_url } });
              setRuns((previous) => ({ ...previous, [key]: run }));
            }).catch((error) => {
              if (!signal.aborted) setRuns((previous) => ({ ...previous, [key]: { error } }));
            });
          }
        }
      })
      .catch((error) => {
        if (!signal.aborted) setError(error);
      });

    return () => controller.abort();
  }, []);

  const atlas = useMemo(() => {
    const data = buildAtlasData(manifest, {});
    for (const [model, datasets] of Object.entries(data.models)) {
      for (const dataset of Object.keys(datasets)) {
        const run = runs[`${model}::${dataset}`];
        if (!run || run.error) continue;
        datasets[dataset] = run.models[model][dataset];
        data.records.push(...run.records);
      }
    }
    return data;
  }, [manifest, runs]);

  if (error) return <AppStatus>Could not read outputs/manifest.json. Check that the run outputs are published, then reload.</AppStatus>;
  if (!manifest) return <AppStatus>Reading run manifest.</AppStatus>;

  return <div className="app-shell"><ModelForm atlas={atlas} runs={runs} /></div>;
}

export default App;
