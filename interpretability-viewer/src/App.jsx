import { createContext, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import './App.css';

const ALL_METHODS = '__all_methods__';
const EMPTY_OBJ = {};
const BASE_URL = import.meta.env.BASE_URL ?? '/';

// Overlay display preference (heatmap composited over the original), shared by all views.
const OverlayContext = createContext({ enabled: false, opacity: 0.8 });
const useOverlay = () => useContext(OverlayContext);

const VS_KEYS = ['mode', 'model', 'dataset', 'classId', 'imageId', 'method'];

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
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
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
      'GuidedGradCam', 'GradientShap', 'Saliency', 'IntegratedGradients',
      'LayerGradCam', 'InputXGradient', 'DeepLift', 'DeepLiftShap', 'GuidedBackprop',
      'Deconvolution', 'LayerIntegratedGradients',
    ],
  },
  perturbation: {
    label: 'Perturbation',
    methods: [
      'Occlusion', 'KernelShap', 'Lime', 'FeaturePermutation',
      'FeatureAblation', 'ShapleyValueSampling',
    ],
  },
};

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

function resolveMethodEntries(methodValue, outputs = {}) {
  if (!methodValue) return [];
  if (methodValue === ALL_METHODS) return Object.entries(outputs);
  if (methodValue.startsWith('__group_')) {
    const cat = methodValue.slice(8).replace(/__$/, '');
    return Object.entries(outputs).filter(([m]) => categorizeMethod(m) === cat);
  }
  return [[methodValue, outputs[methodValue] ?? null]];
}

function resolveAssetUrl(path) {
  if (!path) return null;
  const value = String(path);
  if (/^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('data:')) return value;
  return `${BASE_URL}${value.replace(/^\/+/, '')}`;
}

async function fetchJson(path) {
  const response = await fetch(resolveAssetUrl(path));
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function buildLegacyOutputStructure(manifest, runPayloads) {
  const structure = { models: {} };

  for (const model of manifest?.models ?? []) {
    const datasets = manifest?.datasets_by_model?.[model] ?? [];
    const modelNode = structure.models[model] ??= { datasets: {} };

    for (const dataset of datasets) {
      const runKey = `${model}::${dataset}`;
      const run = runPayloads[runKey];
      const datasetNode = modelNode.datasets[dataset] ??= { classes: {} };

      if (run?.summary?.metrics) {
        datasetNode.metrics = run.summary.metrics;
      }

      for (const image of run?.images?.images ?? []) {
        const classId = String(image.class_id);
        const imageId = String(image.image_id);
        const classNode = datasetNode.classes[classId] ??= { images: {} };
        classNode.images[imageId] = {
          outputs: image.outputs ?? {},
          prediction: image.prediction ?? null,
          interpretability_metrics: image.interpretability_metrics ?? {},
          original_url: image.original_url ?? null,
        };
      }
    }
  }

  for (const modelNode of Object.values(structure.models)) {
    const datasets = modelNode?.datasets ?? EMPTY_OBJ;
    const firstDatasetWithMetrics = Object.values(datasets).find((datasetNode) => datasetNode?.metrics);
    if (firstDatasetWithMetrics?.metrics) {
      modelNode.metrics = firstDatasetWithMetrics.metrics;
    }
  }

  return structure;
}

function buildImageRecords(outputStructure, imgCache, lblCache) {
  const records = [];
  for (const [model, { datasets = {} }] of Object.entries(outputStructure?.models ?? {})) {
    for (const [dataset, { classes = {} }] of Object.entries(datasets)) {
      const imgLookup = imgCache?.[dataset] ?? {};
      const lblLookup = lblCache?.[dataset] ?? {};
      for (const [classId, { images = {} }] of Object.entries(classes)) {
        const filenames = imgLookup[classId] ?? [];
        const classLabel = lblLookup[classId] ?? classId;
        for (const [imageId, { outputs = {}, prediction = null, original_url: originalUrl = null } = {}] of Object.entries(images)) {
          const filename = filenames[imageId] ?? null;
          records.push({
            model, dataset, classId, classLabel, imageId, filename,
            originalUrl: originalUrl ?? (filename ? `${dataset}/val/${classId}/${filename}` : null),
            outputs,
            prediction,
          });
        }
      }
    }
  }
  return records.sort((a, b) =>
    a.dataset.localeCompare(b.dataset) || compareMixedIds(a.classId, b.classId) ||
    compareMixedIds(a.imageId, b.imageId) || a.model.localeCompare(b.model)
  );
}

function getClassCompareMatrix(records, { dataset, classId, method }) {
  if (!dataset || !classId) return { models: [], rows: [] };

  const scoped = records.filter((r) => r.dataset === dataset && r.classId === classId);
  const models = [...new Set(scoped.map((r) => r.model))].sort();

  const byImage = new Map();
  for (const r of scoped) {
    if (!byImage.has(r.imageId)) byImage.set(r.imageId, new Map());
    byImage.get(r.imageId).set(r.model, r);
  }

  const rows = [...byImage.keys()].sort(compareMixedIds).map((imageId) => {
    const rowMap = byImage.get(imageId);
    const exemplar = rowMap.values().next().value;
    return {
      imageId,
      filename: exemplar?.filename ?? null,
      classId,
      classLabel: exemplar?.classLabel ?? classId,
      cells: models.map((model) => {
        const record = rowMap.get(model) ?? null;
        return { model, record, methodUrl: method ? record?.outputs?.[method] ?? null : null };
      }),
    };
  });

  return { models, rows };
}

function getModelMetrics(outputStructure) {
  const models = outputStructure?.models ?? EMPTY_OBJ;
  const byModel = {};
  const byModelAndDataset = {};

  for (const [modelName, modelNode] of Object.entries(models)) {
    const datasets = modelNode?.datasets ?? EMPTY_OBJ;
    byModelAndDataset[modelName] = {};

    if (modelNode?.metrics) {
      byModel[modelName] = modelNode.metrics;
    }

    for (const [datasetName, datasetNode] of Object.entries(datasets)) {
      if (datasetNode?.metrics) {
        byModelAndDataset[modelName][datasetName] = datasetNode.metrics;
      }
    }
  }

  return { byModel, byModelAndDataset };
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

function applyJet(img, canvas, overlay) {
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
    // Overlay: amplify low attributions via gamma so diffuse methods are visible.
    const alpha = overlay ? Math.round(Math.pow(gray / 255, 0.5) * 255) : 255;
    px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = alpha;
  }
  ctx.putImageData(d, 0, 0);
}

function JetCanvas({ src, className, alt, overlay = false, opacity }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!src) return undefined;
    const img = new Image();
    let cancelled = false;

    img.decoding = 'async';
    img.onload = () => {
      if (!cancelled) applyJet(img, canvasRef.current, overlay);
    };
    img.src = src;

    return () => { cancelled = true; };
  }, [src, overlay]);

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
  if (!enabled || !originalSrc) return <JetCanvas className={className} src={src} alt={alt} />;
  return (
    <div className={`${className} overlay-stack`}>
      <img className="overlay-stack__base" src={originalSrc} alt="" loading="lazy" />
      <JetCanvas className="overlay-stack__heat" src={src} alt={alt} overlay opacity={opacity} />
    </div>
  );
}

// Original shown cropped to the model's view (Resize 256 -> CenterCrop 224);
// click toggles to the full untouched image inside the same box.
function OriginalImage({ src, alt, className }) {
  const [expanded, setExpanded] = useState(false);
  if (!src) return null;
  return (
    <img
      className={`${className} original-crop${expanded ? ' is-expanded' : ''}`}
      src={src} alt={alt} loading="lazy"
      onClick={() => setExpanded((v) => !v)}
      title={expanded ? 'Full image — click to crop to model view' : 'Cropped to model view (224) — click for full image'}
    />
  );
}

function MiniImage({ caption, src, alt, missingText = 'Not available', variant = 'attribution', originalSrc }) {
  const resolvedSrc = resolveAssetUrl(src);
  return (
    <figure className="mini-image">
      <figcaption>{caption}</figcaption>
      {!resolvedSrc
        ? <div className="mini-image__missing">{missingText}</div>
        : variant === 'original'
          ? <OriginalImage className="mini-image__asset mini-image__asset--original" src={resolvedSrc} alt={alt} />
          : <Attribution className="mini-image__asset mini-image__asset--attribution" src={resolvedSrc} originalSrc={resolveAssetUrl(originalSrc)} alt={alt} />}
    </figure>
  );
}

function OverlayControl({ enabled, opacity, onToggle, onOpacity }) {
  return (
    <div className="overlay-control">
      <button
        type="button" className={`overlay-control__toggle${enabled ? ' is-on' : ''}`}
        onClick={onToggle} aria-pressed={enabled}
      >Overlay heatmap</button>
      {enabled && (
        <label className="overlay-control__opacity">
          Opacity
          <input
            type="range" min="0" max="1" step="0.05" value={opacity}
            onChange={(e) => onOpacity(Number(e.target.value))}
          />
        </label>
      )}
    </div>
  );
}

function ColorbarLegend() {
  return (
    <div className="colorbar-legend">
      <span className="colorbar-legend__title">Attribution</span>
      <div className="colorbar-legend__bar" role="img" aria-label="Attribution color scale from 0 to 1" />
      <div className="colorbar-legend__ticks" aria-hidden="true">
        <span>0</span>
        <span>0.25</span>
        <span>0.5</span>
        <span>0.75</span>
        <span>1</span>
      </div>
      <p className="colorbar-legend__note">Normalized relative to each image</p>
    </div>
  );
}

function MethodFigures({ method, outputs, imageId, originalSrc }) {
  const entries = resolveMethodEntries(method, outputs);
  if (!entries.length) return <MiniImage caption="Method not selected" missingText="—" />;
  return entries.map(([name, url]) => (
    <MiniImage key={name} caption={name} src={url} originalSrc={originalSrc} alt={`${name} for image ${imageId}`} />
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
  return (
    <div className={`prediction-badge prediction-badge--${isCorrect ? 'correct' : 'incorrect'}`}>
      <span className="prediction-badge__icon">{isCorrect ? '\u2713' : '\u2717'}</span>
      <span className="prediction-badge__text">
        Predicted: <strong>{predLabel}</strong>
        {' \u2014 '}
        <em>{isCorrect ? 'Correct' : 'Incorrect'}</em>
        {confidencePct ? ` \u2014 p(=${confidencePct})` : ''}
      </span>
    </div>
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

function SummaryStrip({ text }) {
  return <div className="summary-strip" role="status" aria-live="polite">{text}</div>;
}

function ModelStatsPanel({ model, dataset, stats }) {
  if (!model || !dataset || !stats) return null;

  const items = [
    { label: 'Samples', value: stats.total },
    { label: 'Correct', value: stats.correct },
    { label: 'Accuracy', value: formatMetricPercent(stats.accuracy) },
    { label: 'Macro Precision', value: formatMetricPercent(stats.macroPrecision) },
    { label: 'Macro Recall', value: formatMetricPercent(stats.macroRecall) },
    { label: 'Macro F1', value: formatMetricPercent(stats.macroF1) },
  ];

  return (
    <section className="model-stats" aria-label="Selected model metrics">
      <div className="model-stats__header">
        <div>
          <h2>Model Stats</h2>
          <p>{model} on {dataset}</p>
        </div>
      </div>
      <div className="model-stats__grid">
        {items.map((item) => (
          <article key={item.label} className="model-stats__item">
            <span className="model-stats__label">{item.label}</span>
            <strong className="model-stats__value">{item.value}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}

function ModeSwitcher({ value, onChange }) {
  const items = [
    { value: 'single', label: 'Single Image' },
    { value: 'model_grid', label: 'By Model' },
    { value: 'class_compare', label: 'By Class Across Models' },
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

function SearchableSelect({ label, value, items, onSelect, placeholder, disabled }) {
  const inputId = useId();
  const list = (items ?? []).map((i) => typeof i === 'string' ? { value: i, label: i } : i);
  const selectedLabel = value == null ? '' : list.find((i) => i.value === value)?.label ?? String(value);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');

  const q = normalize(query);
  const filtered = q ? list.filter((i) => !i.isHeader && normalize(i.label).includes(q)) : list;

  const commit = (v) => { onSelect(v); setOpen(false); setQuery(''); };

  return (
    <div className="tiny-form">
      <label className="tiny-form__label" htmlFor={inputId}>{label}</label>
      <div className="combo">
        <input
          id={inputId} className="combo__input"
          value={open ? query : selectedLabel}
          placeholder={placeholder} disabled={disabled}
          onFocus={() => setOpen(true)}
          onChange={(e) => { setOpen(true); setQuery(e.target.value); }}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setOpen(false);
            if (e.key === 'Enter') { const f = filtered.find((i) => !i.isHeader); if (f) commit(f.value); }
          }}
          onBlur={() => setOpen(false)}
        />
        {open && !disabled && (
          <div className="combo__list" onMouseDown={(e) => e.preventDefault()}>
            {filtered.length === 0 ? (
              <div className="combo__empty">No matches</div>
            ) : filtered.slice(0, 250).map((item) =>
              item.isHeader ? (
                <div key={item.value} className="combo__group-header">{item.label}</div>
              ) : (
                <button type="button" key={item.value} className="combo__option" onClick={() => commit(item.value)}>
                  {item.label}
                </button>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── View Components ────────────────────────────────────────── */

function SingleImageGallery({ imageData, labels }) {
  if (!imageData?.original) {
    return <EmptyState title="No image selected yet" description="Choose a model, dataset, class, and image to inspect." />;
  }

  const outputs = Object.entries(imageData.outputs ?? {});
  return (
    <section className="image-gallery">
      <div className="gallery-divider">
        <div>
          <h2>Images and Explanations</h2>
          <p>Original plus attribution maps grouped together.</p>
        </div>
      </div>
      <PredictionBadge prediction={imageData.prediction} classId={imageData.classId} labels={labels} />
      <div className="gallery-grid gallery-grid--single">
        <div className="image-card" style={{ '--delay': '80ms' }}>
          <div className="image-card__header"><h3>Image</h3></div>
          <OriginalImage className="image-card__image image-card__image--original" src={resolveAssetUrl(imageData.original)} alt="Original selection" />
        </div>
        {outputs.map(([method, url]) => (
          <div key={method} className="image-card">
            <div className="image-card__header"><h3>{method}</h3></div>
            <Attribution className="image-card__image image-card__image--attribution" src={resolveAssetUrl(url)} originalSrc={resolveAssetUrl(imageData.original)} alt={`${method} explanation`} />
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelGridView({ records, method, ready, labels }) {
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
            <MethodFigures method={method} outputs={record.outputs} imageId={record.imageId} originalSrc={record.originalUrl} />
          </div>
        </article>
      ))}
    </section>
  );
}

function ClassCompareView({ matrix, method, ready, labels }) {
  if (!ready) return <EmptyState title="Choose dataset and class" description="Pick a dataset and class to align the same image IDs across models." />;
  if (!matrix.rows.length) return <EmptyState title="No aligned rows found" description="No images are available for this class across the selected models." />;

  const style = { '--compare-columns': matrix.models.length + 1 };
  return (
    <section className="compare-matrix" aria-label="Class comparison matrix">
      <div className="compare-header" style={style}>
        <div className="compare-header__cell compare-header__cell--original">Original</div>
        {matrix.models.map((m) => <div key={m} className="compare-header__cell">{m}</div>)}
      </div>
      {matrix.rows.map((row) => {
        const origUrl = row.cells.find((c) => c.record?.originalUrl)?.record?.originalUrl ?? null;
        return (
          <div key={row.imageId} className="compare-row" style={style}>
            <article className="compare-cell compare-cell--original">
              <MiniImage caption={`Image ${row.imageId}`} src={origUrl} alt={`Original image ${row.imageId}`} missingText="Original unavailable" variant="original" />
            </article>
            {row.cells.map((cell) => (
              <article key={`${row.imageId}__${cell.model}`} className="compare-cell" data-model={cell.model}>
                <PredictionBadge prediction={cell.record?.prediction} classId={row.classId} labels={labels} />
                <MethodFigures method={method} outputs={cell.record?.outputs ?? {}} imageId={row.imageId} originalSrc={cell.record?.originalUrl} />
              </article>
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

function ModelForm({ outputStructure }) {
  const modelsStruct = outputStructure?.models ?? EMPTY_OBJ;

  const [vs, setVs] = useState(() => ({
    mode: 'single', model: null, dataset: null, classId: null, imageId: null, method: null,
    ...readStateFromUrl(),
  }));
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [imgCache, setImgCache] = useState({});
  const [lblCache, setLblCache] = useState({});
  const [dsStatus, setDsStatus] = useState({});

  const patch = (update) => { const next = { ...vs, ...update }; setVs(next); writeStateToUrl(next); };

  const modelOptions = useMemo(() => Object.keys(modelsStruct).sort(), [modelsStruct]);

  const effectiveModel = resolveSelection(modelOptions, vs.model);

  const datasetOptions = useMemo(() => {
    if (!effectiveModel) return [];
    return Object.keys(modelsStruct[effectiveModel]?.datasets ?? {}).sort(compareMixedIds);
  }, [effectiveModel, modelsStruct]);

  const effectiveDataset = resolveSelection(datasetOptions, vs.dataset);

  // Load dataset metadata (images structure + labels)
  useEffect(() => {
    if (!effectiveDataset) return;
    const controller = new AbortController();
    const { signal } = controller;
    const ds = effectiveDataset;
    const updStatus = (fields) =>
      !signal.aborted && setDsStatus((p) => ({ ...p, [ds]: { ...p[ds], ...fields } }));

    if (!imgCache[ds]) {
      updStatus({ imagesLoading: true, imagesError: null });
      fetchJson(`${ds}/${ds}_structure.json`, { signal })
        .then((data) => { if (!signal.aborted) setImgCache((p) => ({ ...p, [ds]: data })); })
        .catch((e) => { if (e.name !== 'AbortError') updStatus({ imagesError: 'Failed to load.' }); })
        .finally(() => updStatus({ imagesLoading: false }));
    }

    if (!lblCache[ds]) {
      updStatus({ labelsLoading: true, labelsError: null });
      fetchJson('imagenet-mini/imagenet-1k-id2label.json', { signal })
        .then((data) => { if (!signal.aborted) setLblCache((p) => ({ ...p, [ds]: data })); })
        .catch((e) => {
          if (e.name !== 'AbortError') {
            setLblCache((p) => ({ ...p, [ds]: {} }));
            updStatus({ labelsError: 'Failed to load.' });
          }
        })
        .finally(() => updStatus({ labelsLoading: false }));
    }

    return () => controller.abort();
  }, [effectiveDataset, imgCache, lblCache]);

  const imageRecords = useMemo(
    () => buildImageRecords(outputStructure, imgCache, lblCache),
    [outputStructure, imgCache, lblCache]
  );

  const modelMetrics = useMemo(
    () => getModelMetrics(outputStructure),
    [outputStructure]
  );

  const classOptions = useMemo(() => {
    if (!effectiveDataset || !effectiveModel) return [];
    const labels = lblCache[effectiveDataset] ?? {};
    const classIds = Object.keys(modelsStruct[effectiveModel]?.datasets?.[effectiveDataset]?.classes ?? {});
    return classIds.sort(compareMixedIds).map((id) => ({ value: id, label: `${id} - ${labels[id] ?? id}` }));
  }, [effectiveDataset, effectiveModel, lblCache, modelsStruct]);

  const effectiveClassId = resolveSelection(classOptions, vs.classId);

  const imageOptions = useMemo(() => {
    if (!effectiveModel || !effectiveDataset || !effectiveClassId) return [];
    const images = modelsStruct[effectiveModel]?.datasets?.[effectiveDataset]?.classes?.[effectiveClassId]?.images ?? {};
    const filenames = imgCache[effectiveDataset]?.[effectiveClassId] ?? [];
    return Object.keys(images).sort(compareMixedIds).map((id) => ({
      value: id, label: filenames[id] ? `${id} - ${filenames[id]}` : id,
    }));
  }, [effectiveModel, effectiveDataset, effectiveClassId, modelsStruct, imgCache]);

  const effectiveImageId = vs.mode === 'single' ? resolveSelection(imageOptions, vs.imageId) : null;

  const methodOptions = useMemo(() => {
    if (!effectiveDataset) return [];
    const methods = new Set();
    for (const r of imageRecords) if (r.dataset === effectiveDataset) Object.keys(r.outputs).forEach((m) => methods.add(m));
    const sorted = [...methods].sort();

    const grouped = {};
    for (const m of sorted) { const c = categorizeMethod(m); (grouped[c] ??= []).push(m); }
    const cats = [...Object.keys(METHOD_CATEGORIES), 'other'].filter((c) => grouped[c]?.length);

    if (cats.length <= 1) {
      return [{ value: ALL_METHODS, label: 'All Methods' }, ...sorted.map((m) => ({ value: m, label: m }))];
    }
    const result = [{ value: ALL_METHODS, label: 'All Methods' }];
    for (const cat of cats) {
      const lbl = METHOD_CATEGORIES[cat]?.label ?? 'Other';
      result.push({ value: `__header_${cat}__`, label: lbl, isHeader: true });
      result.push({ value: `__group_${cat}__`, label: `All ${lbl}` });
      for (const m of grouped[cat]) result.push({ value: m, label: m });
    }
    return result;
  }, [imageRecords, effectiveDataset]);

  const selectableMethods = methodOptions.filter((m) => !m.isHeader);
  const effectiveMethod = resolveSelection(selectableMethods, vs.method);

  const singleImageData = useMemo(() => {
    if (vs.mode !== 'single' || !effectiveModel || !effectiveDataset || !effectiveClassId || !effectiveImageId) return null;
    const r = imageRecords.find((i) =>
      i.model === effectiveModel && i.dataset === effectiveDataset && i.classId === effectiveClassId && i.imageId === effectiveImageId
    );
    if (!r) return null;
    return {
      original: r.originalUrl,
      outputs: Object.fromEntries(resolveMethodEntries(effectiveMethod, r.outputs)),
      prediction: r.prediction,
      classId: r.classId,
    };
  }, [imageRecords, vs.mode, effectiveModel, effectiveDataset, effectiveClassId, effectiveImageId, effectiveMethod]);

  const modelGridRecords = useMemo(() => {
    if (!effectiveModel || !effectiveDataset) return [];
    return imageRecords.filter((r) =>
      r.model === effectiveModel && r.dataset === effectiveDataset &&
      (!effectiveClassId || r.classId === effectiveClassId)
    );
  }, [imageRecords, effectiveModel, effectiveDataset, effectiveClassId]);

  const classCompareMatrix = useMemo(
    () => getClassCompareMatrix(imageRecords, { dataset: effectiveDataset, classId: effectiveClassId, method: effectiveMethod }),
    [imageRecords, effectiveDataset, effectiveClassId, effectiveMethod]
  );

  const selectedModelStats = effectiveModel && effectiveDataset
    ? modelMetrics.byModelAndDataset[effectiveModel]?.[effectiveDataset] ?? null
    : null;

  const hasContent =
    (vs.mode === 'single' && Boolean(singleImageData?.original)) ||
    (vs.mode === 'model_grid' && modelGridRecords.length > 0) ||
    (vs.mode === 'class_compare' && classCompareMatrix.rows.length > 0);

  const dsInfo = effectiveDataset ? dsStatus[effectiveDataset] ?? {} : {};
  const isLoading = dsInfo.imagesLoading || dsInfo.labelsLoading;

  const handleModeChange = (mode) => {
    const next = { ...vs, mode };
    if (mode !== 'single') next.imageId = null;
    setVs(next);
    writeStateToUrl(next);
  };

  const methodLabel = !effectiveMethod
    ? 'no method selected'
    : selectableMethods.find((m) => m.value === effectiveMethod)?.label ?? effectiveMethod;

  const summaryText = (() => {
    if (vs.mode === 'single') {
      if (!singleImageData?.original) return 'Single image mode. Choose a full selection to begin.';
      return `1 image selected · ${Object.keys(singleImageData.outputs ?? {}).length} attribution methods available`;
    }
    if (vs.mode === 'model_grid') {
      if (!effectiveModel || !effectiveDataset) return 'By Model mode. Select model and dataset.';
      const cls = effectiveClassId ? `class ${effectiveClassId}` : 'all classes';
      return `${modelGridRecords.length} images · ${effectiveModel} · ${cls} · ${methodLabel}`;
    }
    if (!effectiveDataset || !effectiveClassId) return 'By Class Across Models mode. Select dataset and class.';
    return `${classCompareMatrix.rows.length} image rows · ${classCompareMatrix.models.length} models · ${methodLabel}`;
  })();

  return (
    <OverlayContext.Provider value={{ enabled: overlay, opacity: overlayOpacity }}>
    <div className={`viewer-layout${panelCollapsed ? ' viewer-layout--collapsed' : ''}`}>
      <aside className={`controls-panel${panelCollapsed ? ' controls-panel--collapsed' : ''}`}>
        {panelCollapsed ? (
          <button className="panel-toggle" onClick={() => setPanelCollapsed(false)} title="Expand controls">
            <span className="panel-toggle__icon">|||</span>
            <span className="panel-toggle__label">Controls</span>
          </button>
        ) : (
          <>
            <div className="panel-header">
              <div className="panel-title">
                <h2>Viewer Controls</h2>
                <p>Switch analysis mode and filter the image space.</p>
              </div>
              {hasContent && (
                <button className="panel-collapse-btn" onClick={() => setPanelCollapsed(true)} title="Collapse panel">x</button>
              )}
            </div>
            <div className="panel-fields">
              <ModeSwitcher value={vs.mode} onChange={handleModeChange} />

              {vs.mode !== 'class_compare' && (
                <SearchableSelect
                  label="Model" value={effectiveModel} items={modelOptions}
                  onSelect={(v) => patch({ model: v, classId: null, imageId: null })}
                  placeholder="Search model" disabled={!modelOptions.length}
                />
              )}

              <SearchableSelect
                label="Dataset" value={effectiveDataset}
                items={datasetOptions}
                onSelect={(v) => patch({ dataset: v })}
                placeholder="Search dataset"
                disabled={!datasetOptions.length}
              />

              <SearchableSelect
                label="Class" value={effectiveClassId} items={classOptions}
                onSelect={(v) => patch({ classId: v, imageId: null })}
                placeholder={isLoading ? 'Loading class metadata...' : 'Search class'}
                disabled={!effectiveDataset || !classOptions.length}
              />

              {vs.mode === 'single' && (
                <SearchableSelect
                  label="Image" value={effectiveImageId} items={imageOptions}
                  onSelect={(v) => patch({ imageId: v })}
                  placeholder="Search image id / filename"
                  disabled={!effectiveClassId}
                />
              )}

              <SearchableSelect
                label="Method" value={effectiveMethod} items={methodOptions}
                onSelect={(v) => patch({ method: v })}
                placeholder="Search method"
                disabled={!effectiveDataset || !methodOptions.some((m) => !m.isHeader)}
              />

              {(dsInfo.imagesError || dsInfo.labelsError) && (
                <div className="status-message" role="status">Some dataset metadata failed to load.</div>
              )}
            </div>
            <OverlayControl
              enabled={overlay} opacity={overlayOpacity}
              onToggle={() => setOverlay((v) => !v)} onOpacity={setOverlayOpacity}
            />
            <ColorbarLegend />
          </>
        )}
      </aside>

      <main className="viewer-content">
        <header className="page-header">
          <div>
            <h1 className="page-title"><span>Model Explainability</span> <span>Viewer</span></h1>
            <p className="page-subtitle">Explore model behavior by image, by model, or by class across models.</p>
          </div>
          <ThemeToggle />
        </header>

        <SummaryStrip text={summaryText} />

        {vs.mode === 'model_grid' && (
          <ModelStatsPanel model={effectiveModel} dataset={effectiveDataset} stats={selectedModelStats} />
        )}

        {vs.mode === 'single' && <SingleImageGallery imageData={singleImageData} labels={lblCache[effectiveDataset]} />}
        {vs.mode === 'model_grid' && (
          <ModelGridView records={modelGridRecords} method={effectiveMethod} ready={Boolean(effectiveModel && effectiveDataset)} labels={lblCache[effectiveDataset]} />
        )}
        {vs.mode === 'class_compare' && (
          <ClassCompareView matrix={classCompareMatrix} method={effectiveMethod}
            ready={Boolean(effectiveDataset && effectiveClassId)} labels={lblCache[effectiveDataset]} />
        )}
      </main>
    </div>
    </OverlayContext.Provider>
  );
}

async function loadOutputStructure(signal) {
  const manifest = await fetchJson('outputs/manifest.json', { signal });

  const entries = Object.entries(manifest?.runs ?? {}).flatMap(([model, datasets]) =>
    Object.entries(datasets ?? {}).map(([dataset, paths]) => ({ model, dataset, paths }))
  );

  const runPayloads = Object.fromEntries(
    await Promise.all(
      entries.map(async ({ model, dataset, paths }) => {
        const [images, summary] = await Promise.all([
          fetchJson(paths.images),
          fetchJson(paths.summary),
        ]);
        return [`${model}::${dataset}`, { images, summary }];
      })
    )
  );

  return buildLegacyOutputStructure(manifest, runPayloads);
}

function App() {
  const [outputStructure, setOutputStructure] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
  const controller = new AbortController();

  loadOutputStructure(controller.signal)
    .then(setOutputStructure)
    .catch((e) => {
      if (e.name !== 'AbortError') setError(e);
    });

  return () => controller.abort();
}, []);

  if (error) return <div className="app-status">Error loading outputs metadata.</div>;
  if (!outputStructure) return <div className="app-status">Loading viewer data...</div>;

  return <div className="app-shell"><ModelForm outputStructure={outputStructure} /></div>;
}

export default App;
