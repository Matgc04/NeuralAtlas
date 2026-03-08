import { useEffect, useId, useMemo, useState } from 'react';
import './App.css';

const ALL_METHODS = '__all_methods__';
const EMPTY_OBJ = {};

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


function buildImageRecords(outputStructure, imgCache, lblCache) {
  const records = [];
  for (const [model, { datasets = {} }] of Object.entries(outputStructure?.models ?? {})) {
    for (const [dataset, { classes = {} }] of Object.entries(datasets)) {
      const imgLookup = imgCache?.[dataset] ?? {};
      const lblLookup = lblCache?.[dataset] ?? {};
      for (const [classId, { images = {} }] of Object.entries(classes)) {
        const filenames = imgLookup[classId] ?? [];
        const classLabel = lblLookup[classId] ?? classId;
        for (const [imageId, { outputs = {}, prediction = null } = {}] of Object.entries(images)) {
          const filename = filenames[imageId] ?? null;
          records.push({
            model, dataset, classId, classLabel, imageId, filename,
            originalUrl: filename ? `/${dataset}/val/${classId}/${filename}` : null,
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

/* ── Reusable UI Components ─────────────────────────────────── */

const COLORBAR_SRC = '/outputs/master_colorbar_jet.webp';

function MiniImage({
  caption,
  src,
  alt,
  missingText = 'Not available',
  showColorbar = false,
  variant = 'attribution',
}) {
  return (
    <figure className="mini-image">
      <figcaption>{caption}</figcaption>
      {src
        ? <img className={`mini-image__asset mini-image__asset--${variant}`} src={src} alt={alt} loading="lazy" />
        : <div className="mini-image__missing">{missingText}</div>}
      {showColorbar && src && <img className="colorbar" src={COLORBAR_SRC} alt="" />}
    </figure>
  );
}

function MethodFigures({ method, outputs, imageId }) {
  const entries = resolveMethodEntries(method, outputs);
  if (!entries.length) return <MiniImage caption="Method not selected" missingText="—" />;
  return entries.map(([name, url]) => (
    <MiniImage key={name} caption={name} src={url} alt={`${name} for image ${imageId}`} showColorbar />
  ));
}

function PredictionBadge({ prediction, classId, labels }) {
  if (!prediction) return null;
  const predId = prediction.predicted_class_id;
  const predLabel = labels?.[predId] ?? `Class ${predId}`;
  const isCorrect = String(predId) === String(classId);
  return (
    <div className={`prediction-badge prediction-badge--${isCorrect ? 'correct' : 'incorrect'}`}>
      <span className="prediction-badge__icon">{isCorrect ? '\u2713' : '\u2717'}</span>
      <span className="prediction-badge__text">
        Predicted: <strong>{predLabel}</strong>
        {' \u2014 '}
        <em>{isCorrect ? 'Correct' : 'Incorrect'}</em>
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
      <div className="gallery-grid">
        <div className="image-card image-card--featured" style={{ '--delay': '80ms' }}>
          <div className="image-card__header"><h3>Original Image</h3></div>
          <img className="image-card__image image-card__image--original" src={imageData.original} alt="Original selection" />
        </div>
        {outputs.map(([method, url]) => (
          <div key={method} className="image-card">
            <div className="image-card__header"><h3>{method}</h3></div>
            <img className="image-card__image image-card__image--attribution" src={url} alt={`${method} explanation`} loading="lazy" />
            <img className="colorbar" src={COLORBAR_SRC} alt="" />
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
            <MethodFigures method={method} outputs={record.outputs} imageId={record.imageId} />
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
                <MethodFigures method={method} outputs={cell.record?.outputs ?? {}} imageId={row.imageId} />
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

  const [vs, setVs] = useState({
    mode: 'single', model: null, dataset: null, classId: null, imageId: null, method: null,
  });
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [imgCache, setImgCache] = useState({});
  const [lblCache, setLblCache] = useState({});
  const [dsStatus, setDsStatus] = useState({});

  const patch = (update) => setVs((p) => ({ ...p, ...update }));

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
    let cancelled = false;
    const ds = effectiveDataset;
    const updStatus = (fields) => !cancelled && setDsStatus((p) => ({ ...p, [ds]: { ...p[ds], ...fields } }));

    if (!imgCache[ds]) {
      updStatus({ imagesLoading: true, imagesError: null });
      fetch(`/${ds}/${ds}_structure.json`)
        .then((r) => r.json())
        .then((data) => { if (!cancelled) setImgCache((p) => ({ ...p, [ds]: data })); })
        .catch(() => updStatus({ imagesError: 'Failed to load.' }))
        .finally(() => updStatus({ imagesLoading: false }));
    }

    if (!lblCache[ds]) {
      updStatus({ labelsLoading: true, labelsError: null });
      fetch('/imagenet-mini/imagenet-1k-id2label.json')
        .then((r) => r.json())
        .then((data) => { if (!cancelled) setLblCache((p) => ({ ...p, [ds]: data })); })
        .catch(() => { if (!cancelled) { setLblCache((p) => ({ ...p, [ds]: {} })); updStatus({ labelsError: 'Failed to load.' }); } })
        .finally(() => updStatus({ labelsLoading: false }));
    }

    return () => { cancelled = true; };
  }, [effectiveDataset, imgCache, lblCache]);

  const imageRecords = useMemo(
    () => buildImageRecords(outputStructure, imgCache, lblCache),
    [outputStructure, imgCache, lblCache]
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

  const hasContent =
    (vs.mode === 'single' && Boolean(singleImageData?.original)) ||
    (vs.mode === 'model_grid' && modelGridRecords.length > 0) ||
    (vs.mode === 'class_compare' && classCompareMatrix.rows.length > 0);

  const dsInfo = effectiveDataset ? dsStatus[effectiveDataset] ?? {} : {};
  const isLoading = dsInfo.imagesLoading || dsInfo.labelsLoading;

  const handleModeChange = (mode) => {
    setVs((p) => {
      const next = { ...p, mode };
      if (mode !== 'single') next.imageId = null;
      return next;
    });
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
                onSelect={(v) => patch({ dataset: v, classId: null, imageId: null })}
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
          </>
        )}
      </aside>

      <main className="viewer-content">
        <header className="page-header">
          <div>
            <h1 className="page-title"><span>Model Explainability</span> <span>Viewer</span></h1>
            <p className="page-subtitle">Explore model behavior by image, by model, or by class across models.</p>
          </div>
        </header>

        <SummaryStrip text={summaryText} />

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
  );
}

function App() {
  const [outputStructure, setOutputStructure] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/outputs/outputs_structure.json').then((r) => r.json()).then(setOutputStructure).catch(setError);
  }, []);

  if (error) return <div className="app-status">Error loading outputs metadata.</div>;
  if (!outputStructure) return <div className="app-status">Loading viewer data...</div>;

  return <div className="app-shell"><ModelForm outputStructure={outputStructure} /></div>;
}

export default App;
