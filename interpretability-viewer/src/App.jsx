import { useState, useEffect, useMemo, useDeferredValue, useRef } from 'react'
import './App.css'

function ImageGallery({ imageData }) {
  if (!imageData?.original) {
    return (
      <div className="image-gallery image-gallery--empty">
        <div className="empty-state">
          <h3>No image selected yet</h3>
          <p>Choose a model, dataset, class, and image to explore the original and its explanations.</p>
        </div>
      </div>
    );
  }

  const outputs = Object.entries(imageData.outputs ?? {});

  return (
    <div className="image-gallery">
      <div className="gallery-divider">
        <div>
          <h2>Images & Explanations</h2>
          <p>Original + attribution maps grouped together.</p>
        </div>
      </div>
      <div className="gallery-grid">
        <div className="image-card image-card--featured" style={{ "--delay": "80ms" }}>
          <div className="image-card__header">
            <h3>Original Image</h3>
            <span className="badge">Original</span>
          </div>
          <img className="image-card__image" src={imageData.original} alt="Original selection" />
        </div>
        {outputs.map(([method, url], index) => (
          <div key={method} className="image-card" style={{ "--delay": `${160 + index * 80}ms` }}>
            <div className="image-card__header">
              <h3>{method}</h3>
              <span className="badge badge--muted">Explanation</span>
            </div>
            <img className="image-card__image" src={url} alt={`${method} explanation`} />
          </div>
        ))}
      </div>
    </div>
  );
}

function normalize(s) {
  return String(s ?? "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

function SearchableSelect({ label, value, items, onSelect, placeholder, disabled }) {
  const fieldId = label.toLowerCase().replace(/[^a-z0-9]+/g, "-");

  const list = (items ?? []).map((it) =>
    typeof it === "string" ? { value: it, label: it } : it
  );

  const selectedLabel =
    value == null ? "" : (list.find((x) => x.value === value)?.label ?? String(value));

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const q = normalize(query);
  const filtered = q ? list.filter((it) => normalize(it.label).includes(q)) : list;

  const commit = (val) => {
    onSelect(val);
    setOpen(false);
    setQuery("");
  };

  return (
    <div className="tiny-form">
      <label className="tiny-form__label" htmlFor={fieldId}>{label}</label>

      <div className="combo">
        <input
          id={fieldId}
          className="combo__input"
          value={open ? query : selectedLabel}
          placeholder={placeholder}
          disabled={disabled}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            setOpen(true);
            setQuery(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") setOpen(false);
            if (e.key === "Enter" && filtered[0]) commit(filtered[0].value);
          }}
          onBlur={() => setOpen(false)}
        />

        {open && !disabled && (
          <div
            className="combo__list"
            onMouseDown={(e) => e.preventDefault()} // prevents blur when clicking options
          >
            {filtered.length === 0 ? (
              <div className="combo__empty">No matches</div>
            ) : (
              filtered.slice(0, 200).map((it) => (
                <button
                  type="button"
                  key={it.value}
                  className="combo__option"
                  onClick={() => commit(it.value)}
                >
                  {it.label}
                </button>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}



function ModelForm({outputStructure}) {

  console.log(outputStructure);

  const [model, setModel] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [datasetImagesStructure, setDatasetImagesStructure] = useState(null);
  const [classId, setClassId] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [panelCollapsed, setPanelCollapsed] = useState(false);

  // Auto-collapse when image is selected
  const hasSelection = imageData?.original != null;

  const resetImageData = () => setImageData(null);

  console.log(`Model: ${model}, Dataset: ${dataset}, Class: ${classId}, Image: ${imageId}`);

  const options = {
    'model': [],
    'dataset': [],
    'class': [],
    'image': []
  }

  options['model'] = Object.keys(outputStructure);
  options['dataset'] = model ? Object.keys(outputStructure[model]) : [];
  options['class'] = (model && dataset) ? Object.keys(outputStructure[model][dataset]) : [];
  options['image'] = (model && dataset && classId && datasetImagesStructure?.[classId])
  ? Object.keys(datasetImagesStructure[classId])
  : [];

  const imageItems = model && dataset && classId && datasetImagesStructure?.[classId]
  ? Object.entries(datasetImagesStructure[classId]).map(([id, filename]) => ({
      value: id,
      label: `${id} — ${filename}`,
    }))
  : [];

  const handleModelSelect = (value) => {
    setModel(value);
    setDataset(null);
    setClassId(null);
    setImageId(null);
    resetImageData();
};

const handleDatasetSelect = (value) => {
  setDataset(value);
  setClassId(null);
  setImageId(null);
  resetImageData();

  fetch(`/${value}/${value}_structure.json`)
    .then(r => r.json())
    .then(setDatasetImagesStructure)
    .catch(err => console.error('Error loading dataset structure:', err));
};

const handleClassSelect = (value) => {
  setClassId(value);
  setImageId(null);
  resetImageData();
};

 const handleImageSelect = (value) => {
  setImageId(value);

  const originalFilename = datasetImagesStructure?.[classId]?.[value];
  const original = `/${dataset}/val/${classId}/${originalFilename}`;

  const methods = outputStructure?.[model]?.[dataset]?.[classId] ?? {};
  const outputs = Object.fromEntries(
    Object.keys(methods).map((method) => [
      method,
      `/outputs/${model}/${dataset}/${classId}/${method}/${value}.jpg`,
    ])
  );

  setImageData({ original, outputs });
};

  return (
    <div className={`viewer-layout ${panelCollapsed ? 'viewer-layout--collapsed' : ''}`}>
      <aside className={`controls-panel ${panelCollapsed ? 'controls-panel--collapsed' : ''}`}>
        {panelCollapsed ? (
          <button 
            className="panel-toggle" 
            onClick={() => setPanelCollapsed(false)}
            title="Expand controls"
          >
            <span className="panel-toggle__icon">☰</span>
            <span className="panel-toggle__label">Controls</span>
          </button>
        ) : (
          <>
            <div className="panel-header">
              <div className="panel-title">
                <h2>Viewer Controls</h2>
                <p>Set the model, dataset, class, and image you want to inspect.</p>
              </div>
              {hasSelection && (
                <button 
                  className="panel-collapse-btn" 
                  onClick={() => setPanelCollapsed(true)}
                  title="Collapse panel"
                >
                  ✕
                </button>
              )}
            </div>
            <div className="panel-fields">
              <SearchableSelect
                label="Model:"
                value={model}
                items={options['model']}
                onSelect={handleModelSelect}
                placeholder="Search model…"
              />

            <SearchableSelect
              label="Dataset:"
              value={dataset}
              items={options['dataset']}
              onSelect={handleDatasetSelect}
              placeholder="Search dataset…"
              disabled={!model}
            />

            <SearchableSelect
              label="Class:"
              value={classId}
              items={options['class']}
              onSelect={handleClassSelect}
              placeholder="Search class…"
              disabled={!model || !dataset}
            />

            <SearchableSelect
              label="Image:"
              value={imageId}
              items={imageItems}
              onSelect={handleImageSelect}
              placeholder="Search image id / filename…"
              disabled={!classId}
            />
            </div>
          </>
        )}
      </aside>
      <main className="viewer-content">
        <header className="page-header">
          <div>
            <h1 className="page-title">
              <span>Model Explainability</span>
              <span>Viewer</span>
            </h1>
            <p className="page-subtitle">Explore what the network learns with multiple attribution methods.</p>
          </div>
        </header>
        <ImageGallery imageData={imageData} />
      </main>
    </div>
  )
}

function App() {

  function loging(){console.log("Cargando estructura de salidas...")}

  const [outputStructure, setOutputStructure] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/outputs/outputs_structure.json")
      .then(r => r.json())
      .then(setOutputStructure)
      .then(loging)
      .catch(setError);
  }, []);

  if (error) return <div>Error cargando datos</div>;

  if (!outputStructure) return <div>Cargando…</div>;

  return (
    <div className="app-shell">
      <ModelForm 
        outputStructure={outputStructure}
      />
    </div>
  )
}

export default App
