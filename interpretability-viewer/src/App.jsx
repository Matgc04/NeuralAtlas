import { useState, useEffect } from 'react'
import './App.css'

function TinyForm({ label, value, onChange, items }) {
  const fieldId = label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  return (
    <div className="tiny-form">
      <label className="tiny-form__label" htmlFor={fieldId}>{label}</label>
      <select id={fieldId} className="tiny-form__select" value={value ?? ""} onChange={onChange}>
        <option value="" disabled>Select {label.slice(0,-1)}</option>
        {items.map((it, i) => <option key={i} value={it}>{it}</option>)}
      </select>
    </div>
  );
}

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
  options['image'] = (model && dataset && classId) ? Object.keys(datasetImagesStructure[classId]) : [];

  const handleModelChange = (e) => {
    const value = e.target.value;
    setModel(value);
    setDataset(null);
    setClassId(null);
    setImageId(null);
    resetImageData();
  };

  const handleDatasetChange = (e) => {
    const value = e.target.value;
    setDataset(value);
    setClassId(null);
    setImageId(null);
    resetImageData();

    fetch(`/${value}/${value}_structure.json`)
      .then(r => r.json())
      .then(setDatasetImagesStructure)
      .catch(err => console.error('Error loading dataset structure:', err));
  };

  const handleClassChange = (e) => {
    const value = e.target.value;
    setClassId(value);
    setImageId(null);
    resetImageData();
  };

  const handleImageChange = (e) => {
    const value = e.target.value;
    setImageId(value);


    const fetchBlobUrl = async (path) => {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`Failed request ${path}: ${response.status}`);
      }
      const blob = await response.blob();
      return URL.createObjectURL(blob);
    };

    const originalFilename = datasetImagesStructure?.[classId]?.[value];
    const originalFileUrl = `/${dataset}/val/${classId}/${originalFilename}`;

    
    const methods = outputStructure?.[model]?.[dataset]?.[classId] ?? {};
    const methodNames = Object.keys(methods);
    const derivedImageName = `${value}.jpg`;

    const loadImages = async () => {
      try {
        const originalPromise = fetchBlobUrl(`${originalFileUrl}`);
        const outputsPromise = methodNames.length
          ? Promise.all(methodNames.map(async (method) => {
              const url = await fetchBlobUrl(`/outputs/${model}/${dataset}/${classId}/${method}/${derivedImageName}`);
              return [method, url];
            }))
          : Promise.resolve([]);

        const [originalUrl, outputs] = await Promise.all([originalPromise, outputsPromise]);

        setImageData({
          original: originalUrl,
          outputs: Object.fromEntries(outputs)
        });
      } catch (err) {
        console.error('Error loading image assets:', err);
      }
    };

    if (classId && value && model && dataset) {
      loadImages();
    }

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
              <TinyForm 
                label="Model:"
                onChange={handleModelChange}
                items={options['model']}
                value={model}
              />
              <TinyForm 
                label="Dataset:"
                onChange={handleDatasetChange}
                items={options['dataset']}
                value={dataset}
              />
              <TinyForm 
                label="Class:"
                onChange={handleClassChange}
                items={options['class']}
                value={classId}
              />
              <TinyForm 
                label="Image:"
                onChange={handleImageChange}
                items={options['image']}
                value={imageId}
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
