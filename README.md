# NeuralAtlas

Generate attribution maps for image classification models and serve them through a React viewer.

## Backend (Python)

The backend is managed with [uv](https://docs.astral.sh/uv/). Dependencies and the
Python version are pinned in `pyproject.toml` / `uv.lock` / `.python-version` 

1. Install uv 
2. Create the local environment and install locked dependencies:
   ```bash
   uv sync
   ```
   This creates a project-local `.venv/` and installs the exact versions from `uv.lock`.
3. Run the attribution pipeline:
   ```bash
   uv run python main.py --help
   ```

   The default dataset is `imagenet-pico`. You can select another dataset
   under `interpretability-viewer/public/` (see "Bringing your own dataset") with `--dataset`:

   ```bash
   uv run python main.py --dataset imagenet-pico-ai --num-samples 20
   ```

### Bringing your own dataset

A dataset is a folder under `interpretability-viewer/public/` with an ImageFolder tree
and a `dataset.json` next to it:

```
interpretability-viewer/public/my-dogs/
  dataset.json
  images/golden_retriever/*.jpg
  images/tench/*.jpg
```

```json
{
  "schema_version": 1,
  "title": "My dogs",
  "label_space": "imagenet-1k",
  "images_dir": "images",
  "classes": { "golden_retriever": 207, "tench": 0 }
}
```

`label_space` names a file in `interpretability-viewer/public/label_spaces/` whose
`labels` list is indexed by model output; `classes` maps each class folder to its index
there. Leave `classes` out when the folders are already named after their index
(`0/`, `207/`, ...). Then run `main.py --dataset my-dogs`. Classes outside an existing
label space need a new label space file and a model that predicts it (see below).

### Bringing your own model

Every model `main.py --model <id>` can run has a spec at `model_specs/<id>.json`:

```json
{
  "schema_version": 1,
  "architecture": "resnet18",
  "weights": "my-dogs.pt",
  "label_space": "my-dogs",
  "preprocess": {"resize": 256, "crop": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
}
```

`architecture` is a torchvision builder name. `weights` is `"DEFAULT"` for torchvision's
pretrained ImageNet weights, or a `state_dict` path relative to `model_specs/`; the head is
then sized to the label space. A model only runs on datasets labelled in its label space.

## Frontend (React + Vite)

To run in development mode:

1. Clone the repository
2. Navigate to the `interpretability-viewer` directory
3. Install dependencies with `npm install`
4. Start the development server with `npm run dev`
