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

   The default dataset is `imagenet-pico-ai`. You can select another dataset
   directory under `interpretability-viewer/public/` with `--dataset`:

   ```bash
   uv run python main.py --dataset imagenet-pico --num-samples 20
   ```

### Full sweep on a remote GPU box

`scripts/run_sweep.py` runs the whole matrix (many models x the whole dataset) without
ever holding more than one chunk of attribution images on disk. It is idempotent and
resumable: before each model it restores that run's JSON checkpoint from Hugging Face,
so a fresh GPU box can continue an existing run.

```bash
cp .env.example .env          # fill in HF_TOKEN
uv run python scripts/run_sweep.py --dry-run     # print the plan, touch nothing
uv run python scripts/run_sweep.py --chunk 100   # the real run
```

Use `--methods` to recompute only selected attribution methods. A metadata-only run
calculates and merges metrics without rendering image files, then uploads a JSON-only
checkpoint after each chunk:

```bash
uv run python scripts/run_sweep.py \
  --methods Occlusion GradientShap IntegratedGradients \
  --metrics fidelity --recompute --metadata-only
```

Passing `--metrics` with no values makes metadata-only recompute predictions and
aggregate accuracy without constructing attribution methods:

```bash
uv run python scripts/run_sweep.py --metadata-only --metrics
```

Both metadata-only and regular image sweeps upload one checkpoint per chunk; only the
regular sweep includes rendered attribution files.

Per (model, chunk) it runs the pipeline, uploads `outputs/images/<model>__<dataset>__*`
plus only that model/dataset's `images.json` and `summary.json`, then deletes the
uploaded images locally. GPU workers never publish shared catalogs or `manifest.json`,
so different models can run concurrently on different servers without overwriting
global metadata.

If the dataset is missing, the current fallback builds it with
`scripts/download_nano_imagenet.py` from Kaggle (~4 GB).

Because it shells out one process per chunk, it survives a crash in any single
model/chunk: the failing model is abandoned and the sweep moves on to the next one.

Run it under `tmux`/`nohup` — a full sweep is measured in days.

Once one or more workers have uploaded checkpoints, rebuild the global JSON kept in
GitHub from every per-model HF repo:

```bash
uv run python scripts/sync_hf_metadata.py --dry-run  # inspect repositories and runs
uv run python scripts/sync_hf_metadata.py            # download JSON and rebuild indexes
```

Repositories are discovered from the `HF_ATTRIBUTIONS_REPO` prefix, so this does not
need a hardcoded model list. Each run in the rebuilt manifest includes a `base_url`
pinned to the HF commit that supplied its JSON. The frontend appends the class id and
attribution filename to that URL; older runs without it keep using the legacy shared
repository route.

### AI dataset

The paired AI dataset generation feature lives in the backend and can be run with:

```bash
uv run python -m backend.ai_dataset --help
```

## Frontend (React + Vite)

To run in development mode:

1. Clone the repository
2. Navigate to the `interpretability-viewer` directory
3. Install dependencies with `npm install`
4. Start the development server with `npm run dev`
