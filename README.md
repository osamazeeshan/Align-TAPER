# MM-VLM-TT

Multimodal (vision–language) experiments around CLIP-style models: test-time prompt tuning, AU-related adapters, and zero-shot evaluation on affect / pain / stress-style benchmarks.

## Environment

Install dependencies from **`requirements.txt`** (from the repository root):

```bash
pip install -r requirements.txt
```

For a **GPU** build of PyTorch, install the matching `torch` / `torchvision` / `torchaudio` wheels for your CUDA version first (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)), then run `pip install -r requirements.txt` so the rest of the stack aligns with that install.

**Notes**

- **`pypandoc`** expects [Pandoc](https://pandoc.org/installing.html) on your `PATH` if you use code paths that call it.
- **`comet_ml`** uses API keys in `config.py` (`COMET_*`); set `COMET_DISABLED` if you do not want experiment logging.

## Configuration

Dataset roots and server-specific paths are centralized in **`config.py`**. Before running anything, set variables such as `DATASET_PATH`, `BIOVID_SOURCE_DATASET_PATH`, `STRESS_PATH`, and related entries so they point at **your** data layout. Weight paths often use `WeightFiles/` (see `WEIGHTS_FOLDER`).

## How to run

Run from the **repository root** so imports resolve (`python aexp_tt.py`, not from inside `clip/`).

### `aexp_tt.py`

Test-time prompt tuning and related pipelines (video CLIP, fusion flags, etc.):

```bash
python aexp_tt.py --data /path/to/dataset --gpu 0 -a ViT-B/32 --batch-size 8
```

Useful flags (see the `argparse` block at the bottom of `aexp_tt.py` for the full list):

| Flag | Role |
|------|------|
| `--data` | Dataset root (overrides the default from `config`) |
| `-a` / `--arch` | CLIP architecture name (must be loadable by `clip.load`; see below) |
| `--gpu` | CUDA device index |
| `--tpt` | Enable test-time prompt tuning |
| `--current_ds`, `--pain_db_root_path` | Which benchmark / root path (`config` constants) |
| `--test_sets` | Subsets to evaluate, slash-separated |

Paths and class lists in **`config.py`** and **`data/`** must match your machine.

---

## Where the visual model lives (`clip/clip.py`) and how to add your own

### Two layers: loading vs. architecture

1. **`clip/clip.py`** — **how weights are chosen and loaded**, not the low-level layer definitions.
   - **`_MODELS`**: maps short names (e.g. `ViT-B/32`, `RN50`) to **OpenAI-hosted** `.pt` URLs. Used by **`load()`**.
   - **`_OPENCLIP_MODELS`**: maps aliases to `(open_clip_model_name, pretrained_tag)` for **`open_clip.create_model_and_transforms`**. Used by **`load_copy()`** when the name is in this dict.
   - **`load(name, ...)`** (main path for `aexp_tt.py` via `clip/custom_clip.py`):
     - If `name` is in `_MODELS`, the checkpoint is downloaded (or taken from cache) and loaded.
     - Else if `name` is a **path to a file**, that file is loaded as a JIT archive or a **`state_dict`**.
     - A full **`CLIP`** module is then built with **`build_model(...)`** from **`clip/model.py`**.
   - Preprocessing for images comes from the loaded model’s **`input_resolution`** (see `_transform` in `clip.py`).

2. **`clip/model.py`** — **where the visual backbone is defined and attached**.
   - Class **`CLIP`** constructs **`self.visual`** as either:
     - **`VisionTransformer`** (ViT-style CLIP), or
     - **`ModifiedResNet`** (ResNet-style CLIP),
     depending on the **`state_dict`** layout (see **`build_model`**).
   - **`encode_image`** runs **`self.visual(...)`** — that is the **image tower** used everywhere downstream.

So: **`clip.py` wires names and checkpoints → `build_model` → `CLIP.visual` in `model.py`.**

### Ways to plug in “your own” model

Pick the approach that matches what you have (name, file, or new architecture).

#### A. Same architecture as OpenAI CLIP, your own weights file

Train or export a checkpoint whose keys match the official CLIP **`state_dict`** (prefix `visual.`, `token_embedding`, `text_projection`, etc.). Then pass the **file path** as `-a` / `--arch`:

```bash
python aexp_tt.py -a /path/to/your_clip_weights.pt --data /path/to/dataset --gpu 0
```

No change to `_MODELS` is required if you always pass the path.

#### B. Add a friendly name for a URL or keep using OpenAI’s list

Edit **`_MODELS`** in **`clip/clip.py`**:

```python
_MODELS = {
    # ...existing entries...
    "ViT-B/32": "...",
    "MyViT": "https://.../your_sha256/your_weights.pt",  # URL must match OpenAI-style layout if you reuse load()
}
```

The downloader expects the second-to-last URL segment to be the **SHA256** of the file (same convention as the existing OpenAI URLs).

#### C. OpenCLIP pretrained naming

If your model is available through **OpenCLIP**, add an entry to **`_OPENCLIP_MODELS`** and call **`load_copy()`** (not `load()`) from your code, **or** extend **`load()`** to branch on the same map if you want one code path. The tuple is **`(model_name, pretrained_weights_tag)`** as in **`open_clip.create_model_and_transforms`**.

#### D. New visual backbone (different layers or tensor shapes)

You need a **`nn.Module`** compatible with how **`CLIP`** is used (`encode_image` → feature dim **`embed_dim`**, etc.):

1. Implement your image tower (e.g. subclass **`nn.Module`**) with the same **output dimension** as **`embed_dim`** unless you also change the text side and logits logic.
2. In **`clip/model.py`**, either:
   - extend **`build_model`** / **`CLIP.__init__`** to construct your module when you detect your checkpoint layout, or
   - replace **`self.visual`** after loading with a wrapper that maps your outputs to the expected shape and dtype.

Your **`state_dict`** must either match what **`build_model`** expects or you load partial weights and fill the rest manually.

#### E. Heavier customization (prompts, video, fusion)

**`clip/custom_clip.py`** builds on **`clip.load`** and wraps or extends behavior (e.g. **`VClip`**, adapters). If you only swap the base CLIP checkpoint, often **A** or **B** is enough; if you change the **forward** contract, update those wrappers accordingly.

### Quick reference

| File | Responsibility |
|------|------------------|
| `clip/clip.py` | Model name tables, download, **`load` / `load_copy`**, image transform helper, **`tokenize`** |
| `clip/model.py` | **`VisionTransformer`**, **`ModifiedResNet`**, **`CLIP`**, **`build_model`** ( **`self.visual`** ) |
| `clip/custom_clip.py` | Higher-level CLIP variants and training / eval helpers |

After changing Python modules, restart your process so imports pick up edits.
