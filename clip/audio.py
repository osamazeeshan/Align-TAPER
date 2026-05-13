"""
audio.py  —  Audio model loader for Align-TAPER
================================================
Mirrors the structure and interface of clip.py so the rest of the codebase
can load an audio backbone the same way it loads a CLIP model:

    from .audio import load, available_models
    model, hidden_dim, feature_extractor = audio.load("wavlm-large", device=device)

Available models
----------------
  "hubert-base"        facebook/hubert-base-ls960                  768
  "hubert-large"       facebook/hubert-large-ls960-ft             1024   ★ strong SER
  "wavlm-base"         microsoft/wavlm-base-plus                   768
  "wavlm-large"        microsoft/wavlm-large                      1024   ★ best SUPERB / emotion
  "wav2vec2-base"      facebook/wav2vec2-base-960h                 768
  "wav2vec2-large"     facebook/wav2vec2-large-960h-lv60-self     1024
  "wav2vec2-emotion"   audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  1024  ★ emotion-tuned

Literature
----------
- WavLM-large:  surpasses HuBERT on 14/15 SUPERB tasks incl. emotion recognition.
  Chen et al., IEEE JSTSP 2022.
- HuBERT-large: best frozen SER in SUPERB (67.6% WA); 79.6% WA with fine-tuning.
  Wang et al., arXiv:2111.02735.
- wav2vec2-emotion (audeering): directly fine-tuned for dimensional emotion on
  MSP-Podcast; strongest plug-and-play emotion backbone.
"""

import warnings
from typing import Union

import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    HubertModel,
    Wav2Vec2Model,
    WavLMModel,
)

__all__ = ["available_models", "load"]

# ─── Model registry ──────────────────────────────────────────────────────────
# name → (hf_checkpoint, hidden_dim, model_class)

_MODELS = {
    "hubert-base": (
        "facebook/hubert-base-ls960",
        768,
        HubertModel,
    ),
    "hubert-large": (
        "facebook/hubert-large-ls960-ft",
        1024,
        HubertModel,
    ),
    "wavlm-base": (
        "microsoft/wavlm-base-plus",
        768,
        WavLMModel,
    ),
    "wavlm-large": (
        "microsoft/wavlm-large",
        1024,
        WavLMModel,
    ),
    "wav2vec2-base": (
        "facebook/wav2vec2-base-960h",
        768,
        Wav2Vec2Model,
    ),
    "wav2vec2-large": (
        "facebook/wav2vec2-large-960h-lv60-self",
        1024,
        Wav2Vec2Model,
    ),
    "wav2vec2-emotion": (
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        1024,
        Wav2Vec2Model,
    ),
}


# ─── Backbone wrapper ─────────────────────────────────────────────────────────

class AudioModel(nn.Module):
    """
    Frozen audio backbone that extracts a fixed-dim embedding from raw waveforms.

    Interface mirrors CLIP's encode_image / encode_text:

        z = model.encode_audio(input_values)          # (B, hidden_dim)
        z = model.encode_audio(input_values, mask)    # with attention mask

    The backbone is always frozen. The lightweight adapter that projects
    hidden_dim → CLIP_DIM lives in a separate file (audio_adapter.py),
    matching the pattern of the AU adapter (At) in CLIP-AU.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, name: str):
        super().__init__()
        self.backbone   = backbone
        self.hidden_dim = hidden_dim
        self.name       = name

        # freeze all backbone weights
        for p in self.backbone.parameters():
            p.requires_grad = False

    def encode_audio(
        self,
        input_values:   torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract a single embedding per audio clip by mean-pooling over time.

        Args:
            input_values  : (B, T_samples)  raw waveform, 16 kHz, float32
            attention_mask: (B, T_samples)  optional 0/1 padding mask

        Returns:
            z : (B, hidden_dim)
        """
        with torch.no_grad():
            out    = self.backbone(input_values, attention_mask=attention_mask)
            hidden = out.last_hidden_state          # (B, T_frames, hidden_dim)

        if attention_mask is not None:
            frame_mask = self._downsample_mask(attention_mask, hidden.size(1))
            hidden     = hidden * frame_mask.unsqueeze(-1)
            z          = hidden.sum(1) / frame_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            z = hidden.mean(1)                      # (B, hidden_dim)

        return z

    # forward alias so the module behaves like a standard encoder
    def forward(
        self,
        input_values:   torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.encode_audio(input_values, attention_mask)

    @staticmethod
    def _downsample_mask(
        attention_mask: torch.Tensor,
        num_frames:     int,
    ) -> torch.Tensor:
        """
        Interpolate a sample-level mask (B, T_samples) to a frame-level mask
        (B, T_frames).  All SSL models use a CNN front-end with stride ≈ 320,
        so T_frames ≈ T_samples / 320.
        """
        import torch.nn.functional as F
        mask = attention_mask.float().unsqueeze(1)              # (B, 1, T_s)
        mask = F.interpolate(mask, size=num_frames, mode="nearest").squeeze(1)
        return mask                                             # (B, T_frames)

    def __repr__(self) -> str:
        return (
            f"AudioModel(\n"
            f"  name       = {self.name}\n"
            f"  hidden_dim = {self.hidden_dim}\n"
            f"  backbone   = {type(self.backbone).__name__}\n"
            f"  frozen     = True\n"
            f")"
        )


# ─── Public API ──────────────────────────────────────────────────────────────

def available_models():
    """Returns the names of all available audio models."""
    return list(_MODELS.keys())


def audio_load(
    name:          str,
    device:        Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
) -> tuple:
    """
    Load a pretrained audio backbone.

    Parameters
    ----------
    name : str
        A model name from available_models(), or a local path to a saved
        HuggingFace model directory / checkpoint.

    device : str or torch.device
        Device to place the model on.

    download_root : str, optional
        HuggingFace cache directory override.  If None, uses the HF default
        (~/.cache/huggingface/hub).

    Returns
    -------
    model : AudioModel
        The frozen audio backbone wrapped in AudioModel.

    hidden_dim : int
        Output feature dimension of the backbone (768 or 1024).
        Use this to configure the downstream AudioAdapter.

    feature_extractor : AutoFeatureExtractor
        Preprocessor that converts raw numpy/torch audio arrays at 16 kHz
        into model-ready input_values tensors.

    Examples
    --------
    >>> from .audio import load
    >>> model, hidden_dim, fe = load("wavlm-large", device="cuda")
    >>> inputs = fe(raw_audio, sampling_rate=16000, return_tensors="pt")
    >>> z = model(inputs.input_values.to("cuda"))  # (B, 1024)
    """
    # resolve model info
    if name in _MODELS:
        hf_name, hidden_dim, model_cls = _MODELS[name]
    elif name not in _MODELS:
        # allow passing a local path directly
        import os
        if os.path.isdir(name) or os.path.isfile(name):
            warnings.warn(
                f"'{name}' is not a registered model name; "
                "attempting to load as a local HuggingFace directory."
            )
            hf_name    = name
            hidden_dim = None   # will be read from config below
            model_cls  = None
        else:
            raise RuntimeError(
                f"Model '{name}' not found.  "
                f"Available models: {available_models()}"
            )

    hf_kwargs = {}
    if download_root is not None:
        hf_kwargs["cache_dir"] = download_root

    # load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_name, **hf_kwargs)

    # load backbone
    if model_cls is not None:
        backbone = model_cls.from_pretrained(hf_name, **hf_kwargs)
    else:
        # local path fallback — infer class from config
        from transformers import AutoModel
        backbone   = AutoModel.from_pretrained(hf_name, **hf_kwargs)
        hidden_dim = backbone.config.hidden_size

    backbone = backbone.to(device)
    if str(device) == "cpu":
        backbone.float()

    model = AudioModel(backbone, hidden_dim, name).to(device)

    return model, hidden_dim, feature_extractor


# ─── Convenience: tokenize-style preprocessor wrapper ────────────────────────

def preprocess(
    raw_audio,
    feature_extractor,
    sampling_rate: int = 16000,
    device:        Union[str, torch.device] = "cpu",
) -> dict:
    """
    Convenience wrapper analogous to clip.py's _transform / tokenize.

    Args:
        raw_audio        : np.ndarray or list of np.ndarray  (B, T) at `sampling_rate`
        feature_extractor: returned by load()
        sampling_rate    : should always be 16000 for SSL models
        device           : target device

    Returns:
        dict with keys:
          "input_values"   : (B, T_samples) float32 tensor
          "attention_mask" : (B, T_samples) int64  tensor  (None if not padded)
    """
    out = feature_extractor(
        raw_audio,
        sampling_rate   = sampling_rate,
        return_tensors  = "pt",
        padding         = True,
        return_attention_mask = True,
    )
    return {
        "input_values":   out.input_values.to(device),
        "attention_mask": out.get("attention_mask", None),
    }