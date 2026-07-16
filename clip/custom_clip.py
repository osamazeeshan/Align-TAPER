
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
import seaborn as sns
import os
import opensmile
import joblib

from clip import load, tokenize
from clip.audio import audio_load
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.biovid_prompts import biosub_classes
from data.bah_prompts import bahssub_classes
from data.cls_to_names import *
from .transformer import TemporalTransformer
import copy
import types
import config

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        """
        Compute Grad-CAM for given input and target class.
        """
        logits = self.model(input_tensor)[0]  # forward
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop for selected class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients.mean(dim=[2, 3], keepdim=True)  # global average pool
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def patch_clip_attention(block):
    """
    Monkey-patch a CLIP ResidualAttentionBlock.attn.forward
    so it returns (attn_out, attn_probs).
    """
    attn = block.attn

    if getattr(attn, "_is_patched", False):
        return  # already patched

    def forward_with_probs(self, query, key, value, need_weights=True, attn_mask=None):
        # identical to torch.nn.functional.multi_head_attention_forward
        # but we capture the weights
        import torch.nn.functional as F
        attn_out, attn_probs = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            need_weights=True,
            attn_mask=attn_mask,
            use_separate_proj_weight=False
        )
        return attn_out, attn_probs

    # replace the forward method
    attn.forward = types.MethodType(forward_with_probs, attn)
    attn._is_patched = True


class VClip(nn.Module):
    def __init__(
            self,
            arch,
            device,
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 4,
            dim_forward: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_forward = dim_forward
        # model, _ = load("ViT-B/32", device=device, jit=False)
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        for name, param in clip.named_parameters():
            param.requires_grad = False
        self.backbone = clip
        self.temporal = TemporalTransformer(
            input_dim=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_dim=d_model,
            dim_head=dim_forward
        )
        self.logit_scale = nn.Parameter(self.backbone.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True
    
    def forward(self, x, text, device):
        image_features = self.encode_video(x)

        # text_features = self.encode_text(text)
        text_features = self.encode_text(tokenize(text).to(device))

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
    def encode_video(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        v = self.backbone.encode_image(x).reshape(B, T, -1)
        v = self.temporal(v)
        v = v[:, 0]
        return v
    def encode_text(self, text):
        encoded_text = self.backbone.encode_text(text)
        return encoded_text



class TemporalGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()  # shape [B, T, D]
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, inputs, texts, target_class=None):
        logits, _ = self.model(inputs, texts)
        if target_class is None:
            target_class = logits.argmax(dim=-1)

        self.model.zero_grad()
        logits[:, target_class].sum().backward(retain_graph=True)

        grads = self.gradients.mean(dim=1)  # temporal average over time steps
        cams = (self.activations * grads.unsqueeze(1)).sum(dim=-1)
        cams = F.relu(cams)
        cams = (cams - cams.min()) / (cams.max() + 1e-8)
        return cams.cpu().numpy()  # shape [B, T]



'''
CLIP IMAGE AND VIDEO BASED MODELS

'''
class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # seq_len = prompts.size(1)
        # pos_embed = self.positional_embedding[:seq_len,:].unsqueeze(0)
        # x = prompts + pos_embed
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]

            # replace [CLS] with the actual class name
            # prompts = [
            #     self.prompt_prefix.replace("[CLS]", name) + "." 
            #     for name in classnames
            # ]

            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # replace [CLS] with the actual class name
            # prompts = [
            #     self.prompt_prefix.replace("[CLS]", name) + "." 
            #     for name in classnames
            # ]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class AUPromptLearner(nn.Module):
    """
    Learnable prompt encoder for Action Units (AUs).
    Each AU has its own trainable prompt context.
    """
    def __init__(self, clip_model, au_list, n_ctx=8, ctx_init=None):
        super().__init__()

        self.device = clip_model.visual.conv1.weight.device
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_aus = len(au_list)
        self.n_ctx = n_ctx
        self.au_list = au_list

        # 🔹 Initialize context tokens (trainable)
        if ctx_init:
            print(f"[INFO] Initializing AU context with phrase: '{ctx_init}'")
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            print(f"[INFO] Random init of {n_ctx} context tokens per AU.")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        # one learnable context per AU (duplicate the same initialization)
        ctx_vectors = ctx_vectors.unsqueeze(0).repeat(self.n_aus, 1, 1)
        self.ctx = nn.Parameter(ctx_vectors)  # [num_aus, n_ctx, D]

        # keep a copy for reset
        self.ctx_init_state = ctx_vectors.detach().clone()

        # 🔹 Build AU prompt templates and tokenize
        au_prompts = [au.replace("_", " ") + "." for au in au_list]
        tokenized = torch.cat([tokenize(p) for p in au_prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)

        # first token [SOS], last token [EOS]
        self.register_buffer("prefix", embedding[:, :1, :])
        self.register_buffer("suffix", embedding[:, -1:, :])

        self.tokenized_prompts = tokenized
        print(f"[INFO] AUPromptLearner initialized with {self.n_aus} prompts × {self.n_ctx} context tokens.")

    def reset(self):
        """Reset learnable context tokens to their initial state."""
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)
        print("[INFO] AU prompt contexts reset to initial state.")

    def reset_prompts(self, new_au_list, clip_model):
        """Optional: reinitialize prompt templates for a new AU list."""
        self.au_list = new_au_list
        au_prompts = [au.replace("_", " ") + "." for au in new_au_list]
        tokenized = torch.cat([tokenize(p) for p in au_prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(self.dtype)
        self.prefix = embedding[:, :1, :]
        self.suffix = embedding[:, -1:, :]
        self.tokenized_prompts = tokenized
        print(f"[INFO] AU prompt templates reset for {len(new_au_list)} AUs.")

    def forward(self):
        """Construct trainable AU prompts and pad to 77 tokens for CLIP compatibility."""
        prefix = self.prefix   # [num_aus, 1, D]
        suffix = self.suffix   # [num_aus, 1, D]
        ctx = self.ctx         # [num_aus, n_ctx, D]

        # Construct AU prompt sequence: [SOS] + ctx + [EOS]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # [num_aus, n_ctx+2, D]

        # ✅ Pad to 77 tokens (CLIP text encoder expected length)
        max_len = 77
        seq_len = prompts.size(1)
        if seq_len < max_len:
            pad_len = max_len - seq_len
            pad = torch.zeros(
                prompts.size(0), pad_len, prompts.size(2),
                device=prompts.device, dtype=prompts.dtype
            )
            prompts = torch.cat([prompts, pad], dim=1)  # [num_aus, 77, D]

        return prompts


class ClipTestTimeTuning(nn.Module):
    # def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
    #                     n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
    def __init__(self, device, classnames, batch_size,
                 criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                 num_aus=30, num_classes=2, text_hidden=512, clf_hidden=256):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.clip = clip
        self.device = device

        # 🔹 text adapter for AU prompts (trainable)
        d = self.image_encoder.output_dim  # CLIP embed dim
        self.text_adapter = nn.Sequential(
            nn.Linear(d, text_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden, d)
        )

        # 🔹 AU classifier head (trainable)
        self.au_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # 🔹 Subject-specific adapter (very small; train per subject)
        # for example: a learnable affine transform on AU features
        self.subject_adapter = nn.Sequential(
            nn.Linear(num_classes, num_classes)
        )

        # fusion weight for simple weighting of the two logits
        self.fusion_alpha = 0.5  # can be tuned or learned


        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    # 🔹 NEW METHOD for custom prompts
    @torch.no_grad()
    def get_text_features_from_prompts(self, prompt_list):
        """
        Encode arbitrary custom prompt strings into normalized text features.
        prompt_list: list of strings
        returns: tensor [len(prompt_list), D]
        """
        # from clip import tokenize  # import tokenize here or at top

        tokenized_prompts = torch.cat([tokenize(p) for p in prompt_list]).to(self.image_encoder.conv1.weight.device)
        # reuse CLIP's token embedding from prompt_learner's clip
        # clip, _, _ = load(self.bb, device=self.image_encoder.conv1.weight.device, download_root=DOWNLOAD_ROOT)
        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized_prompts).type(self.text_encoder.dtype)
        t_features = self.text_encoder(embedding, tokenized_prompts)
        return t_features / t_features.norm(dim=-1, keepdim=True)  # [N,D]
        
    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    # AU pathway logits (text adapter + AU classifier)
    def au_pathway_logits(self, image, au_prompts):
        with torch.no_grad():
            img_features = self.image_encoder(image.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_features = F.normalize(img_features, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        # adapted_embeds = self.text_adapter(base_text_embeds)  # trainable
        # adapted_embeds = F.normalize(adapted_embeds, dim=-1)

        sim = img_features @ base_text_embeds.T  # [B, num_aus]
        logits = self.au_classifier(sim)       # [B, num_classes]
        return logits

    # fused logits with subject-specific adapter on top
    def fused_logits(self, image, au_prompts, alpha=None):
        if alpha is None:
            alpha = self.fusion_alpha
        logits_clip = self.inference(image)             # [B,C]
        logits_au = self.au_pathway_logits(image, au_prompts)  # [B,C]
        fused = alpha * logits_au + (1 - alpha) * logits_clip  # [B,C]
        # subject-specific adapter (train per subject)
        fused = self.subject_adapter(fused)  # [B,C]
        return fused

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, input, au_prompts=None, mode="clip"):
        """
        mode:
          "clip"  -> original CLIP logits
          "au"    -> AU pathway logits
          "fused" -> weighted combination of the two + subject adapter
        """
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            if mode == "clip":
                return self.inference(input)
            elif mode == "au":
                assert au_prompts is not None
                return self.au_pathway_logits(input, au_prompts)
            elif mode == "adapt":
                assert au_prompts is not None
                return self.fused_logits(input, au_prompts)

    def compute_au_similarities(self, images, au_prompts, device):
        """
        Compute AU similarity vectors for a batch of images.
        Returns: [B,num_aus] tensor
        """
        with torch.no_grad():
            img_features = self.image_encoder(images.type(self.dtype))
            img_features = F.normalize(img_features, dim=-1)

            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted_embeds = self.text_adapter(base_text_embeds)  # keep grad if training adapter
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)

        sim = img_features @ adapted_embeds.T  # [B,num_aus]
        return sim
        
    def get_au_similarity(self, image_features, au_prompts):
        """
        image_features: [B,D] CLIP image embeddings (normalized)
        au_prompts: list of AU prompt strings
        returns: [B, num_aus] similarity scores
        """
        with torch.no_grad():
            base_text_embeds = self.clip.encode_text(
                tokenize(au_prompts).to(self.device)
            )  # [num_aus,D]
        adapted_embeds = self.text_adapter(base_text_embeds)      # [num_aus,D]
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)
        sim = image_features @ adapted_embeds.T                  # [B,num_aus]
        return sim

    
    # def forward(self, input):
    #     if isinstance(input, Tuple):
    #         view_0, view_1, view_2 = input
    #         return self.contrast_prompt_tuning(view_0, view_1, view_2)
    #     elif len(input.size()) == 2:
    #         return self.directional_prompt_tuning(input)
    #     else:
    #         return self.inference(input)

    # def forward(self, images, au_prompts):
    #     """
    #     images: [B,3,H,W] tensor
    #     au_prompts: list of AU strings (len = num_aus)
    #     returns: [B,num_classes] logits
    #     """
    #     # 1. image features from CLIP
    #     with torch.no_grad():  # keep CLIP frozen initially
    #         img_features = self.image_encoder(images.type(self.dtype))
    #     # img_features = F.normalize(img_features, dim=-1)          # [B,D]

    #     # 2. AU similarity vector
    #     sim = self.get_au_similarity(img_features, au_prompts)    # [B,num_aus]

    #     # 3. classify using AU similarities
    #     logits = self.au_classifier(sim)                          # [B,num_classes]
    #     return logits
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, v, a):
        """
        v: (B, D)
        a: (B, D)
        """

        # make them sequences of length 1
        v = v.unsqueeze(1)  # (B, 1, D)
        a = a.unsqueeze(1)  # (B, 1, D)

        # visual attends to audio
        out, attn_weights = self.attn(
            query=v,
            key=a,
            value=a
        )

        # residual connection
        out = self.norm(out + v)

        return out.squeeze(1), attn_weights

class ClipTestTimeVideoTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, au_prompts,
                 criterion='cosine', arch="ViT-L/14",
                 n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                 num_aus=46, num_classes=2, text_hidden=512, clf_hidden=256,
                 d_model=192, num_layers=2, nhead=4, dim_forward=512,
                 delta_stride=1, audio_arch="wavlm-large", audio_dictionary_path=None, opensmile_scaler_path=None, 
                 audio_cluster_labels_path=None, audio_response_hidden=128, audio_dropout=0.2, audio_feature_dim=88):
        super(ClipTestTimeVideoTuning, self).__init__()

        # -------------------------------------------------------
        # 1️⃣ Load CLIP backbone
        # -------------------------------------------------------
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        # self.logit_scale = clip.logit_scale.data
        self.prompt_learner = PromptLearner(
            clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls
        )
        self.au_prompt_learner = AUPromptLearner(clip, au_prompts, n_ctx) 
        self.device = device
        self.clip = clip
        self.classnames = classnames

        self.logit_scale = nn.Parameter(clip.logit_scale.clone().detach())
        self.logit_scale.requires_grad = True

        # -------------------------------------------------------
        # 2️⃣ Load Audio Backbone
        # -------------------------------------------------------
        audio_model, hidden_dim, audio_fe = audio_load(audio_arch, device=device)
        self.audio_model = audio_model
        self.audio_fe = audio_fe
        self.num_classes = num_classes

        # Audio adapter (trainable)
        # self.audio_adapter = nn.Sequential(
        #     nn.Linear(hidden_dim, text_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(text_hidden, text_hidden),
        #     nn.LayerNorm(text_hidden)
        # )

        # ----------------------------------------------------------
        # 2️⃣.1 Audio Fixed OpenSMILE feature extractor
        # ----------------------------------------------------------
        # This is not trainable. It converts waveform -> 88-D features.
        self.opensmile_model = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # -------------------------------------------------------
        # 2️⃣ Trainable AU adapter + classifier
        # -------------------------------------------------------
        d = self.image_encoder.output_dim  # CLIP embed dim

        # Text adapter (trainable)
        self.text_adapter = nn.Sequential(
            nn.Linear(d, text_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(text_hidden, d)
        )

        # AU classifier (trainable)
        self.au_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # 🔹 Subject-specific adapter (very small; train per subject)
        # for example: a learnable affine transform on AU features
        self.subject_adapter = nn.Sequential(
            nn.Linear(num_classes, num_classes)
        )

        # fusion weight for simple weighting of the two logits
        self.fusion_alpha = 0.5  # can be tuned or learned

        # temporal_classifier (trainable)
        self.temporal_classifier = nn.Sequential(
            nn.Linear(num_aus, clf_hidden), # IMPORANT CHNAGE BACK TO num_aus
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden, num_classes)
        )

        # -------------------------------------------------------
        # 3️⃣ Temporal Transformer for AU sequence modeling
        # -------------------------------------------------------
        self.temporal_proj = nn.Linear(d*2, d_model)
        # self.temporal_proj = nn.Linear(2 * num_aus, d_model)
        # self.temporal = TemporalTransformer(
        #     input_dim=d_model,
        #     depth=num_layers,
        #     heads=nhead,
        #     mlp_dim=d_model,
        #     dim_head=dim_forward
        # )
        # self.temporal_norm = nn.LayerNorm(d_model)

        self.temporal = nn.Conv1d(d, d, kernel_size=3, padding=1) # for biovid and stress kernel_size= 3 for BAH=test=6,9,12

        self.dropout = nn.Dropout(p=0.1)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, d) * 0.02)
        # how frequently to compute ΔA_t (e.g., every 1, 2, or 3 frames)
        self.delta_stride = delta_stride

        # ✅ Add this projection here (persistent parameter)
        self.class_to_au = nn.Linear(num_classes, num_aus, bias=False)

        # -- Fusion Module 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512)
        )

        # -- Fusion: Cross-Attention
        self.crossAtten_fusion = CrossAttentionFusion(512)

        # -- Fusion: Gated
        self.gated_fusion = nn.Linear(512 * 2, 512) 

        # -- MOE - It will decide which modality is dominant and based on that add weights
        self.router = nn.Linear(512 * 2, 2)

        self.contr_logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

        # ==================================================
        # 4. Load fixed OpenSMILE audio dictionary
        # ==================================================

        if os.path.exists(audio_cluster_labels_path):
            cluster_labels_np = np.load(audio_cluster_labels_path)  # shape [K]
            self.register_buffer(
                "audio_cluster_labels",
                torch.tensor(cluster_labels_np, dtype=torch.long),
                persistent=True,
            )
        else:
            cluster_labels_np = None

        # if audio_dictionary_path is None:
        #     raise ValueError(
        #         "audio_dictionary_path must point to the saved "
        #         "OpenSMILE dictionary .npy file."
        #     )

        # if not os.path.exists(audio_dictionary_path):
        #     raise FileNotFoundError(
        #         f"Audio dictionary not found: {audio_dictionary_path}"
        #     )

        if os.path.exists(audio_dictionary_path):
            audio_dictionary_np = np.load(
                audio_dictionary_path
            )

            audio_dictionary_np = np.asarray(
                audio_dictionary_np,
                dtype=np.float32,
            )

            audio_dictionary_np = np.nan_to_num(
                audio_dictionary_np,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if audio_dictionary_np.ndim != 2:
                raise ValueError(
                    "Audio dictionary must have shape [N, 88], "
                    f"but received {audio_dictionary_np.shape}."
                )
            self.num_audio_atoms = audio_dictionary_np.shape[0]

        # audio_dictionary = torch.as_tensor(
        #     audio_dictionary_np,
        #     dtype=torch.float32,
        # )


        self.audio_feature_dim = audio_dictionary_np.shape[1]

        if self.audio_feature_dim != 88:
            raise ValueError(
                "Expected an 88-D eGeMAPSv02 dictionary, "
                f"but received dimension {self.audio_feature_dim}."
            )

        audio_dictionary_tensor = torch.from_numpy(
            audio_dictionary_np
        )

        # IMPORTANT:
        # Store the exact source-derived dictionary.
        # Do not transform or normalize it here.
        self.register_buffer(
            "audio_dictionary",
            audio_dictionary_tensor,
            persistent=True,
        )

        # ----------------------------------------------------------
        # Load scaler used when constructing the dictionary
        # ----------------------------------------------------------
        if opensmile_scaler_path is None:
            raise ValueError(
                "opensmile_scaler_path is required because inference/training "
                "features must use the same scaler as the source dictionary."
            )

        if not os.path.exists(opensmile_scaler_path):
            raise FileNotFoundError(
                f"OpenSMILE scaler not found: {opensmile_scaler_path}"
            )

        opensmile_scaler = joblib.load(opensmile_scaler_path)

        if len(opensmile_scaler.mean_) != audio_feature_dim:
            raise ValueError(
                f"Scaler dimension is {len(opensmile_scaler.mean_)}, "
                f"but expected {audio_feature_dim}."
            )

        # Store scaler statistics as fixed torch buffers.
        # This lets preprocessing run on the same device as the model.
        self.register_buffer(
            "opensmile_mean",
            torch.tensor(
                opensmile_scaler.mean_,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self.register_buffer(
            "opensmile_scale",
            torch.tensor(
                opensmile_scaler.scale_,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        # ==================================================
        # Trainable audio query adapter
        # ==================================================
        #
        # This adapter transforms only the input audio feature.
        # The fixed audio dictionary is never transformed.
        #
        # Input:  processed OpenSMILE feature [B, 88]
        # Output: adapted query feature [B, 88]

        self.audio_adapter = nn.Sequential(
            nn.LayerNorm(88),
            nn.Linear(88, clf_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(clf_hidden, 88),
        )

        # Initialize with a small residual contribution.
        # This prevents the adapter from immediately destroying
        # the original OpenSMILE geometry.
        self.audio_adapter_scale = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

        fusion_similarity_dim = num_aus + self.num_audio_atoms

        self.au_audio_classifier = nn.Sequential(
            # nn.LayerNorm(fusion_similarity_dim),
            nn.Linear(self.num_audio_atoms, clf_hidden), # fusion_similarity_dim
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(clf_hidden, num_classes),
        )

        # audio starts with very small influence
        self.audio_fusion_alpha = nn.Parameter(torch.tensor(1.0))


        align_dim = 128

        self.visual_align_proj = nn.Sequential(
            nn.Linear(512, align_dim),          # 46 -> 128
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )

        self.audio_align_proj = nn.Sequential(
            nn.Linear(88, align_dim),  # K -> 128
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )

        self.subject_fusion_delta = nn.Parameter(
            torch.zeros(())
        )


    def visualize_au_text_embeddings(self, video, au_prompts, adapt_tar, key_frame_sel, key_frames, device="cuda"):
        """
        Visualize AU prompt embeddings before and after the text adapter (AU adapter)
        and compute similarity with visual embeddings.

        Args:
            video (torch.Tensor): input video [B, T, C, H, W]
            au_prompts (list[str]): list of 46 AU prompt strings
            adapt_tar (bool): whether to apply adaptation
            key_frame_sel (bool): whether to use key-frame selection
            device (str): "cuda" or "cpu"
        """

        self.eval()
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            # 1️⃣ Encode AU prompts via CLIP text encoder
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

            # 2️⃣ Pass through trained AU adapter
            adapted_text_embeds = self.text_adapter(base_text_embeds)
            adapted_text_embeds = F.normalize(adapted_text_embeds, dim=-1)

            # 3️⃣ Extract visual temporal embeddings
            img_feats = self.image_encoder(x.type(self.dtype))  # [B*T, D]
            img_feats = img_feats.reshape(B, T, -1)  # [B, T, D]

            # Optional key-frame selection
            selected_feats = []
            if adapt_tar and key_frame_sel:
                for b in range(B):
                    frame_feats = img_feats[b]  # [T, D]
                    key_idx = self.select_key_consec_frames(
                        frame_feats, base_text_embeds, top_k=key_frames
                    )
                    selected_feats.append(frame_feats[key_idx])

                max_T = max(len(f) for f in selected_feats)
                selected_feats = torch.stack([
                    F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
                ])  # [B, max_T, D]

            v = self.temporal(selected_feats.transpose(1, 2) if key_frame_sel else img_feats.transpose(1, 2))
            v = F.gelu(v)
            visual_embeds = v.mean(dim=-1)
            visual_embeds_norm = F.normalize(visual_embeds, dim=-1)

        # 4️⃣ Compute cosine similarities
        sim_text_visual = torch.cosine_similarity(
            visual_embeds_norm.mean(0, keepdim=True), base_text_embeds.mean(0, keepdim=True)
        )
        sim_adapter_visual = torch.cosine_similarity(
            visual_embeds_norm.mean(0, keepdim=True), adapted_text_embeds.mean(0, keepdim=True)
        )

        print(f"Mean cosine similarity (Visual ↔ Text before adapter):  {sim_text_visual.item():.3f}")
        print(f"Mean cosine similarity (Visual ↔ Text after adapter):   {sim_adapter_visual.item():.3f}")

        # 5️⃣ t-SNE visualization
        X = torch.cat([base_text_embeds, adapted_text_embeds], dim=0).cpu().numpy()
        labels = ["Before Adapter"] * len(base_text_embeds) + ["After Adapter"] * len(adapted_text_embeds)

        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(7, 6))
        plt.scatter(X_2d[:len(base_text_embeds), 0], X_2d[:len(base_text_embeds), 1],
                    c='red', label='Before Adapter', alpha=0.7)
        plt.scatter(X_2d[len(base_text_embeds):, 0], X_2d[len(base_text_embeds):, 1],
                    c='blue', label='After Adapter', alpha=0.7)
        plt.title("t-SNE of AU Prompt Embeddings\nBefore vs After AU Adapter")
        plt.legend()
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()

        os.makedirs("visual/tsne", exist_ok=True)
        save_path = "visual/tsne/tsne_au_adapter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to: {save_path}")
        plt.show()

        # 6️⃣ Text–Adapter internal similarity
        sim_text_adapter = torch.cosine_similarity(base_text_embeds, adapted_text_embeds, dim=1)
        print(f"Mean cosine similarity (Text before vs after adapter): {sim_text_adapter.mean().item():.3f}")

    def visualize_video_alignment_tsne(self, video, au_prompts, adapt_tar, key_frame_sel, key_frames, device="cuda"):
        """
        Visualize t-SNE embeddings of video features, AU text embeddings (before), 
        and AU adapter embeddings (after).
        """

        self.eval()
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            # --- AU text embeddings ---
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

            adapted_text_embeds = self.text_adapter(base_text_embeds)
            adapted_text_embeds = F.normalize(adapted_text_embeds, dim=-1)

            # --- Visual (temporal) embeddings ---
            img_feats = self.image_encoder(x.type(self.dtype))  # [B*T, D]
            img_feats = img_feats.reshape(B, T, -1)  # [B, T, D]

            selected_feats = []
            if adapt_tar and key_frame_sel:
                for b in range(B):
                    frame_feats = img_feats[b]
                    key_idx = self.select_key_consec_frames(frame_feats, base_text_embeds, top_k=key_frames)
                    selected_feats.append(frame_feats[key_idx])
                max_T = max(len(f) for f in selected_feats)
                selected_feats = torch.stack([
                    F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
                ])
            else:
                selected_feats = img_feats

            v = self.temporal(selected_feats.transpose(1, 2))
            v = F.gelu(v)
            visual_embeds = v.mean(dim=-1)
            visual_embeds = F.normalize(visual_embeds, dim=-1)

        # --- Cosine similarity summary ---
        sim_text_visual = torch.cosine_similarity(
            visual_embeds.mean(0, keepdim=True), base_text_embeds.mean(0, keepdim=True))
        sim_adapter_visual = torch.cosine_similarity(
            visual_embeds.mean(0, keepdim=True), adapted_text_embeds.mean(0, keepdim=True))
        print(f"Mean cosine similarity (Visual ↔ Text before): {sim_text_visual.item():.3f}")
        print(f"Mean cosine similarity (Visual ↔ Text after):  {sim_adapter_visual.item():.3f}")

        # --- Prepare for t-SNE ---
        X = torch.cat([
            visual_embeds.cpu(), 
            base_text_embeds.cpu(), 
            adapted_text_embeds.cpu()
        ], dim=0).numpy()

        labels = (["Visual"] * len(visual_embeds) +
                ["AU_Text_Before"] * len(base_text_embeds) +
                ["AU_Text_After"] * len(adapted_text_embeds))

        tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X)

        # --- Plot ---
        plt.figure(figsize=(7, 6))
        plt.scatter(X_2d[:len(visual_embeds), 0], X_2d[:len(visual_embeds), 1],
                    c="green", label="Visual", alpha=0.7)
        plt.scatter(X_2d[len(visual_embeds):len(visual_embeds)+len(base_text_embeds), 0],
                    X_2d[len(visual_embeds):len(visual_embeds)+len(base_text_embeds), 1],
                    c="red", label="AU Text (Before)", alpha=0.7)
        plt.scatter(X_2d[-len(adapted_text_embeds):, 0],
                    X_2d[-len(adapted_text_embeds):, 1],
                    c="blue", label="AU Text (After Adapter)", alpha=0.7)
        plt.title("t-SNE: Visual vs. AU Text Embeddings\nBefore & After AU Adapter")
        plt.legend()
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()

        # os.makedirs("visual/tsne", exist_ok=True)
        save_path = "visual/tsne/tsne_visual_text_alignment.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to: {save_path}")
        plt.show()

    def visualize_video_text_au_tsne(self, video, au_prompts, adapt_tar, key_frame_sel, key_frames, labels, sub_id="00", device="cuda"):
        """
        Visualize t-SNE embeddings of video features, AU text embeddings (before), 
        and AU adapter embeddings (after).
        """

        self.eval()
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            # --- AU text embeddings ---
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

            adapted_text_embeds = self.text_adapter(base_text_embeds)
            adapted_text_embeds = F.normalize(adapted_text_embeds, dim=-1)

            # --- Visual (temporal) embeddings ---
            img_feats = self.image_encoder(x.type(self.dtype))  # [B*T, D]
            img_feats = img_feats.reshape(B, T, -1)  # [B, T, D]

            selected_feats = []
            if adapt_tar and key_frame_sel:
                for b in range(B):
                    frame_feats = img_feats[b]
                    key_idx = self.select_key_consec_frames(frame_feats, adapted_text_embeds, top_k=key_frames)
                    selected_feats.append(frame_feats[key_idx])
                max_T = max(len(f) for f in selected_feats)
                selected_feats = torch.stack([
                    F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
                ])
            else:
                selected_feats = img_feats

            v = self.temporal(selected_feats.transpose(1, 2))
            v = F.gelu(v)
            visual_embeds = v.mean(dim=-1)
            visual_embeds = F.normalize(visual_embeds, dim=-1)

        # Cosine similarities: 32×46 matrices
        sim_before = F.cosine_similarity(
            visual_embeds.unsqueeze(1), base_text_embeds.unsqueeze(0), dim=-1
        )  # [32, 46]
        sim_after = F.cosine_similarity(
            visual_embeds.unsqueeze(1), adapted_text_embeds.unsqueeze(0), dim=-1
        )  # [32, 46]

        # ---- Prepare data for t-SNE ----
        V = sim_before.shape[0]  # number of videos
        X = torch.cat([sim_before, sim_after], dim=0).cpu().numpy()
        color_domain = np.array(["Frozen CLIP"] * V + ["AU Adapter"] * V)
        label_vals = labels.detach().cpu().numpy()
        shape_labels = np.concatenate([label_vals, label_vals])  # repeated for before/after

        # ---- t-SNE ----
        tsne = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")
        X_2d = tsne.fit_transform(X)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.set_style("whitegrid")

        # Optional: use class-based markers for shape distinction
        # markers = {0: "o", 1: "^"}  # 0 = neutral, 1 = emotion
        colors = {"Frozen CLIP": "#d62728", "AU Adapter": "#1f77b4"}  # red/blue

        # ---- Plot each domain ----
        for domain in ["Frozen CLIP", "AU Adapter"]:
            mask = color_domain == domain
            for cls in np.unique(shape_labels):
                cls_mask = mask & (shape_labels == cls)
                ax.scatter(
                    X_2d[cls_mask, 0],
                    X_2d[cls_mask, 1],
                    c=colors[domain],
                    # marker=markers[cls],
                    s=70,
                    alpha=0.85,
                    linewidth=0.5,
                )

        # ---- Legend (only 2 color labels) ----
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors["Frozen CLIP"], marker='o', linestyle='', markersize=10, label="Frozen CLIP"),
            Line2D([0], [0], color=colors["AU Adapter"], marker='o', linestyle='', markersize=10, label="AU Adapter")
        ]
        ax.legend(
            handles=legend_elements,
            # title="Model Stage",
            loc="lower left",
            frameon=True,          # enable box
            framealpha=1.0,        # solid background
            # edgecolor="black",     # black border
            fancybox=False,        # square corners (CVPR-style)
            fontsize=15
        )

        # ---- Axis style ----
        # ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.3)
            spine.set_color("black")

        ax.grid(False)

        # ax.set_title("t-SNE of Video–AU Similarity\nColor: Model Stage | Shape: Class", fontsize=15, weight="bold")
        plt.tight_layout()
        plt.savefig("visual/tsne/tsne_frozen_vs_adapter"+str(sub_id)+".jpg", dpi=400, bbox_inches="tight", transparent=True)
        plt.show()


        # plt.tight_layout()
        # save_path = "visual/tsne/tsne_joint_before_after"+str(sub_id)+".jpg"
        # plt.savefig(save_path, dpi=400, bbox_inches="tight", transparent=False)
        # print(f"Joint t-SNE plot saved to: {save_path}")
        # plt.show()


    def visualize_video_text_au_tsne_paper(self, video, au_prompts, adapt_tar, key_frame_sel, key_frames, labels, sub_id="00", device="cuda"):
        """
        Visualize t-SNE embeddings of video features, AU text embeddings (before), 
        and AU adapter embeddings (after).
        """

        self.eval()
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            # --- AU text embeddings ---
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(device))
            base_text_embeds = F.normalize(base_text_embeds, dim=-1)

            adapted_text_embeds = self.text_adapter(base_text_embeds)
            adapted_text_embeds = F.normalize(adapted_text_embeds, dim=-1)

            # --- Visual (temporal) embeddings ---
            img_feats = self.image_encoder(x.type(self.dtype))  # [B*T, D]
            img_feats = img_feats.reshape(B, T, -1)  # [B, T, D]

            selected_feats = []
            if adapt_tar and key_frame_sel:
                for b in range(B):
                    frame_feats = img_feats[b]
                    key_idx = self.select_key_consec_frames(frame_feats, adapted_text_embeds, top_k=key_frames)
                    selected_feats.append(frame_feats[key_idx])
                max_T = max(len(f) for f in selected_feats)
                selected_feats = torch.stack([
                    F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
                ])
            else:
                selected_feats = img_feats

            v = self.temporal(selected_feats.transpose(1, 2))
            v = F.gelu(v)
            visual_embeds = v.mean(dim=-1)
            visual_embeds = F.normalize(visual_embeds, dim=-1)

        # Cosine similarities: 32×46 matrices
        sim_before = F.cosine_similarity(
            visual_embeds.unsqueeze(1), base_text_embeds.unsqueeze(0), dim=-1
        )  # [32, 46]
        sim_after = F.cosine_similarity(
            visual_embeds.unsqueeze(1), adapted_text_embeds.unsqueeze(0), dim=-1
        )  # [32, 46]

        # ---- Convert to numpy ----
        sim_before_np = sim_before.cpu().numpy()
        sim_after_np = sim_after.cpu().numpy()
        label_vals = labels.detach().cpu().numpy()

        # ---- Run t-SNE separately for before and after ----
        tsne_before = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")
        tsne_after = TSNE(n_components=2, perplexity=10, random_state=42, init="pca")

        X_before = tsne_before.fit_transform(sim_before_np)
        X_after = tsne_after.fit_transform(sim_after_np)

        # ---- Create side-by-side CVPR-style figure ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # --- Before Adapter ---
        scatter1 = axes[0].scatter(
            X_before[:, 0], X_before[:, 1],
            c=label_vals, cmap="coolwarm", s=60, edgecolor="k", alpha=0.85
        )
        axes[0].set_title("Before AU Adapter", fontsize=18, weight="semibold")
        # axes[0].set_xlabel("t-SNE dim 1", fontsize=14)
        # axes[0].set_ylabel("t-SNE dim 2", fontsize=14)

        # --- After Adapter ---
        scatter2 = axes[1].scatter(
            X_after[:, 0], X_after[:, 1],
            c=label_vals, cmap="coolwarm", s=60, edgecolor="k", alpha=0.85
        )
        axes[1].set_title("After AU Adapter", fontsize=18, weight="semibold")
        # axes[1].set_xlabel("t-SNE dim 1", fontsize=14)
        # axes[1].set_ylabel("t-SNE dim 2", fontsize=14)

        # Shared colorbar
        # cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        # fig.colorbar(scatter2, cax=cbar_ax, label="Class Label")

        # --- Hide ticks and tick labels, keep box frames ---
        for ax in axes:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)   # keep the box frame
                spine.set_linewidth(1.5)
                spine.set_color("black")

        plt.subplots_adjust(wspace=0.25)
        plt.suptitle(
            "t-SNE of Video–AU Similarity Patterns\nBefore and After AU Adapter",
            fontsize=20, weight="bold"
        )

        save_path = "visual/tsne/tsne_dual_video_similarity_"+str(sub_id)+".png"
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Dual t-SNE figure saved to: {save_path}")

        plt.show()


    # ===========================================================
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_states(self):
        """
        Reset runtime states (hidden, cache, buffer) 
        for temporal modules after each video.
        Keeps pretrained weights intact.
        """
        with torch.no_grad():
            modules = [self.text_adapter, self.temporal, self.temporal_classifier]
            for module in modules:
                module.zero_grad(set_to_none=True)  # optional, just clears residual grads
    
    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    # ===========================================================
    @torch.no_grad()
    def get_text_features_from_prompts(self, prompt_list):
        """Compute text embeddings from arbitrary prompt strings."""
        tokenized = torch.cat([tokenize(p) for p in prompt_list]).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized).type(self.text_encoder.dtype)
        t_features = self.text_encoder(embedding, tokenized)
        return F.normalize(t_features, dim=-1)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def get_au_features(self):
        prompts = self.au_prompt_learner()
        tokenized_prompts = self.au_prompt_learner.tokenized_prompts
        # with torch.no_grad():
        au_features = self.text_encoder(prompts, tokenized_prompts)
        au_features_norm = F.normalize(au_features, dim=-1)

        return au_features_norm
    # ===========================================================
    def compute_au_similarities(self, video, au_prompts):
        """
        Compute per-frame AU similarity vectors for a video.
        video: [B, T, 3, H, W]
        au_prompts: list of AU text prompts
        """
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)

        with torch.no_grad():
            img_feats = self.image_encoder(flat.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_feats = F.normalize(img_feats, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted = self.text_adapter(base_text_embeds)
        adapted = F.normalize(adapted, dim=-1)

        sim = img_feats @ adapted.T
        return sim.view(B, T, -1)  # [B, T, num_aus]

    # ===========================================================
    def forward_temporal(self, video, au_prompts):
        """
        Perform forward pass for temporal FER.
        During training, this is where gradients flow.
        """
        B, T, _, _, _ = video.shape

        # 1️⃣ Compute per-frame AU activations (A_t)
        A = self.compute_au_similarities(video, au_prompts)  # [B, T, num_aus]

        # 2️⃣ Compute AU temporal difference ΔA_t
        if self.delta_stride > 1:
            dA = A[:, self.delta_stride:, :] - A[:, :-self.delta_stride, :]
            pad = torch.zeros_like(A[:, :self.delta_stride, :])
            dA = torch.cat([pad, dA], dim=1)
        else:
            dA = torch.diff(A, dim=1, prepend=A[:, 0:1, :])

        # 3️⃣ Concatenate [A_t, ΔA_t] and project to transformer dim
        X = torch.cat([A, dA], dim=-1)      # [B, T, 2*num_aus]
        X = self.temporal_proj(X)           # [B, T, d_model]

        # 4️⃣ Pass through Temporal Transformer
        out = self.temporal(X)              # [B, T+1, d_model]
        z_cls = self.temporal_norm(out[:, 0, :])  # [B, d_model] (CLS token)

        # 5️⃣ Classify
        logits = self.temporal_classifier(z_cls)
        return logits

    def select_key_frames(self, video_feats, adapted_embeds, top_k=5):
        """
        Select key frames using entropy-based confidence on AU similarities.
        
        Args:
            video_feats: [T, D] visual features (CLIP frame embeddings)
            adapted_embeds: [num_AUs, D] normalized AU text embeddings
            top_k: number of frames to keep

        Returns:
            List[int]: indices of the most confident (low-entropy) frames
        """
        T = video_feats.size(0)
        if T <= top_k:
            return list(range(T))

        # --- Compute AU similarity per frame ---
        video_feats_norm = F.normalize(video_feats, dim=-1)
        au_sim = video_feats_norm @ adapted_embeds.T  # [T, num_AUs]

        # --- Classifier logits per frame ---
        logits_per_frame = self.temporal_classifier(au_sim)  # [T, num_classes]
        probs = F.softmax(logits_per_frame, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [T]
        
        # --- Select lowest entropy frames (highest confidence) ---
        top_idx = torch.topk(-entropy, k=top_k).indices
        return sorted(top_idx.cpu().tolist())

    def select_key_consec_frames(self, video_feats, adapted_embeds, top_k=5, percentile=0.3):
        """
        Select top-K consecutive frames with lowest average entropy.
        Args:
            video_feats: [T, D] CLIP visual frame features (normalized)
            adapted_embeds: [num_AUs, D] AU text embeddings (after adapter + norm)
            top_k: number of consecutive frames to keep
        Returns:
            List[int]: indices of selected consecutive frames
        """
        T = video_feats.size(0)
        if T <= top_k:
            return list(range(T))

        # --- Compute per-frame AU similarity and logits ---
        au_sim = F.normalize(video_feats, dim=-1) @ adapted_embeds.T  # [T, num_AUs]
        logits_per_frame = self.temporal_classifier(au_sim)            # [T, num_classes]

        # --- Compute entropy for each frame ---
        probs = F.softmax(logits_per_frame, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)       # [T]

        # --- Sliding window average entropy ---
        window_entropies = torch.stack([
            entropy[i:i+top_k].mean() for i in range(T - top_k + 1)
        ])  # [T - top_k + 1]

        # --- Find window with lowest mean entropy (most confident segment) ---
        start_idx = torch.argmin(window_entropies).item()
        selected_idx = list(range(start_idx, start_idx + top_k))

        return selected_idx

    def select_key_consec_frames_temporal(self, video_feats, adapted_embeds, top_k=16):
        """
        Select top-K consecutive frames with lowest entropy calculated after temporal aggregation.
        Args:
            video_feats: [T, D] CLIP visual frame features (normalized)
            adapted_embeds: [num_AUs, D] AU text embeddings (after adapter + norm)
            top_k: number of consecutive frames to keep
        Returns:
            List[int]: indices of selected consecutive frames
        """
        T = video_feats.size(0)
        
        # If video is shorter than window size, return all frames
        if T <= top_k:
            return list(range(T))

        window_entropies = []

        # Iterate through all possible consecutive windows
        for i in range(T - top_k + 1):
            # 1. Select window features: [top_k, D]
            window_feats = video_feats[i : i + top_k]
            
            # 2. Add batch dimension: [1, top_k, D]
            # The temporal module snippet expects [B, D, T] input (transposed)
            window_feats_batch = window_feats.unsqueeze(0) 
            
            # 3. Apply Temporal Module
            # Input transposes to [1, D, top_k]
            v = self.temporal(window_feats_batch.transpose(1, 2))
            v = F.gelu(v)
            
            # 4. Aggregate temporally: [1, D]
            visual_embeds = v.mean(dim=-1)
            
            # Normalize the aggregated visual embedding
            visual_embeds = F.normalize(visual_embeds, dim=-1)
            
            # 5. Compute AU Similarity: [1, num_AUs]
            # adapted_embeds is [num_AUs, D], so we transpose it
            au_sim = visual_embeds @ adapted_embeds.T
            
            # 6. Apply Temporal Classifier to get logits: [1, num_classes]
            logits = self.temporal_classifier(au_sim)
            
            # 7. Compute Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) # Scalar [1]
            
            window_entropies.append(entropy.item())

        # --- Find window with lowest entropy (most confident aggregated segment) ---
        start_idx = torch.argmin(torch.tensor(window_entropies)).item()
        selected_idx = list(range(start_idx, start_idx + top_k))

        return selected_idx

    @torch.no_grad()
    def extract_opensmile_batch(
        self,
        audio_arr: torch.Tensor,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """
        Extract processed OpenSMILE features for a batch of audio waveforms.

        Args:
            audio_arr:
                Batched mono waveforms with shape [N, T].
                Example: [N, 240000].

            sample_rate:
                Sampling rate of the waveforms. Defaults to
                self.audio_sample_rate.

        Returns:
            base_audio_embeds:
                Processed OpenSMILE features with shape [N, 88].

                Processing:
                    waveform
                    -> OpenSMILE eGeMAPSv02
                    -> source-domain standardization
                    -> optional activation threshold
                    -> L2 normalization
        """

        if sample_rate is None:
            sample_rate = self.audio_sample_rate

        if not isinstance(audio_arr, torch.Tensor):
            raise TypeError(
                f"audio_arr must be a torch.Tensor, got {type(audio_arr)}"
            )

        # Allow one waveform [T] by adding a batch dimension.
        if audio_arr.ndim == 1:
            audio_arr = audio_arr.unsqueeze(0)

        if audio_arr.ndim != 2:
            raise ValueError(
                "Expected audio_arr with shape [N, T], "
                f"but received {tuple(audio_arr.shape)}"
            )

        batch_size, num_samples = audio_arr.shape

        if num_samples == 0:
            raise ValueError("Received an empty audio waveform.")

        # OpenSMILE runs on CPU/NumPy.
        audio_np = (
            audio_arr
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .numpy()
        )

        raw_feature_list = []

        for batch_idx in range(batch_size):
            waveform = audio_np[batch_idx]

            waveform = np.nan_to_num(
                waveform,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

            # OpenSMILE expects one mono waveform with shape [T].
            feature_df = self.opensmile_model.process_signal(
                waveform,
                sample_rate,
            )

            feature = (
                feature_df
                .to_numpy(dtype=np.float32)
                .reshape(-1)
            )

            if feature.shape[0] != self.audio_feature_dim:
                raise RuntimeError(
                    f"OpenSMILE returned {feature.shape[0]} features "
                    f"for sample {batch_idx}, but expected "
                    f"{self.audio_feature_dim}."
                )

            feature = np.nan_to_num(
                feature,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            raw_feature_list.append(feature)

        # [N, 88]
        raw_features = np.stack(
            raw_feature_list,
            axis=0,
        )

        # Move features to the same device as the fixed dictionary.
        raw_features = torch.from_numpy(raw_features).to(
            device=self.opensmile_mean.device,
            dtype=torch.float32,
        )

        # Apply the exact source-domain scaler used to create the dictionary.
        base_audio_embeds = (
            raw_features - self.opensmile_mean.unsqueeze(0)
        ) / self.opensmile_scale.unsqueeze(0).clamp_min(1e-8)

        # Apply the same activation threshold used during dictionary creation.
        # # threshold=0.0 means all features are retained.
        # if self.activation_threshold > 0.0:
        #     if self.positive_only:
        #         activation_mask = (
        #             base_audio_embeds >= self.activation_threshold
        #         )
        #     else:
        #         activation_mask = (
        #             base_audio_embeds.abs() >= self.activation_threshold
        #         )

        #     base_audio_embeds = torch.where(
        #         activation_mask,
        #         base_audio_embeds,
        #         torch.zeros_like(base_audio_embeds),
        #     )

        # Match the normalized feature space used by the dictionary.
        base_audio_embeds = F.normalize(
            base_audio_embeds,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return base_audio_embeds

    def audio_similarity_to_class_logits(self, audio_similarity, reduce="max",):
        """
        Convert audio dictionary similarity [B, K]
        into class logits [B, num_classes] using fixed dictionary labels.
        """

        B = audio_similarity.shape[0]

        audio_class_logits = []

        for class_id in range(self.num_classes):
            mask = self.audio_cluster_labels == class_id

            if mask.sum() == 0:
                class_score = torch.full(
                    (B,),
                    -1e4,
                    device=audio_similarity.device,
                    dtype=audio_similarity.dtype,
                )
            else:
                class_sims = audio_similarity[:, mask]

                if reduce == "max":
                    class_score = class_sims.max(dim=1).values
                elif reduce == "mean":
                    class_score = class_sims.mean(dim=1)
                elif reduce == "logsumexp":
                    class_score = torch.logsumexp(class_sims, dim=1)
                else:
                    raise ValueError(f"Unknown reduce: {reduce}")

            audio_class_logits.append(class_score)

        audio_class_logits = torch.stack(
            audio_class_logits,
            dim=1,
        )  # [B, num_classes]

        return audio_class_logits

    def compute_audio_dictionary_similarity(self, processed_opensmile_features):
        """
        Compute cosine similarity with the unchanged audio dictionary.

        Args:
            processed_opensmile_features:
                Tensor [B, 88].

                These features must already use the same:
                    - source StandardScaler,
                    - activation threshold,
                    - L2 processing

                used when constructing the dictionary.

        Returns:
            audio_similarity:
                Tensor [B, N], where N is the number of dictionary atoms.
        """

        if processed_opensmile_features.ndim == 1:
            processed_opensmile_features = (
                processed_opensmile_features.unsqueeze(0)
            )

        if processed_opensmile_features.shape[-1] != self.audio_feature_dim:
            raise ValueError(
                f"Expected audio feature dimension {self.audio_feature_dim}, "
                f"but received {processed_opensmile_features.shape[-1]}."
            )

        # OpenSMILE features are fixed, so no gradient is required here.
        audio_features = processed_opensmile_features.detach().float()

        audio_features = F.normalize(
            audio_features,
            p=2,
            dim=-1,
        )

        # audio_dictionary is a fixed registered buffer [N, 88].
        audio_similarity = (
            audio_features @ self.audio_dictionary.T
        )

        return audio_similarity
    
    @torch.no_grad()
    def normalized_entropy(prob, eps=1e-8):
        """
        prob: [B, C]

        Returns:
            entropy: [B]
            0 = confident
            1 = uncertain
        """
        entropy = -(
            prob * torch.log(prob.clamp_min(eps))
        ).sum(dim=-1)

        return entropy / math.log(prob.shape[-1])
    @torch.no_grad()
    def estimate_subject_modality_weights(self,
        visual_prob_all,
        audio_prob_all,
        eps=1e-8,
        weight_temperature=0.2,
    ):
        """
        Estimate one visual and audio weight for a target subject.

        Returns:
            visual_weight: scalar
            audio_weight: scalar
        """
        visual_entropy = -(
            visual_prob_all * torch.log(visual_prob_all.clamp_min(eps))
        ).sum(dim=-1)
        visual_entropy = (visual_entropy / math.log(visual_prob_all.shape[-1])).mean()

        # visual_entropy = self.normalized_entropy(
        #     visual_prob_all
        # ).mean()

        audio_entropy = -(
            audio_prob_all * torch.log(audio_prob_all.clamp_min(eps))
        ).sum(dim=-1)
        audio_entropy = (audio_entropy / math.log(audio_prob_all.shape[-1])).mean()

        # audio_entropy = self.normalized_entropy(
        #     audio_prob_all
        # ).mean()

        # Negative entropy is used as a confidence score.
        confidence_scores = torch.stack(
            [
                -visual_entropy,
                -audio_entropy,
            ],
            dim=-1,
        )  # [B, 2]

        modality_weights = F.softmax(
            confidence_scores / weight_temperature,
            dim=-1,
        )

        visual_weight = modality_weights[0]
        audio_weight = modality_weights[1]

        fused_prob = (
            visual_weight * visual_prob_all
            + audio_weight * audio_prob_all
        )

        return {
            "fused_prob": fused_prob,
            "visual_weight": visual_weight,
            "audio_weight": audio_weight,
            "visual_entropy": visual_entropy,
            "audio_entropy": audio_entropy,
        }



    def certainty_aware_fusion(self,
        visual_prob,
        audio_prob,
        base_visual_weight=0.5,
        base_audio_weight=0.5,
        confidence_threshold=0.80,
        boost_strength=3.0,
        eps=1e-8,
    ):
        """
        Args:
            visual_prob:
                Visual class probabilities, shape [C] or [B, C].

            audio_prob:
                Audio class probabilities, shape [C] or [B, C].

            base_visual_weight:
                Default visual modality weight.

            base_audio_weight:
                Default audio modality weight.

            confidence_threshold:
                A modality is boosted only if its maximum probability
                exceeds this threshold.

            boost_strength:
                Controls how strongly a confident modality is boosted.

        Returns:
            fused_prob:
                Fused class probabilities.

            visual_weight:
                Final visual weight.

            audio_weight:
                Final audio weight.
        """

        if visual_prob.ndim == 1:
            visual_prob = visual_prob.unsqueeze(0)

        if audio_prob.ndim == 1:
            audio_prob = audio_prob.unsqueeze(0)

        # Maximum class probability for each modality
        visual_confidence = visual_prob.max(
            dim=-1
        ).values

        audio_confidence = audio_prob.max(
            dim=-1
        ).values

        # Confidence above the threshold, normalized to [0, 1]
        visual_excess = (
            (
                visual_confidence
                - confidence_threshold
            )
            / (1.0 - confidence_threshold)
        ).clamp(0.0, 1.0)

        audio_excess = (
            (
                audio_confidence
                - confidence_threshold
            )
            / (1.0 - confidence_threshold)
        ).clamp(0.0, 1.0)

        # Start from the generic/default modality weights
        base_log_weights = torch.tensor(
            [
                base_visual_weight,
                base_audio_weight,
            ],
            device=visual_prob.device,
            dtype=visual_prob.dtype,
        ).clamp_min(eps).log()

        # Add a boost only when confidence exceeds the threshold
        modality_scores = torch.stack(
            [
                base_log_weights[0]
                + boost_strength * visual_excess,

                base_log_weights[1]
                + boost_strength * audio_excess,
            ],
            dim=-1,
        )

        modality_weights = F.softmax(
            modality_scores,
            dim=-1,
        )

        visual_weight = modality_weights[:, 0:1]
        audio_weight = modality_weights[:, 1:2]

        fused_prob = (
            visual_weight * visual_prob
            + audio_weight * audio_prob
        )

        return {
            "fused_prob": fused_prob,
            "visual_weight": visual_weight,
            "audio_weight": audio_weight,
            "visual_confidence": visual_confidence,
            "audio_confidence": audio_confidence,
    }



    def entropy_confidence(self, prob: torch.Tensor, eps: float = 1e-8):
        """
        Convert class probabilities into normalized confidence.

        Args:
            prob: [C] or [B, C]

        Returns:
            confidence: [B]
                0 = maximally uncertain
                1 = maximally confident
        """
        if prob.ndim == 1:
            prob = prob.unsqueeze(0)

        prob = prob.clamp_min(eps)
        prob = prob / prob.sum(dim=-1, keepdim=True)

        entropy = -(
            prob * torch.log(prob)
        ).sum(dim=-1)

        max_entropy = math.log(prob.shape[-1])

        confidence = 1.0 - entropy / max_entropy

        return confidence


    def conservative_confidence_fusion(self,
        visual_prob: torch.Tensor,
        audio_prob: torch.Tensor,
        base_audio_weight: float = 0.72,
        min_audio_weight: float = 0.60,
        max_audio_weight: float = 0.90,
        high_confidence: float = 0.75,
        audio_margin: float = 0.10,
        visual_margin: float = 0.15,
    ):
        """
        Audio-dominant fusion with bounded confidence-based corrections.

        The generic audio weight is preserved unless one modality provides
        sufficiently strong evidence.

        Args:
            visual_prob: [C] or [B, C]
            audio_prob:  [C] or [B, C]

        Returns:
            Dictionary containing fused probabilities, modality weights,
            confidence values, and predictions.
        """

        if visual_prob.ndim == 1:
            visual_prob = visual_prob.unsqueeze(0)

        if audio_prob.ndim == 1:
            audio_prob = audio_prob.unsqueeze(0)

        visual_confidence = self.entropy_confidence(
            visual_prob
        )

        audio_confidence = self.entropy_confidence(
            audio_prob
        )

        visual_prediction = visual_prob.argmax(dim=-1)
        audio_prediction = audio_prob.argmax(dim=-1)

        disagreement = (
            visual_prediction != audio_prediction
        )

        # Positive means audio is more confident.
        confidence_gap = (
            audio_confidence - visual_confidence
        )

        audio_weight = torch.full_like(
            audio_confidence,
            fill_value=base_audio_weight,
        )

        # --------------------------------------------------
        # Increase audio weight only when audio is clearly
        # more reliable and visual is not highly confident.
        # --------------------------------------------------
        audio_override = (
            disagreement
            & (audio_confidence >= high_confidence)
            & (visual_confidence < high_confidence)
            & (confidence_gap >= audio_margin)
        )

        audio_strength = (
            (confidence_gap - audio_margin)
            / (1.0 - audio_margin)
        ).clamp(0.0, 1.0)

        audio_adjustment = (
            max_audio_weight - base_audio_weight
        ) * audio_strength

        audio_weight = torch.where(
            audio_override,
            audio_weight + audio_adjustment,
            audio_weight,
        )

        # --------------------------------------------------
        # Increase visual influence only when visual is
        # clearly more reliable and audio is uncertain.
        # --------------------------------------------------
        visual_override = (
            disagreement
            & (visual_confidence >= high_confidence)
            & (audio_confidence < high_confidence)
            & (-confidence_gap >= visual_margin)
        )

        visual_strength = (
            (-confidence_gap - visual_margin)
            / (1.0 - visual_margin)
        ).clamp(0.0, 1.0)

        visual_adjustment = (
            base_audio_weight - min_audio_weight
        ) * visual_strength

        audio_weight = torch.where(
            visual_override,
            audio_weight - visual_adjustment,
            audio_weight,
        )

        audio_weight = audio_weight.clamp(
            min=min_audio_weight,
            max=max_audio_weight,
        )

        visual_weight = 1.0 - audio_weight

        visual_weight = visual_weight.unsqueeze(-1)
        audio_weight = audio_weight.unsqueeze(-1)

        fused_prob = (
            visual_weight * visual_prob
            + audio_weight * audio_prob
        )

        return {
            "fused_prob": fused_prob,
            "visual_weight": visual_weight,
            "audio_weight": audio_weight,
            "visual_confidence": visual_confidence,
            "audio_confidence": audio_confidence,
            "visual_prediction": visual_prediction,
            "audio_prediction": audio_prediction,
            "disagreement": disagreement,
        }


    def crop_synchronized_audio(self,
        audio: torch.Tensor,
        total_frames: int,
        window_start: int,
        window_size: int,
    ):
        """
        audio: [B, num_samples]
        total_frames: total visual frames, e.g. 30
        window_start: selected visual-window start index
        window_size: selected number of frames
        """
        window_end = min(
            window_start + window_size,
            total_frames,
        )

        num_samples = audio.shape[-1]

        start_sample = round(
            window_start / total_frames * num_samples
        )

        end_sample = round(
            window_end / total_frames * num_samples
        )

        return audio[..., start_sample:end_sample]


    def select_key_consec_frames_multimodal(
        self,
        video_feats,
        audio,
        adapted_embeds,
        top_k=16,
        sample_rate=16000,
        visual_temp=1.0,
        audio_temp=0.2,
        visual_selection_weight=0.5,
        audio_selection_weight=0.5,
    ):
        """
        Select the best synchronized visual-audio consecutive window.

        Selection criteria:
            1. Low visual entropy relative to other visual windows.
            2. Low audio entropy relative to other audio windows.
            3. Prefer windows where visual and audio predict the same class.

        Args:
            video_feats:
                CLIP visual features for all frames.
                Shape: [T, D]

            audio:
                Complete waveform synchronized with all T frames.
                Shape: [1, num_audio_samples] or [num_audio_samples]

            adapted_embeds:
                Adapted AU text embeddings.
                Shape: [num_AUs, D]

            top_k:
                Number of consecutive frames in each candidate window.

            sample_rate:
                Audio sample rate used by OpenSMILE.

            visual_temp:
                Temperature applied to visual logits.

            audio_temp:
                Temperature applied to audio dictionary class scores.

            visual_selection_weight:
                Contribution of visual rank to window selection.

            audio_selection_weight:
                Contribution of audio rank to window selection.

        Required class attributes:
            self.temporal
            self.temporal_classifier
            self.extract_opensmile_batch
            self.audio_dictionary
            self.audio_dictionary_labels

        Returns:
            Dictionary containing selected visual indices, synchronized audio,
            probabilities, entropies, ranks, and diagnostic information.
        """

        # ---------------------------------------------------------
        # Validate input shapes
        # ---------------------------------------------------------
        if video_feats.ndim != 2:
            raise ValueError(
                "video_feats must have shape [T, D], "
                f"but received {tuple(video_feats.shape)}."
            )

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim != 2:
            raise ValueError(
                "audio must have shape [1, S] or [S], "
                f"but received {tuple(audio.shape)}."
            )

        if audio.shape[0] != 1:
            raise ValueError(
                "This function expects one sample at a time. "
                f"Received audio batch size {audio.shape[0]}."
            )

        if adapted_embeds.ndim != 2:
            raise ValueError(
                "adapted_embeds must have shape [num_AUs, D], "
                f"but received {tuple(adapted_embeds.shape)}."
            )

        T = video_feats.shape[0]

        if T == 0:
            raise ValueError("video_feats contains no frames.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # If the sequence is shorter than top_k, use the whole sequence.
        top_k = min(top_k, T)

        number_of_windows = T - top_k + 1
        num_audio_samples = audio.shape[-1]

        # ---------------------------------------------------------
        # Normalize selection weights
        # ---------------------------------------------------------
        selection_weight_sum = (
            visual_selection_weight
            + audio_selection_weight
        )

        if selection_weight_sum <= 0:
            raise ValueError(
                "The sum of visual_selection_weight and "
                "audio_selection_weight must be positive."
            )

        visual_selection_weight = (
            visual_selection_weight
            / selection_weight_sum
        )

        audio_selection_weight = (
            audio_selection_weight
            / selection_weight_sum
        )

        # ---------------------------------------------------------
        # Prepare fixed dictionary information
        # ---------------------------------------------------------
        audio_dictionary = self.audio_dictionary

        if audio_dictionary.ndim != 2:
            raise ValueError(
                "self.audio_dictionary must have shape [K, audio_dim], "
                f"but received {tuple(audio_dictionary.shape)}."
            )

        audio_dictionary = audio_dictionary.to(
            device=video_feats.device,
            dtype=video_feats.dtype,
        )

        audio_dictionary_norm = F.normalize(
            audio_dictionary,
            dim=-1,
        )

        audio_dictionary_labels = (
            self.audio_cluster_labels
            .to(video_feats.device)
            .long()
        )

        if audio_dictionary_labels.ndim != 1:
            raise ValueError(
                "self.audio_dictionary_labels must have shape [K]."
            )

        if (
            audio_dictionary_labels.shape[0]
            != audio_dictionary.shape[0]
        ):
            raise ValueError(
                "The number of audio dictionary labels must match "
                "the number of dictionary atoms."
            )

        adapted_embeds_norm = F.normalize(
            adapted_embeds,
            dim=-1,
        )

        visual_probs_list = []
        audio_probs_list = []

        visual_entropies_list = []
        audio_entropies_list = []

        audio_start_samples = []
        audio_end_samples = []

        # Window selection is discrete, so there is no reason to retain
        # gradients through every candidate window.
        with torch.no_grad():

            for start_idx in range(number_of_windows):

                end_idx = start_idx + top_k

                # =====================================================
                # 1. Visual branch
                # =====================================================

                window_feats = video_feats[
                    start_idx:end_idx
                ]  # [top_k, D]

                window_feats_batch = window_feats.unsqueeze(
                    0
                )  # [1, top_k, D]

                # Temporal module expects [B, D, T].
                temporal_features = self.temporal(
                    window_feats_batch.transpose(1, 2)
                )

                temporal_features = F.gelu(
                    temporal_features
                )

                # Aggregate temporally.
                visual_embeds = temporal_features.mean(
                    dim=-1
                )  # [1, D]

                visual_embeds = F.normalize(
                    visual_embeds,
                    dim=-1,
                )

                # AU similarities.
                au_similarity = (
                    visual_embeds
                    @ adapted_embeds_norm.T
                )  # [1, num_AUs]

                visual_logits = self.temporal_classifier(
                    au_similarity
                )  # [1, num_classes]

                visual_prob = F.softmax(
                    visual_logits / visual_temp,
                    dim=-1,
                )

                num_classes = visual_prob.shape[-1]

                visual_entropy = -(
                    visual_prob
                    * torch.log(
                        visual_prob.clamp_min(1e-8)
                    )
                ).sum(dim=-1)

                visual_entropy = (
                    visual_entropy
                    / math.log(num_classes)
                )

                # =====================================================
                # 2. Crop synchronized audio
                # =====================================================

                # The complete waveform corresponds exactly to all T
                # visual frames, so crop it proportionally.
                start_sample = round(
                    start_idx
                    / T
                    * num_audio_samples
                )

                end_sample = round(
                    end_idx
                    / T
                    * num_audio_samples
                )

                start_sample = max(
                    0,
                    min(
                        start_sample,
                        num_audio_samples - 1,
                    ),
                )

                end_sample = max(
                    start_sample + 1,
                    min(
                        end_sample,
                        num_audio_samples,
                    ),
                )

                audio_window = audio[
                    :,
                    start_sample:end_sample,
                ]

                # =====================================================
                # 3. OpenSMILE audio embedding
                # =====================================================

                audio_embeds_norm = (
                    self.extract_opensmile_batch(
                        audio_window,
                        sample_rate=sample_rate,
                    )
                )  # [1, audio_dim]

                audio_embeds_norm = audio_embeds_norm.to(
                    device=audio_dictionary_norm.device,
                    dtype=audio_dictionary_norm.dtype,
                )

                # Safe even when extract_opensmile_batch already performs
                # L2 normalization.
                audio_embeds_norm = F.normalize(
                    audio_embeds_norm,
                    dim=-1,
                )

                if (
                    audio_embeds_norm.shape[-1]
                    != audio_dictionary_norm.shape[-1]
                ):
                    raise ValueError(
                        "OpenSMILE embedding dimension does not match "
                        "the audio dictionary dimension. Received "
                        f"{audio_embeds_norm.shape[-1]} and "
                        f"{audio_dictionary_norm.shape[-1]}."
                    )

                # =====================================================
                # 4. Audio dictionary similarity
                # =====================================================

                audio_similarity = (
                    audio_embeds_norm
                    @ audio_dictionary_norm.T
                )  # [1, K]

                # Convert dictionary-atom similarities to class scores.
                audio_class_scores = torch.full(
                    (
                        audio_similarity.shape[0],
                        num_classes,
                    ),
                    fill_value=-1e4,
                    device=audio_similarity.device,
                    dtype=audio_similarity.dtype,
                )

                for class_idx in range(num_classes):

                    class_mask = (
                        audio_dictionary_labels
                        == class_idx
                    )

                    if class_mask.any():
                        # Nearest dictionary atom belonging to this class.
                        audio_class_scores[
                            :,
                            class_idx,
                        ] = (
                            audio_similarity[
                                :,
                                class_mask,
                            ]
                            .max(dim=-1)
                            .values
                        )

                audio_prob = F.softmax(
                    audio_class_scores / audio_temp,
                    dim=-1,
                )

                audio_entropy = -(
                    audio_prob
                    * torch.log(
                        audio_prob.clamp_min(1e-8)
                    )
                ).sum(dim=-1)

                audio_entropy = (
                    audio_entropy
                    / math.log(num_classes)
                )

                # =====================================================
                # 5. Store candidate-window information
                # =====================================================

                visual_probs_list.append(
                    visual_prob.squeeze(0)
                )

                audio_probs_list.append(
                    audio_prob.squeeze(0)
                )

                visual_entropies_list.append(
                    visual_entropy.squeeze(0)
                )

                audio_entropies_list.append(
                    audio_entropy.squeeze(0)
                )

                audio_start_samples.append(
                    start_sample
                )

                audio_end_samples.append(
                    end_sample
                )

            # ---------------------------------------------------------
            # Stack results from all candidate windows
            # ---------------------------------------------------------
            visual_window_probs = torch.stack(
                visual_probs_list,
                dim=0,
            )  # [W, C]

            audio_window_probs = torch.stack(
                audio_probs_list,
                dim=0,
            )  # [W, C]

            visual_entropies = torch.stack(
                visual_entropies_list,
                dim=0,
            )  # [W]

            audio_entropies = torch.stack(
                audio_entropies_list,
                dim=0,
            )  # [W]

            # ---------------------------------------------------------
            # Rank windows independently for each modality
            # ---------------------------------------------------------
            visual_order = torch.argsort(
                visual_entropies
            )

            audio_order = torch.argsort(
                audio_entropies
            )

            visual_ranks = torch.empty(
                number_of_windows,
                device=visual_entropies.device,
                dtype=torch.float32,
            )

            audio_ranks = torch.empty(
                number_of_windows,
                device=audio_entropies.device,
                dtype=torch.float32,
            )

            visual_ranks[visual_order] = torch.arange(
                number_of_windows,
                device=visual_entropies.device,
                dtype=torch.float32,
            )

            audio_ranks[audio_order] = torch.arange(
                number_of_windows,
                device=audio_entropies.device,
                dtype=torch.float32,
            )

            # Normalize ranks to [0, 1].
            rank_denominator = max(
                number_of_windows - 1,
                1,
            )

            visual_ranks = (
                visual_ranks / rank_denominator
            )

            audio_ranks = (
                audio_ranks / rank_denominator
            )

            # Lower combined rank is better.
            combined_rank = (
                visual_selection_weight
                * visual_ranks
                + audio_selection_weight
                * audio_ranks
            )

            # ---------------------------------------------------------
            # Prefer windows where both modalities predict same class
            # ---------------------------------------------------------
            visual_predictions = (
                visual_window_probs.argmax(dim=-1)
            )

            audio_predictions = (
                audio_window_probs.argmax(dim=-1)
            )

            agreement = (
                visual_predictions
                == audio_predictions
            )

            if agreement.any():
                # Among agreeing windows, select the one with the best
                # combined visual/audio rank.
                selection_scores = combined_rank.masked_fill(
                    ~agreement,
                    float("inf"),
                )

                used_agreement_fallback = False

            else:
                # No candidate window has modality agreement.
                # Select the best combined-rank window.
                selection_scores = combined_rank

                used_agreement_fallback = True

            selected_start = int(
                selection_scores.argmin().item()
            )

        # -------------------------------------------------------------
        # Construct selected synchronized pair
        # -------------------------------------------------------------
        selected_end = selected_start + top_k

        selected_idx = list(
            range(
                selected_start,
                selected_end,
            )
        )

        selected_audio_start = audio_start_samples[
            selected_start
        ]

        selected_audio_end = audio_end_samples[
            selected_start
        ]

        selected_audio = audio[
            :,
            selected_audio_start:selected_audio_end,
        ]

        selected_video_feats = video_feats[
            selected_start:selected_end
        ]

        return {
            "selected_idx": selected_idx,
            "selected_start": selected_start,
            "selected_end": selected_end,

            "selected_video_feats": selected_video_feats,
            "selected_audio": selected_audio,

            "selected_audio_start": selected_audio_start,
            "selected_audio_end": selected_audio_end,

            "selected_visual_prob": (
                visual_window_probs[
                    selected_start:selected_start + 1
                ]
            ),

            "selected_audio_prob": (
                audio_window_probs[
                    selected_start:selected_start + 1
                ]
            ),

            "selected_visual_entropy": (
                visual_entropies[selected_start]
            ),

            "selected_audio_entropy": (
                audio_entropies[selected_start]
            ),

            "selected_visual_prediction": (
                visual_predictions[selected_start]
            ),

            "selected_audio_prediction": (
                audio_predictions[selected_start]
            ),

            "selected_modalities_agree": (
                agreement[selected_start]
            ),

            "used_agreement_fallback": (
                used_agreement_fallback
            ),

            # All candidate-window diagnostics
            "visual_window_probs": visual_window_probs,
            "audio_window_probs": audio_window_probs,

            "visual_entropies": visual_entropies,
            "audio_entropies": audio_entropies,

            "visual_ranks": visual_ranks,
            "audio_ranks": audio_ranks,
            "combined_rank": combined_rank,

            "visual_predictions": visual_predictions,
            "audio_predictions": audio_predictions,

            "agreement": agreement,
            "selection_scores": selection_scores,
        }


    def select_key_consec_frames_audio_guided(
        self,
        video_feats,
        audio,
        adapted_embeds,
        top_k=16,
        sample_rate=16000,
        visual_temp=1.0,
        audio_temp=0.2,
        audio_guidance_strength=0.10,
        audio_atom_reduction="max",
        audio_atom_topk=2,
        audio_atom_temperature=0.1,
    ):
        """
        Select the most informative consecutive visual window using:

            selection_score =
                visual_entropy
                + audio_guidance_strength * visual_audio_JS_divergence

        The complete audio waveform is processed once. Its prediction softly
        guides selection among the candidate visual windows.

        Audio dictionary atoms assigned to the same class can be aggregated
        using max, mean, top-k mean, or a softmax-weighted mean.

        Args:
            video_feats:
                Frame-level visual features.
                Shape: [T, D]

            audio:
                Complete waveform synchronized with the T visual frames.
                Shape: [1, S] or [S]

            adapted_embeds:
                Adapted AU text embeddings.
                Shape: [num_AUs, D]

            top_k:
                Number of consecutive visual frames to select.

            sample_rate:
                Audio sample rate used by OpenSMILE.

            visual_temp:
                Temperature applied to visual logits.

            audio_temp:
                Temperature applied to final audio class scores.

            audio_guidance_strength:
                Strength of audio guidance during window selection.

                0.0:
                    Original visual-only entropy selection.

                0.10:
                    Weak audio guidance.

                0.25:
                    Moderate audio guidance.

                0.50:
                    Strong audio guidance.

            audio_atom_reduction:
                Method used to aggregate similarities from dictionary atoms
                belonging to the same class.

                "max":
                    Use the most similar atom.

                "mean":
                    Average all atoms belonging to the class.

                "topk_mean":
                    Average the top audio_atom_topk most similar atoms.

                "softmax":
                    Compute a similarity-weighted average of all atoms.

            audio_atom_topk:
                Number of atoms used by "topk_mean".

            audio_atom_temperature:
                Temperature used by "softmax" atom aggregation.

                Smaller values:
                    More similar to max aggregation.

                Larger values:
                    More uniform contribution from all class atoms.

        Required class attributes:
            self.temporal
            self.temporal_classifier
            self.extract_opensmile_batch
            self.audio_dictionary
            self.audio_cluster_labels

        Returns:
            Dictionary containing selected indices, visual features,
            synchronized audio, probabilities, class scores, entropies,
            JS divergences, and selection diagnostics.
        """

        # =============================================================
        # 1. Validate inputs
        # =============================================================
        if video_feats.ndim != 2:
            raise ValueError(
                "video_feats must have shape [T, D], "
                f"but received {tuple(video_feats.shape)}."
            )

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim != 2:
            raise ValueError(
                "audio must have shape [1, S] or [S], "
                f"but received {tuple(audio.shape)}."
            )

        if audio.shape[0] != 1:
            raise ValueError(
                "This function expects one audio sample at a time, "
                f"but received batch size {audio.shape[0]}."
            )

        if adapted_embeds.ndim != 2:
            raise ValueError(
                "adapted_embeds must have shape [num_AUs, D], "
                f"but received {tuple(adapted_embeds.shape)}."
            )

        if visual_temp <= 0:
            raise ValueError(
                "visual_temp must be greater than zero."
            )

        if audio_temp <= 0:
            raise ValueError(
                "audio_temp must be greater than zero."
            )

        if audio_guidance_strength < 0:
            raise ValueError(
                "audio_guidance_strength cannot be negative."
            )

        valid_reductions = {
            "max",
            "mean",
            "topk_mean",
            "softmax",
        }

        if audio_atom_reduction not in valid_reductions:
            raise ValueError(
                f"Unsupported audio_atom_reduction="
                f"'{audio_atom_reduction}'. Choose from "
                f"{sorted(valid_reductions)}."
            )

        if audio_atom_topk <= 0:
            raise ValueError(
                "audio_atom_topk must be greater than zero."
            )

        if audio_atom_temperature <= 0:
            raise ValueError(
                "audio_atom_temperature must be greater than zero."
            )

        T = video_feats.shape[0]

        if T == 0:
            raise ValueError(
                "video_feats contains no frames."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if audio.shape[-1] == 0:
            raise ValueError(
                "audio contains no waveform samples."
            )

        top_k = min(top_k, T)

        number_of_windows = T - top_k + 1
        num_audio_samples = audio.shape[-1]

        # =============================================================
        # Helper: aggregate dictionary atoms into class scores
        # =============================================================
        def aggregate_audio_atoms(
            audio_similarity,
            dictionary_labels,
            num_classes,
        ):
            """
            Args:
                audio_similarity:
                    Query-to-dictionary similarities.
                    Shape: [B, K]

                dictionary_labels:
                    Class label for each dictionary atom.
                    Shape: [K]

                num_classes:
                    Number of output classes.

            Returns:
                audio_class_scores:
                    Shape: [B, num_classes]
            """

            batch_size = audio_similarity.shape[0]

            # Preserve the behavior of the original function when a class
            # has no assigned dictionary atom.
            audio_class_scores = torch.full(
                size=(batch_size, num_classes),
                fill_value=-1e4,
                device=audio_similarity.device,
                dtype=audio_similarity.dtype,
            )

            for class_idx in range(num_classes):

                class_mask = (
                    dictionary_labels == class_idx
                )

                if not class_mask.any():
                    continue

                class_atom_scores = audio_similarity[
                    :,
                    class_mask,
                ]  # [B, number_of_class_atoms]

                if audio_atom_reduction == "max":

                    class_score = (
                        class_atom_scores
                        .max(dim=-1)
                        .values
                    )

                elif audio_atom_reduction == "mean":

                    class_score = (
                        class_atom_scores
                        .mean(dim=-1)
                    )

                elif audio_atom_reduction == "topk_mean":

                    current_topk = min(
                        audio_atom_topk,
                        class_atom_scores.shape[-1],
                    )

                    topk_scores = (
                        class_atom_scores
                        .topk(
                            k=current_topk,
                            dim=-1,
                            largest=True,
                        )
                        .values
                    )

                    class_score = (
                        topk_scores.mean(dim=-1)
                    )

                elif audio_atom_reduction == "softmax":

                    atom_weights = F.softmax(
                        class_atom_scores
                        / audio_atom_temperature,
                        dim=-1,
                    )

                    class_score = (
                        atom_weights
                        * class_atom_scores
                    ).sum(dim=-1)

                audio_class_scores[
                    :,
                    class_idx,
                ] = class_score

            return audio_class_scores

        # Window selection is discrete. Gradients through all candidate
        # windows are unnecessary.
        with torch.no_grad():

            # =========================================================
            # 2. Construct every consecutive visual window
            # =========================================================
            # unfold:
            #   [W, D, top_k]
            #
            # permute:
            #   [W, top_k, D]
            visual_windows = (
                video_feats
                .unfold(
                    dimension=0,
                    size=top_k,
                    step=1,
                )
                .permute(0, 2, 1)
                .contiguous()
            )

            # =========================================================
            # 3. Process all visual windows in one batch
            # =========================================================
            temporal_features = self.temporal(
                visual_windows.transpose(1, 2)
            )

            temporal_features = F.gelu(
                temporal_features
            )

            visual_embeds = temporal_features.mean(
                dim=-1
            )  # [W, D]

            visual_embeds = F.normalize(
                visual_embeds,
                dim=-1,
            )

            adapted_embeds_norm = F.normalize(
                adapted_embeds,
                dim=-1,
            )

            au_similarity = (
                visual_embeds
                @ adapted_embeds_norm.T
            )  # [W, num_AUs]

            visual_logits = self.temporal_classifier(
                au_similarity
            )  # [W, C]

            visual_window_probs = F.softmax(
                visual_logits / visual_temp,
                dim=-1,
            )  # [W, C]

            num_classes = visual_window_probs.shape[-1]

            if num_classes < 2:
                raise ValueError(
                    "At least two output classes are required."
                )

            # Normalized visual entropy.
            visual_entropies = -(
                visual_window_probs
                * torch.log(
                    visual_window_probs.clamp_min(1e-8)
                )
            ).sum(dim=-1)

            visual_entropies = (
                visual_entropies
                / math.log(num_classes)
            )  # [W]

            # =========================================================
            # 4. Extract OpenSMILE features from complete audio once
            # =========================================================
            audio_embeds_norm = (
                self.extract_opensmile_batch(
                    audio,
                    sample_rate=sample_rate,
                )
            )  # [1, audio_dim]

            if audio_embeds_norm.ndim == 1:
                audio_embeds_norm = (
                    audio_embeds_norm.unsqueeze(0)
                )

            # =========================================================
            # 5. Prepare fixed audio dictionary
            # =========================================================
            audio_dictionary = self.audio_dictionary.to(
                device=audio_embeds_norm.device,
                dtype=audio_embeds_norm.dtype,
            )

            if audio_dictionary.ndim != 2:
                raise ValueError(
                    "self.audio_dictionary must have shape "
                    "[num_atoms, audio_dim]."
                )

            if (
                audio_embeds_norm.shape[-1]
                != audio_dictionary.shape[-1]
            ):
                raise ValueError(
                    "OpenSMILE feature dimension does not match "
                    "the audio dictionary dimension: "
                    f"{audio_embeds_norm.shape[-1]} versus "
                    f"{audio_dictionary.shape[-1]}."
                )

            audio_embeds_norm = F.normalize(
                audio_embeds_norm,
                dim=-1,
            )

            audio_dictionary_norm = F.normalize(
                audio_dictionary,
                dim=-1,
            )

            # Query-to-dictionary cosine similarity.
            audio_similarity = (
                audio_embeds_norm
                @ audio_dictionary_norm.T
            )  # [1, K]

            dictionary_labels = (
                self.audio_cluster_labels
                .to(audio_similarity.device)
                .long()
            )

            if dictionary_labels.ndim != 1:
                raise ValueError(
                    "self.audio_cluster_labels must have "
                    "shape [num_atoms]."
                )

            if (
                dictionary_labels.shape[0]
                != audio_dictionary.shape[0]
            ):
                raise ValueError(
                    "The number of audio cluster labels must match "
                    "the number of audio dictionary atoms."
                )

            # =========================================================
            # 6. Aggregate audio atoms into class scores
            # =========================================================
            audio_class_scores = aggregate_audio_atoms(
                audio_similarity=audio_similarity,
                dictionary_labels=dictionary_labels,
                num_classes=num_classes,
            )  # [1, C]

            audio_prob = F.softmax(
                audio_class_scores / audio_temp,
                dim=-1,
            )  # [1, C]

            # Move audio outputs to the visual tensor device.
            audio_class_scores = audio_class_scores.to(
                device=visual_window_probs.device,
                dtype=visual_window_probs.dtype,
            )

            audio_prob = audio_prob.to(
                device=visual_window_probs.device,
                dtype=visual_window_probs.dtype,
            )

            # The complete-audio prediction guides every visual window.
            audio_prob_expanded = audio_prob.expand(
                number_of_windows,
                -1,
            )  # [W, C]

            # =========================================================
            # 7. Calculate visual-audio JS divergence
            # =========================================================
            mixture_prob = 0.5 * (
                visual_window_probs
                + audio_prob_expanded
            )

            visual_to_mixture = (
                visual_window_probs
                * (
                    torch.log(
                        visual_window_probs.clamp_min(1e-8)
                    )
                    - torch.log(
                        mixture_prob.clamp_min(1e-8)
                    )
                )
            ).sum(dim=-1)

            audio_to_mixture = (
                audio_prob_expanded
                * (
                    torch.log(
                        audio_prob_expanded.clamp_min(1e-8)
                    )
                    - torch.log(
                        mixture_prob.clamp_min(1e-8)
                    )
                )
            ).sum(dim=-1)

            js_divergence = 0.5 * (
                visual_to_mixture
                + audio_to_mixture
            )

            # Normalize JS divergence to approximately [0, 1].
            js_divergence = (
                js_divergence / math.log(2.0)
            )  # [W]

            # =========================================================
            # 8. Calculate final window-selection score
            # =========================================================
            selection_scores = (
                visual_entropies
                + audio_guidance_strength
                * js_divergence
            )

            selected_start = int(
                selection_scores.argmin().item()
            )

        # =============================================================
        # 9. Construct selected visual window
        # =============================================================
        selected_end = selected_start + top_k

        selected_idx = list(
            range(
                selected_start,
                selected_end,
            )
        )

        selected_video_feats = video_feats[
            selected_start:selected_end
        ]

        # =============================================================
        # 10. Crop synchronized audio
        # =============================================================
        selected_audio_start = round(
            selected_start
            / T
            * num_audio_samples
        )

        selected_audio_end = round(
            selected_end
            / T
            * num_audio_samples
        )

        selected_audio_start = max(
            0,
            min(
                selected_audio_start,
                num_audio_samples - 1,
            ),
        )

        selected_audio_end = max(
            selected_audio_start + 1,
            min(
                selected_audio_end,
                num_audio_samples,
            ),
        )

        selected_audio = audio[
            :,
            selected_audio_start:selected_audio_end,
        ]

        # =============================================================
        # 11. Return selection and diagnostics
        # =============================================================
        return {
            "selected_idx": selected_idx,
            "selected_start": selected_start,
            "selected_end": selected_end,

            "selected_video_feats": (
                selected_video_feats
            ),

            "selected_audio": selected_audio,
            "selected_audio_start": (
                selected_audio_start
            ),
            "selected_audio_end": (
                selected_audio_end
            ),

            "selected_visual_prob": (
                visual_window_probs[
                    selected_start:selected_start + 1
                ]
            ),

            # Calculated from the complete waveform.
            "audio_prob": audio_prob,

            "audio_class_scores": (
                audio_class_scores
            ),

            "audio_similarity": (
                audio_similarity
            ),

            "selected_visual_entropy": (
                visual_entropies[selected_start]
            ),

            "selected_js_divergence": (
                js_divergence[selected_start]
            ),

            "selected_selection_score": (
                selection_scores[selected_start]
            ),

            "audio_atom_reduction": (
                audio_atom_reduction
            ),

            # Candidate-window diagnostics
            "visual_window_probs": (
                visual_window_probs
            ),

            "visual_entropies": (
                visual_entropies
            ),

            "js_divergence": (
                js_divergence
            ),

            "selection_scores": (
                selection_scores
            ),
        }


    def select_key_consec_frames_multimodal_topm(
        self,
        video_feats,
        audio,
        adapted_embeds,
        top_k=16,
        top_m=3,
        sample_rate=16000,
        visual_temp=1.0,
        audio_temp=0.2,
        visual_weight=0.28,
        audio_weight=0.72,
        audio_atom_reduction="max",
        audio_atom_topk=2,
        audio_atom_temperature=0.1,
    ):
        """
        Select a synchronized visual-audio temporal window using two stages.

        Stage 1:
            Evaluate every consecutive visual window and retain the top-M
            windows with the lowest visual prediction entropy.

        Stage 2:
            Extract the synchronized audio for only those top-M windows,
            compute local audio predictions, fuse visual and audio
            probabilities, and select the window with the lowest fused
            prediction entropy.

        Args:
            video_feats:
                Frame-level visual features.
                Shape: [T, D]

            audio:
                Complete waveform synchronized with all T visual frames.
                Shape: [1, S] or [S]

            adapted_embeds:
                Adapted AU text embeddings.
                Shape: [num_AUs, D]

            top_k:
                Number of consecutive visual frames in each window.

            top_m:
                Number of visually informative windows passed to the
                audio reranking stage.

            sample_rate:
                Audio sampling rate used for OpenSMILE extraction.

            visual_temp:
                Temperature applied to visual logits.

            audio_temp:
                Temperature applied to audio class scores.

            visual_weight:
                Visual contribution during candidate reranking.

            audio_weight:
                Audio contribution during candidate reranking.

            audio_atom_reduction:
                How dictionary atoms belonging to the same class are
                converted into one class score.

                Supported:
                    "max"
                    "mean"
                    "topk_mean"
                    "softmax"

            audio_atom_topk:
                Number of closest atoms used when
                audio_atom_reduction="topk_mean".

            audio_atom_temperature:
                Temperature for weighting atoms when
                audio_atom_reduction="softmax".

                Smaller:
                    behaves more like max.

                Larger:
                    distributes weight more evenly across atoms.

        Required class attributes:
            self.temporal
            self.temporal_classifier
            self.extract_opensmile_batch
            self.audio_dictionary
            self.audio_dictionary_labels

        Returns:
            Dictionary containing the selected visual indices, selected
            visual features, synchronized audio waveform, modality
            probabilities, and candidate diagnostics.
        """

        # =============================================================
        # 1. Validate inputs
        # =============================================================
        if video_feats.ndim != 2:
            raise ValueError(
                "video_feats must have shape [T, D], "
                f"but received {tuple(video_feats.shape)}."
            )

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim != 2 or audio.shape[0] != 1:
            raise ValueError(
                "audio must have shape [1, S] or [S], "
                f"but received {tuple(audio.shape)}."
            )

        if adapted_embeds.ndim != 2:
            raise ValueError(
                "adapted_embeds must have shape [num_AUs, D], "
                f"but received {tuple(adapted_embeds.shape)}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if top_m <= 0:
            raise ValueError("top_m must be greater than zero.")

        if visual_temp <= 0:
            raise ValueError("visual_temp must be greater than zero.")

        if audio_temp <= 0:
            raise ValueError("audio_temp must be greater than zero.")

        if visual_weight < 0 or audio_weight < 0:
            raise ValueError(
                "visual_weight and audio_weight cannot be negative."
            )

        if visual_weight + audio_weight <= 0:
            raise ValueError(
                "At least one modality weight must be positive."
            )

        valid_reductions = {
            "max",
            "mean",
            "topk_mean",
            "softmax",
        }

        if audio_atom_reduction not in valid_reductions:
            raise ValueError(
                f"Unsupported audio_atom_reduction="
                f"'{audio_atom_reduction}'. "
                f"Choose from {sorted(valid_reductions)}."
            )

        if audio_atom_topk <= 0:
            raise ValueError(
                "audio_atom_topk must be greater than zero."
            )

        if audio_atom_temperature <= 0:
            raise ValueError(
                "audio_atom_temperature must be greater than zero."
            )

        T = video_feats.shape[0]

        if T == 0:
            raise ValueError("video_feats contains no frames.")

        if audio.shape[-1] == 0:
            raise ValueError("audio waveform contains no samples.")

        top_k = min(top_k, T)

        num_windows = T - top_k + 1
        top_m = min(top_m, num_windows)

        num_audio_samples = audio.shape[-1]

        # Normalize the fusion weights.
        total_modality_weight = visual_weight + audio_weight

        visual_weight = (
            visual_weight / total_modality_weight
        )

        audio_weight = (
            audio_weight / total_modality_weight
        )

        # =============================================================
        # Internal helper: aggregate atoms into class scores
        # =============================================================
        def aggregate_audio_atom_scores(
            audio_similarity,
            dictionary_labels,
            num_classes,
        ):
            """
            Args:
                audio_similarity:
                    Shape [B, K].

                dictionary_labels:
                    Class label for every dictionary atom.
                    Shape [K].

            Returns:
                Class-level audio scores [B, num_classes].
            """

            batch_size = audio_similarity.shape[0]

            audio_class_scores = torch.empty(
                batch_size,
                num_classes,
                device=audio_similarity.device,
                dtype=audio_similarity.dtype,
            )

            for class_idx in range(num_classes):

                class_mask = (
                    dictionary_labels == class_idx
                )

                if not class_mask.any():
                    raise ValueError(
                        f"No audio dictionary atom is assigned "
                        f"to class {class_idx}."
                    )

                class_atom_scores = audio_similarity[
                    :,
                    class_mask,
                ]  # [B, number_of_class_atoms]

                if audio_atom_reduction == "max":

                    class_score = (
                        class_atom_scores
                        .max(dim=-1)
                        .values
                    )

                elif audio_atom_reduction == "mean":

                    class_score = (
                        class_atom_scores
                        .mean(dim=-1)
                    )

                elif audio_atom_reduction == "topk_mean":

                    current_topk = min(
                        audio_atom_topk,
                        class_atom_scores.shape[-1],
                    )

                    top_atom_scores = (
                        class_atom_scores.topk(
                            k=current_topk,
                            dim=-1,
                            largest=True,
                        ).values
                    )

                    class_score = (
                        top_atom_scores.mean(dim=-1)
                    )

                elif audio_atom_reduction == "softmax":

                    atom_weights = F.softmax(
                        class_atom_scores
                        / audio_atom_temperature,
                        dim=-1,
                    )

                    class_score = (
                        atom_weights
                        * class_atom_scores
                    ).sum(dim=-1)

                audio_class_scores[
                    :,
                    class_idx,
                ] = class_score

            return audio_class_scores

        # The window search is discrete, so gradients through all
        # candidates are unnecessary.
        with torch.no_grad():

            # =========================================================
            # 2. Generate all consecutive visual windows
            # =========================================================
            # video_feats.unfold produces [W, D, top_k].
            # Permuting gives [W, top_k, D].
            visual_windows = (
                video_feats
                .unfold(
                    dimension=0,
                    size=top_k,
                    step=1,
                )
                .permute(0, 2, 1)
                .contiguous()
            )

            # =========================================================
            # 3. Process all visual candidates in one batch
            # =========================================================
            temporal_features = self.temporal(
                visual_windows.transpose(1, 2)
            )  # [W, temporal_dim, temporal_length]

            temporal_features = F.gelu(
                temporal_features
            )

            visual_embeds = temporal_features.mean(
                dim=-1
            )  # [W, D]

            visual_embeds = F.normalize(
                visual_embeds,
                dim=-1,
            )

            adapted_embeds_norm = F.normalize(
                adapted_embeds,
                dim=-1,
            )

            au_similarity = (
                visual_embeds
                @ adapted_embeds_norm.T
            )  # [W, num_AUs]

            visual_logits = self.temporal_classifier(
                au_similarity
            )  # [W, C]

            visual_probs = F.softmax(
                visual_logits / visual_temp,
                dim=-1,
            )  # [W, C]

            num_classes = visual_probs.shape[-1]

            if num_classes < 2:
                raise ValueError(
                    "The temporal classifier must output at least "
                    "two classes."
                )

            visual_entropies = -(
                visual_probs
                * torch.log(
                    visual_probs.clamp_min(1e-8)
                )
            ).sum(dim=-1)

            visual_entropies = (
                visual_entropies
                / math.log(num_classes)
            )  # [W]

            # =========================================================
            # 4. Select top-M visual candidates
            # =========================================================
            candidate_indices = torch.topk(
                visual_entropies,
                k=top_m,
                largest=False,
            ).indices  # [top_m]

            # =========================================================
            # 5. Prepare fixed audio dictionary
            # =========================================================
            audio_dictionary = self.audio_dictionary

            if audio_dictionary.ndim != 2:
                raise ValueError(
                    "self.audio_dictionary must have shape "
                    "[num_atoms, audio_feature_dim]."
                )

            dictionary_labels = (
                self.audio_cluster_labels
                .long()
            )

            if dictionary_labels.ndim != 1:
                raise ValueError(
                    "self.audio_dictionary_labels must have "
                    "shape [num_atoms]."
                )

            if (
                dictionary_labels.shape[0]
                != audio_dictionary.shape[0]
            ):
                raise ValueError(
                    "The number of dictionary labels does not match "
                    "the number of audio dictionary atoms."
                )

            # =========================================================
            # 6. Audio reranking for top-M visual candidates
            # =========================================================
            candidate_visual_probs = []
            candidate_audio_probs = []
            candidate_audio_class_scores = []
            candidate_fused_probs = []
            candidate_fused_entropies = []

            candidate_audio_segments = []
            candidate_audio_starts = []
            candidate_audio_ends = []

            for candidate_idx_tensor in candidate_indices:

                candidate_idx = int(
                    candidate_idx_tensor.item()
                )

                visual_start = candidate_idx
                visual_end = candidate_idx + top_k

                # -----------------------------------------------------
                # Map visual interval to synchronized waveform interval
                # -----------------------------------------------------
                audio_start = round(
                    visual_start
                    / T
                    * num_audio_samples
                )

                audio_end = round(
                    visual_end
                    / T
                    * num_audio_samples
                )

                audio_start = max(
                    0,
                    min(
                        audio_start,
                        num_audio_samples - 1,
                    ),
                )

                audio_end = max(
                    audio_start + 1,
                    min(
                        audio_end,
                        num_audio_samples,
                    ),
                )

                audio_window = audio[
                    :,
                    audio_start:audio_end,
                ]

                # -----------------------------------------------------
                # Extract OpenSMILE features for this local audio window
                # -----------------------------------------------------
                audio_embedding = (
                    self.extract_opensmile_batch(
                        audio_window,
                        sample_rate=sample_rate,
                    )
                )

                if audio_embedding.ndim == 1:
                    audio_embedding = (
                        audio_embedding.unsqueeze(0)
                    )

                audio_dictionary_current = (
                    audio_dictionary.to(
                        device=audio_embedding.device,
                        dtype=audio_embedding.dtype,
                    )
                )

                dictionary_labels_current = (
                    dictionary_labels.to(
                        audio_embedding.device
                    )
                )

                if (
                    audio_embedding.shape[-1]
                    != audio_dictionary_current.shape[-1]
                ):
                    raise ValueError(
                        "OpenSMILE embedding dimension does not "
                        "match the audio dictionary dimension: "
                        f"{audio_embedding.shape[-1]} versus "
                        f"{audio_dictionary_current.shape[-1]}."
                    )

                # Cosine-normalize query and dictionary.
                audio_embedding = F.normalize(
                    audio_embedding,
                    dim=-1,
                )

                audio_dictionary_norm = F.normalize(
                    audio_dictionary_current,
                    dim=-1,
                )

                # -----------------------------------------------------
                # Similarity to all dictionary atoms
                # -----------------------------------------------------
                audio_similarity = (
                    audio_embedding
                    @ audio_dictionary_norm.T
                )  # [1, K]

                # -----------------------------------------------------
                # Aggregate atoms into class-level scores
                # -----------------------------------------------------
                audio_class_scores = (
                    aggregate_audio_atom_scores(
                        audio_similarity=audio_similarity,
                        dictionary_labels=(
                            dictionary_labels_current
                        ),
                        num_classes=num_classes,
                    )
                )  # [1, C]

                audio_prob = F.softmax(
                    audio_class_scores / audio_temp,
                    dim=-1,
                )  # [1, C]

                # -----------------------------------------------------
                # Get corresponding visual probability
                # -----------------------------------------------------
                visual_prob = visual_probs[
                    candidate_idx:candidate_idx + 1
                ].to(
                    device=audio_prob.device,
                    dtype=audio_prob.dtype,
                )

                # -----------------------------------------------------
                # Fuse synchronized visual and audio probabilities
                # -----------------------------------------------------
                fused_prob = (
                    visual_weight * visual_prob
                    + audio_weight * audio_prob
                )

                fused_prob = (
                    fused_prob
                    / fused_prob.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-8)
                )

                fused_entropy = -(
                    fused_prob
                    * torch.log(
                        fused_prob.clamp_min(1e-8)
                    )
                ).sum(dim=-1)

                fused_entropy = (
                    fused_entropy
                    / math.log(num_classes)
                )

                # -----------------------------------------------------
                # Store candidate information
                # -----------------------------------------------------
                candidate_visual_probs.append(
                    visual_prob.squeeze(0)
                )

                candidate_audio_probs.append(
                    audio_prob.squeeze(0)
                )

                candidate_audio_class_scores.append(
                    audio_class_scores.squeeze(0)
                )

                candidate_fused_probs.append(
                    fused_prob.squeeze(0)
                )

                candidate_fused_entropies.append(
                    fused_entropy.squeeze(0)
                )

                candidate_audio_segments.append(
                    audio_window
                )

                candidate_audio_starts.append(
                    audio_start
                )

                candidate_audio_ends.append(
                    audio_end
                )

            # =========================================================
            # 7. Select lowest-entropy fused candidate
            # =========================================================
            candidate_visual_probs = torch.stack(
                candidate_visual_probs,
                dim=0,
            )  # [top_m, C]

            candidate_audio_probs = torch.stack(
                candidate_audio_probs,
                dim=0,
            )  # [top_m, C]

            candidate_audio_class_scores = torch.stack(
                candidate_audio_class_scores,
                dim=0,
            )  # [top_m, C]

            candidate_fused_probs = torch.stack(
                candidate_fused_probs,
                dim=0,
            )  # [top_m, C]

            candidate_fused_entropies = torch.stack(
                candidate_fused_entropies,
                dim=0,
            )  # [top_m]

            best_local_idx = int(
                candidate_fused_entropies
                .argmin()
                .item()
            )

            selected_start = int(
                candidate_indices[
                    best_local_idx
                ].item()
            )

            selected_end = selected_start + top_k

            selected_audio = (
                candidate_audio_segments[
                    best_local_idx
                ]
            )

            selected_audio_start = (
                candidate_audio_starts[
                    best_local_idx
                ]
            )

            selected_audio_end = (
                candidate_audio_ends[
                    best_local_idx
                ]
            )

        # =============================================================
        # 8. Return selected synchronized window and diagnostics
        # =============================================================
        selected_idx = list(
            range(
                selected_start,
                selected_end,
            )
        )

        return {
            "selected_idx": selected_idx,
            "selected_start": selected_start,
            "selected_end": selected_end,

            "selected_video_feats": video_feats[
                selected_start:selected_end
            ],

            "selected_audio": selected_audio,
            "selected_audio_start": selected_audio_start,
            "selected_audio_end": selected_audio_end,

            "selected_visual_prob": (
                candidate_visual_probs[
                    best_local_idx:best_local_idx + 1
                ]
            ),

            "selected_audio_prob": (
                candidate_audio_probs[
                    best_local_idx:best_local_idx + 1
                ]
            ),

            "selected_audio_class_scores": (
                candidate_audio_class_scores[
                    best_local_idx:best_local_idx + 1
                ]
            ),

            "selected_fused_prob": (
                candidate_fused_probs[
                    best_local_idx:best_local_idx + 1
                ]
            ),

            "selected_visual_entropy": (
                visual_entropies[selected_start]
            ),

            "selected_fused_entropy": (
                candidate_fused_entropies[
                    best_local_idx
                ]
            ),

            # Candidate diagnostics
            "candidate_indices": candidate_indices,
            "candidate_visual_probs": (
                candidate_visual_probs
            ),
            "candidate_audio_probs": (
                candidate_audio_probs
            ),
            "candidate_audio_class_scores": (
                candidate_audio_class_scores
            ),
            "candidate_fused_probs": (
                candidate_fused_probs
            ),
            "candidate_fused_entropies": (
                candidate_fused_entropies
            ),

            "visual_window_probs": visual_probs,
            "visual_entropies": visual_entropies,

            "audio_atom_reduction": (
                audio_atom_reduction
            ),
        }

    def temporal_clip_au_forward(self, video, audio, au_prompts, class_prompts, fus_type, mod_align, adapt_tar=False, key_frame_sel=False, train_whole_clip=False, key_frames=16, is_opensmile_dict=False):
    
        """
        Simple temporal CLIP forward:
        1. Extract CLIP embeddings per frame.
        2. Model temporal structure via transformer.
        3. Compute similarity with AU prompts (or class prompts).

        Args:
            video: [B, T, 3, H, W] tensor
            au_prompts: list of AU text prompts (len = num_aus)

        Returns:
            logits_per_video: [B, num_aus]
        """
        B, T, C, H, W = video.shape

        # 1️⃣ Encode AU text prompts
        if train_whole_clip:
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))
        else:
            with torch.no_grad():
                base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))  # [num_aus, D]
            if class_prompts is not None:
                if adapt_tar:
                    base_cls_text_embeds = self.get_text_features()
                else:
                    base_cls_text_embeds = self.clip.encode_text(tokenize(class_prompts).to(self.device))  # [num_aus, D]
            else:
                base_cls_text_embeds = None

        # base_text_embeds = F.normalize(base_text_embeds, dim=-1)
        # ---- AU Prompt Tuning
        if adapt_tar:
            base_text_embeds = self.get_au_features()
        # with torch.no_grad():
        text_embeds = self.text_adapter(base_text_embeds)  # trainable
        text_embeds_norm = F.normalize(text_embeds, dim=-1)

        # 3️⃣ Extract per-frame CLIP features
        x = video.reshape(B * T, C, H, W)
        if train_whole_clip:
            img_feats = self.image_encoder(x.type(self.dtype))
        else:
            with torch.no_grad():
                img_feats = self.image_encoder(x.type(self.dtype))  # [B*T, D]
        # img_feats = F.normalize(img_feats, dim=-1)
        img_feats = img_feats.reshape(B, T, -1)  # [B, T, D]

        # 2️⃣ For each video in batch****************
        selected_feats = []
        if adapt_tar and key_frame_sel:
            for b in range(B):
                frame_feats = img_feats[b]  # [T, 512]
                # key_idx = self.select_key_consec_frames(frame_feats, text_embeds_norm, top_k=key_frames) # Biovid=16 and Stress=6
                '''
                    Rebuttal Test: Select key frames using temporal module 
                    [INFO] Selecting Window using Temporal Module: window->AU-sim->EmoClassifier->Entropy: select frame window with lowest entropy										
                '''
                # key_idx = self.select_key_consec_frames_temporal(frame_feats, text_embeds_norm, top_k=key_frames) 
                # selected_feats.append(frame_feats[key_idx])

                # selection = self.select_key_consec_frames_multimodal(
                #     video_feats=frame_feats,       # [30, D]
                #     audio=audio,                   # [1, 464000]
                #     adapted_embeds=text_embeds_norm,
                #     top_k=key_frames,
                #     sample_rate=16000,
                #     visual_temp=1.0,
                #     audio_temp=1.0, #0.2
                #     visual_selection_weight=0.5,
                #     audio_selection_weight=0.5,
                # )

                selection = self.select_key_consec_frames_audio_guided(
                    video_feats=frame_feats,
                    audio=audio,
                    adapted_embeds=text_embeds_norm,
                    top_k=16,
                    sample_rate=16000,
                    visual_temp=1.0,
                    audio_temp=0.2,
                    audio_guidance_strength=0.25,
                    audio_atom_reduction="softmax"
                )

                # selection = self.select_key_consec_frames_multimodal_topm(
                #     video_feats=frame_feats,
                #     audio=audio,
                #     adapted_embeds=text_embeds_norm,
                #     top_k=key_frames,
                #     top_m=3,
                #     visual_temp=1.0,
                #     audio_temp=0.2,
                #     visual_weight=0.28,
                #     audio_weight=0.72,
                #     audio_atom_reduction="mean",
                # )
                key_idx = selection["selected_idx"]
                selected_feats.append(frame_feats[key_idx])

            # # 3️⃣ Stack for temporal model
            max_T = max(len(f) for f in selected_feats)
            selected_feats = torch.stack([
                F.pad(f, (0, 0, 0, max_T - f.size(0))) for f in selected_feats
            ])  # [B, max_T, 512]

        v = self.temporal(selected_feats.transpose(1,2) if key_frame_sel else img_feats.transpose(1,2))
        v = F.gelu(v)
        visual_embeds = v.mean(dim=-1)

        visual_embeds_norm = F.normalize(visual_embeds, dim=-1)

        '''
        NEW PART: Modalities Align + Fusion
        '''

        """
        Process Audio

        """
        # --- Option-1: Use only visual for window selection 
        # selected_audio = self.crop_synchronized_audio(
        #     audio=audio,
        #     total_frames=video.shape[1],  # 30
        #     window_start=key_idx[0],
        #     window_size=key_frames,
        # )

        # --- Option-2: Use visual+Audio for window selection 
        selected_audio = selection["selected_audio"]

        if is_opensmile_dict:
            audio_embeds_norm = self.extract_opensmile_batch(selected_audio, sample_rate=16000)
        else:
            # inputs = self.audio_fe(audio, sampling_rate=16000, return_tensors="pt")
            # Check the last dimension (temporal length)
            if audio.shape[-1] < 100:
                print(f"Audio shape {audio.shape} is too short!")
            base_audio_embeds = self.audio_model(audio)  # (B, 1024)
        # audio_embeds = self.audio_adapter(base_audio_embeds)  # (B, 512)
        # audio_embeds_norm = F.normalize(audio_embeds, p=2, dim=-1)


        if not is_opensmile_dict:
            # 🔹 Step B: Fusion (Concatenation)
            if fus_type == config.FUS_CONCAT:
                fused_embeds = torch.cat(
                    [audio_embeds_norm, visual_embeds_norm],
                    dim=-1
                )  # (B, 1024)

                fused_embeds = self.fusion_mlp(fused_embeds)  # (B, 512)

            # Step B (ii): Fusion (CrossAttention)
            elif fus_type == config.FUS_CROSSATEN:
                z_va, _ = self.crossAtten_fusion(visual_embeds_norm, audio_embeds_norm)
                z_av, _ = self.crossAtten_fusion(audio_embeds_norm, visual_embeds_norm)  # a attends to v

                fused_embeds = (z_va + z_av) / 2
            
            # Step B (iii): Fusion (GATED: give dominant modality more weight)
            elif fus_type == config.FUS_GATED:
                gate = torch.sigmoid(self.gated_fusion(torch.cat([visual_embeds_norm, audio_embeds_norm], dim=-1)))
                fused_embeds = gate * visual_embeds_norm + (1 - gate) * audio_embeds_norm

            elif fus_type == config.FUS_MOE:
                weights = torch.softmax(self.router(torch.cat([visual_embeds_norm.detach(), audio_embeds_norm.detach()], dim=-1)), dim=-1)
                fused_embeds = weights[:, 0:1] * visual_embeds_norm + weights[:, 1:2] * audio_embeds_norm

            # 4️⃣ Cosine similarity between temporal embedding & AUs
            # logit_scale = self.logit_scale.exp()
            # au_sim = logit_scale * (v @ adapted_embeds.T)   # [B, num_aus]
            temp = 5.00
            fused_embeds = F.normalize(fused_embeds, dim=-1)
            au_sim = temp * (fused_embeds @ text_embeds_norm.T)
            if adapt_tar:
                au_sim.retain_grad()

            # ----- *** ABlation -- include class-prompt with Au prompt
            if base_cls_text_embeds is not None:
                base_cls_text_embeds = F.normalize(base_cls_text_embeds, dim=-1)
                cls_sim = temp * (visual_embeds_norm @ base_cls_text_embeds.T)
                # class_proj = self.class_to_au(cls_sim)   # [B, 46]
                alpha=0.5
            logits = self.temporal_classifier(au_sim)   # [B, num_classes]

            if base_cls_text_embeds is not None:
                cls_temp = 0.3
                logits = ((1-cls_temp)*logits) + (cls_temp * cls_sim)
        else:
            # Temporary normalized view.
            # self.audio_dictionary itself remains unchanged.
            dictionary_for_similarity = F.normalize(
                self.audio_dictionary,
                p=2,
                dim=-1,
                eps=1e-8,
            )

            audio_similarity = (
                audio_embeds_norm @ dictionary_for_similarity.T
            )

            visual_similarity = (visual_embeds_norm @ text_embeds_norm.T)

            # visual_similarity_fused = F.layer_norm(
            #     visual_similarity,
            #     visual_similarity.shape[-1:],
            # )

            # audio_similarity_fused = F.layer_norm(
            #     audio_similarity,
            #     audio_similarity.shape[-1:],
            # )

            # joint_similarity = torch.cat(
            #     [
            #         visual_similarity_fused,
            #         audio_similarity,
            #     ],
            #     dim=-1,
            # )

            L_align=None
            # 🔹 Step A: Contrastive alignment (OmniBind Idea) -- but it cannot directly applied here
            if mod_align:
                visual_align = self.visual_align_proj(visual_embeds_norm)
                audio_align = self.audio_align_proj(audio_embeds_norm)

                visual_align = F.normalize(visual_align, p=2, dim=-1, eps=1e-8)
                audio_align = F.normalize(audio_align, p=2, dim=-1, eps=1e-8)

                contr_temp = 0.07   # hard-coded 
                # contr_temp = self.contr_logit_scale       # learnable 
                contr_logits = visual_align @ audio_align.T / contr_temp
                ss_labels = torch.arange(B).to(contr_logits.device)

                L_align = (
                    F.cross_entropy(contr_logits, ss_labels) +
                    F.cross_entropy(contr_logits.T, ss_labels)
                ) / 2
            else:
                L_align = None

            vis_logits = self.temporal_classifier(visual_similarity) 
            # audio_logits = self.au_audio_classifier(audio_similarity)

            audio_logits = self.audio_similarity_to_class_logits(
                audio_similarity,
                reduce="mean",
            )

            visual_temp=1.0
            audio_temp=0.5      # before 0.2 for pretraining
            visual_prob = F.softmax(
                vis_logits / visual_temp,
                dim=-1,
            )

            audio_prob = F.softmax(
                audio_logits / audio_temp,
                dim=-1,
            )

            if adapt_tar: 
                # ---- MM-CLIP-AUTT Personalized Fusion Preference ----
                
                # 1-option: Learnable subject_fusion_delta weight --- Not working/learning keep getting 0 
                # -Generic fusion preference remains frozen.
                # -Only subject_fusion_delta is updated.
                alpha_subject = torch.sigmoid(
                    self.audio_fusion_alpha.detach()
                    + self.subject_fusion_delta
                )
                logits = (
                    (1.0 - alpha_subject) * visual_prob
                    + alpha_subject * audio_prob
                )

                # logits = (
                #     visual_prob + audio_prob
                # )

                # 2-Option: Give higher weight to the dominate modality 
                # -subject_info = self.estimate_subject_modality_weights(visual_prob, audio_prob)

                # subject_info = self.certainty_aware_fusion(visual_prob=visual_prob, audio_prob=audio_prob, confidence_threshold=0.85,boost_strength=1.0)

                # visual_weight = subject_info["visual_weight"]
                # audio_weight = subject_info["audio_weight"]
                # s_logits = (
                #     visual_weight * visual_prob
                #     + audio_weight * audio_prob
                # )
                # print(visual_prob)
                # print(visual_weight)
                # print(audio_prob)
                # print(audio_weight)

                # 3-option: Conservative weighting

                # outputs = self.conservative_confidence_fusion(
                #     visual_prob=visual_prob,
                #     audio_prob=audio_prob,
                #     base_audio_weight=float(alpha_subject),
                #     min_audio_weight=0.30,
                #     max_audio_weight=0.90,
                #     high_confidence=0.65,
                #     audio_margin=0.10,
                #     visual_margin=0.05,
                # )

                # logits = outputs["fused_prob"]


            else:
                # This is for MM-CLI-AU Pre-training Step
                alpha = torch.sigmoid(self.audio_fusion_alpha)
                logits = (1.0 - alpha) * visual_prob + alpha * audio_prob

        return logits, audio_similarity, visual_embeds_norm, L_align
                
    def temporal_clip_au_testime(self, video, audio, au_prompts, class_prompts, fus_type, mod_align, key_frame_sel=False, key_frames=16, is_opensmile_dict=False):
        # 1 Get AU logits and Temporal Video Embeddings using 1DCNN for temporal
        logits_au, au_sim, visual_embeds_norm, L_align = self.temporal_clip_au_forward(video, audio, au_prompts, class_prompts, fus_type, mod_align, adapt_tar=True, key_frame_sel=key_frame_sel, key_frames=key_frames, is_opensmile_dict=is_opensmile_dict)

        return logits_au, au_sim, visual_embeds_norm, L_align

    # ===========================================================
    def forward(self, x, audio, au_prompts=None, class_prompts=None, mode="temporal", adapt_target=False, key_frame_sel=False, key_frames=16, train_whole_clip=False, fus_type=config.FUS_CONCAT, mod_align=False, is_opensmile_dict=False):
        """
        mode:
          - 'temporal': sequence modeling for video (train AU adapter + AU classifier + transformer)
          - 'clip': standard CLIP image-based path
        """
        if mode == "temporal":
            # assert x.ndim == 5, "Expected video input [B, T, 3, H, W]"
            # return self.au_pathway_logits(x, au_prompts)
            if adapt_target:
                return self.temporal_clip_au_testime(x, audio, au_prompts, class_prompts, fus_type, mod_align, key_frame_sel=key_frame_sel, key_frames=key_frames, is_opensmile_dict=is_opensmile_dict)
            logits, au_sim, _, L_align = self.temporal_clip_au_forward(x, audio, au_prompts, class_prompts, fus_type, mod_align, train_whole_clip=train_whole_clip, is_opensmile_dict=is_opensmile_dict)
            return logits, au_sim , _, L_align
            # return self.forward_temporal(x, au_prompts)

        elif mode == "clip":
            with torch.no_grad():
                img_features = self.image_encoder(x.type(self.dtype))
            text_features = self.get_text_features_from_prompts(au_prompts)
            img_features = F.normalize(img_features, dim=-1)
            logits = self.logit_scale.exp() * img_features @ text_features.t()
            return logits

     
    # AU pathway logits (text adapter + AU classifier)
    def au_pathway_logits(self, image, au_prompts):
        with torch.no_grad():
            img_features = self.image_encoder(image.type(self.dtype))
            base_text_embeds = self.clip.encode_text(tokenize(au_prompts).to(self.device))

        img_features = F.normalize(img_features, dim=-1)
        base_text_embeds = F.normalize(base_text_embeds, dim=-1)

        adapted_embeds = self.text_adapter(base_text_embeds)  # trainable
        adapted_embeds = F.normalize(adapted_embeds, dim=-1)

        sim = img_features @ adapted_embeds.T  # [B, num_aus]
        logits = self.au_classifier(sim)       # [B, num_classes]
        return logits

    # ===========================================================
    def inference_temporal(self, video, au_prompts):
        """
        Inference path — identical to training, except with no grad.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_temporal(video, au_prompts)
        return logits


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, num_aus=46, num_classes=2, au_prompts=None, learned_cls=False, is_video_clip=False, frame_stride=1, key_frames=16, save_audio_dict=None, opensmile_scaler_path=None, audio_cluster_labels_path=None):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    elif 'bah' in test_set:
        classnames = bahssub_classes
    else: # -- HERE change it for other datasets
        classnames = biosub_classes 

    if is_video_clip:
        model = ClipTestTimeVideoTuning(device, classnames, None, au_prompts, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_aus=num_aus, num_classes=num_classes,
                            text_hidden=512, clf_hidden=256, 
                            d_model=512, num_layers=4, nhead=8, dim_forward=2048, delta_stride=1, audio_dictionary_path=save_audio_dict, 
                            opensmile_scaler_path=opensmile_scaler_path, audio_cluster_labels_path=audio_cluster_labels_path)
    else:
        model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_aus=num_aus, num_classes=num_classes)

    return model

''' Use by EmoClip
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 4,
            dim_forward: int = 2048
'''
''' Use by EmoClip
            d_model: int = 192,
            nhead: int = 4,
            num_layers: int = 2,
            dim_forward: int = 512
'''