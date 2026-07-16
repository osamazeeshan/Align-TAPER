import argparse
import os

import time

from copy import deepcopy

from PIL import Image
import numpy as np
import csv
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import config
import torch.nn.functional as F
import torch.nn as nn
from thop import profile

import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

from audio.opensmile_audio_dic import *
from utils.visualize_audio_clusters import *


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, GradCAM, VClip
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.biovid_prompts import biosub_classes, raftrains_classes, raftests_classes
from data.stress_prompts import stresssub_classes
from data.bah_prompts import bahssub_classes
from data.ferv39_prompts import ferv39k_classes
from data.dfew_prompts import dfew_classes
from data.mafw_prompts import mafw_classes

from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, create_target_folders, MetricLogger, load_bah_src_subs, create_source_files, create_target_subject_files, save_best_metrics, calculate_aggregate_performance, update_subject_result_xlsx, extract_audio_from_folder
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from data.land_datasets import MediapipeFaceDataset
from utils.visualize import *

from utils.cometml import comet_init, set_comet_exp_name
from data.action_units_prompts import AU_PROMPTS, CLASS_PROMPTS, CLASS_PROMPTS_AMBV
from clip import load, tokenize
from datasets.base_dataset import BaseDataset
from utils.reproducibility import get_default_seed

from tqdm import tqdm
from collections import OrderedDict

experiment = comet_init(config.COMET_PROJECT_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TSNE_ALL_SUB_VIDEOS = []
TSNE_ALL_SUB_LABLES = []

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def entropy_loss(logits, class_weights=None):

    # weighted entropy to avoid collapse
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(class_weights * probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    ent = - (probs * log_probs).sum(dim=-1).mean()
    return ent

def batch_entropy(outputs):
    probs = torch.softmax(outputs, dim=1)
    avg_probs = probs.mean(dim=0)
    return -(avg_probs * avg_probs.log()).sum()

def test_time_tuning(model, inputs, optimizer, scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx), au_prompts=AU_PROMPTS, mode=("adapt" if args.adapt_tar_sub else "clip"))
            else:
                output = model(inputs, au_prompts=AU_PROMPTS, mode=("adapt" if args.adapt_tar_sub else "clip"))

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return

@torch.no_grad()
def predict_with_prompts(model, image_tensor, neutral_prompt, pain_prompt):
    """
    image_tensor: 1x3xHxW tensor (already on device)
    neutral_prompt, pain_prompt: strings (built for this image)
    returns: predicted_class (str), probs (tensor[2])
    """
    device = image_tensor.device

    # 1. Encode the two prompts
    prompt_list = [neutral_prompt, pain_prompt]
    text_features = model.get_text_features_from_prompts(prompt_list)  # [2,D]

    # 2. Encode the image
    img_feat = model.image_encoder(image_tensor.type(model.dtype))  # [1,D]
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # 3. Compute logits and probabilities
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * img_feat @ text_features.t()  # [1,2]
    probs = logits.softmax(dim=-1).squeeze(0)  # [2]

    # 4. Pick predicted class
    classes = ["neutral", "pain"]
    pred_idx = probs.argmax().item()
    pred_class = classes[pred_idx]

    return pred_class, probs


def build_class_prompts_from_template(template, classnames):
    # template like: a_person_with_an_expression_of_[CLS]
    # classnames are your dataset labels (e.g., 'neutral', 'pain', etc.)
    prompts = []
    for name in classnames:
        # make sure no spaces
        cls_token = str(name).replace(" ", "_")
        prompts.append(template.replace("[CLS]", cls_token))
    return prompts

# @torch.no_grad()
# def encode_text_prompts(text_encoder, clip_model, prompts: list, device):
#     # tokenize using CLIP's tokenizer
#     tokenized = torch.cat([tokenize(p) for p in prompts]).to(device)  # [N, L]
#     with torch.no_grad():
#         embeddings = clip_model.token_embedding(tokenized).type(text_encoder.dtype)  # [N, L, D]
#     # run through your TextEncoder
#     features = text_encoder(embeddings, tokenized)  # [N, D]
#     features = F.normalize(features, dim=-1)
#     return features

@torch.no_grad()
def encode_text_prompts_with_model(args, model, prompts: list, device):
    # tokenize with clip.tokenize
    tokenized = torch.cat([tokenize(p) for p in prompts]).to(device)
    # get token embeddings from CLIP’s embedding table
    clip_model = model.prompt_learner  # contains the clip token_embedding
    token_embedding = model.prompt_learner.token_prefix  # but prefix only has 1 token; better to call directly
    # we can get CLIP directly from model.text_encoder by reading its transformer etc.

    # safer: reuse clip from PromptLearner’s reset_classnames
    import clip
    clip_base, _, _ = clip.load(args.arch, device=device)  # small one-time load to access token_embedding

    with torch.no_grad():
        embedding = clip_base.token_embedding(tokenized).type(model.text_encoder.dtype)
    # pass through your TextEncoder
    features = model.text_encoder(embedding, tokenized)
    features = F.normalize(features, dim=-1)
    return features

@torch.no_grad()
def get_image_features(model, images):
    img_feats = model.image_encoder(images.type(model.dtype))  # [B,D]
    return F.normalize(img_feats, dim=-1)

def save_matrix_csv(path, header_cols, row_names, matrix_torch):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AU\\Class"] + header_cols)
        mat = matrix_torch.detach().cpu().float().numpy()
        for r, name in enumerate(row_names):
            writer.writerow([name] + [f"{v:.6f}" for v in mat[r]])

def load_txt_adapter_classifier(save_path, model, args, device):
    checkpoint = torch.load(save_path, map_location=device)
    if args.train_whole_clip_model:
        model.load_state_dict(checkpoint)
        print("[INFO] CLIP Model Loaded: ", save_path)
    else:
        if "text_adapter" in checkpoint:
            model.text_adapter.load_state_dict(checkpoint['text_adapter'])    
        if "au_classifier" in checkpoint:
            model.au_classifier.load_state_dict(checkpoint["au_classifier"])
        if "temporal_classifier" in checkpoint:
            model.temporal_classifier.load_state_dict(checkpoint["temporal_classifier"])
        if "temporal" in checkpoint:
            model.temporal.load_state_dict(checkpoint["temporal"])
        if "temporal_proj" in checkpoint:
            model.temporal_proj.load_state_dict(checkpoint["temporal_proj"])
        if "audio_fusion_alpha" in checkpoint:
            model.audio_fusion_alpha.copy_(checkpoint["audio_fusion_alpha"])

    model = model.cuda(args.gpu)
    return model


def load_clip_encoders(checkpoint_path, model, device='cpu'):
    """
    Loads visual and text encoder weights from a CLIP checkpoint 
    into a composite model (like ClipTestTimeVideoTuning).

    Automatically matches 'visual.*' -> image_encoder
                         'transformer.*' -> text_encoder
    Only matching keys are loaded safely.
    """
    print(f"\n🔹 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # If checkpoint is wrapped in dict
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]

    image_keys = [k for k in checkpoint.keys() if k.startswith("visual.")]
    text_keys  = [k for k in checkpoint.keys() if k.startswith("transformer.")]
    print(f"Found {len(image_keys)} visual keys and {len(text_keys)} text keys.")

    # === Visual encoder ===
    if image_keys:
        visual_state = {k.replace("visual.", ""): v for k, v in checkpoint.items() if k.startswith("visual.")}
        model_dict = model.image_encoder.state_dict()
        matched_visual = {k: v for k, v in visual_state.items() if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(matched_visual)
        model.image_encoder.load_state_dict(model_dict, strict=False)

        print(f"✅ Loaded {len(matched_visual)} visual keys into image_encoder "
              f"(skipped {len(visual_state) - len(matched_visual)})")

    # === Text encoder ===
    if text_keys:
        text_state = {k.replace("transformer.", ""): v for k, v in checkpoint.items() if k.startswith("transformer.")}
        model_dict = model.text_encoder.transformer.state_dict()
        matched_text = {k: v for k, v in text_state.items() if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(matched_text)
        model.text_encoder.transformer.load_state_dict(model_dict, strict=False)

        print(f"✅ Loaded {len(matched_text)} text keys into text_encoder "
              f"(skipped {len(text_state) - len(matched_text)})")

    print("🚀 Encoder loading complete!\n")
    return model

def load_partial_checkpoint(model, ckpt_path, prefix="module."):
    """
    Loads only matching layers from a checkpoint into the model.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)  # handle wrapped or raw dict

    # Remove prefix like 'module.' if present
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            nk = k[len(prefix):]
        else:
            nk = k
        new_state[nk] = v

    # Filter by matching keys
    model_dict = model.state_dict()
    filtered = {k: v for k, v in new_state.items() if k in model_dict and v.shape == model_dict[k].shape}

    print(f"✅ Loading {len(filtered)}/{len(new_state)} compatible weights.")
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return model

def rebuild_optimizer(model, lr):
    # trainable_params = (
    #     list(model.text_adapter.parameters()) +
    #     list(model.temporal.parameters()) +
    #     list(model.temporal_classifier.parameters()) 
    # )
    trainable_params = model.au_prompt_learner.parameters()
    # trainable_params = (
    #     list(model.au_prompt_learner.parameters()) +
    #     list(model.temporal_proj.parameters())
    # )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    return optimizer

def main(args):
    # args = parser.parse_args()
    set_random_seed(get_default_seed())
    # visualize_metrics_from_csv("metrics/comparison_tran.csv")

    # extract_audio_from_folder('/home/ens/AS08960/datasets/StressID/Videos', '/home/ens/AS08960/datasets/StressID/audio_all')

    args.target_sub_set = 10
    # print(calculate_aggregate_performance(filename="best_metrics_"+str(args.target_sub_set)+".csv"))
    # print(calculate_aggregate_performance(filename="clip_au_metrics_"+str(args.target_sub_set)+".csv"))

    # plot_cluster_summary_table()
    # plot_label_acoustic_summary()

    if args.current_ds is config.BIOVID:
        '''
        BioVid Random Source subjects
        '''
        if args.target_sub_set ==  10:
            source_list_name = ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42',
            '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
            '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', 
            '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
            '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
            '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', 
            '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
            '091914_m_46', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']

            # --- Random target subjects
            target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28','100909_w_65','081609_w_40']

        elif args.target_sub_set == 20:
            # --- BioVid target subjects 67
            source_list_name = ['081714_m_36','112610_w_60','071709_w_23','110810_m_62','080209_w_26','110614_m_42','101814_m_58','071313_m_41','102514_w_40','101114_w_37',
            '100509_w_43','082315_w_60','120614_w_61','101514_w_36','092813_w_24','102309_m_61','081617_m_27','080609_w_27','111313_m_64','071614_m_20','101309_m_48','071911_w_24','102316_w_50','100417_m_44','083013_w_47','083009_w_42','080714_m_23','101809_m_59','082909_m_47','101209_w_61',
            '092014_m_56','072414_m_23','101015_w_43','112909_w_20','111609_m_65','100117_w_36','111409_w_63','080709_m_24','072714_m_23','112914_w_51',
            '120514_w_56','083109_m_60','110909_m_29','091814_m_37','071814_w_23','092509_w_51','112809_w_23','100214_m_50','102214_w_36','082714_m_22','082109_m_53','092808_m_51','080309_m_29','102008_w_22','111914_w_63','082809_m_26','072514_m_27','082814_w_46','072609_w_23','101216_m_40','091914_m_46','100914_m_39','112209_m_51','092514_m_50','092009_m_54',
            '082414_m_64','080614_m_24']

            # --- Random target subjects - 20
            target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28',
            '100909_w_65','081609_w_40', '082208_w_45', '101908_m_61', '082014_w_24', '101916_m_40', '112016_m_25', '100514_w_51', '112310_m_20', '092714_m_64',
            '102414_w_58', '083114_w_55']

        elif args.target_sub_set == 30:
             # --- BioVid target subjects 57
            source_list_name = ['081714_m_36','112610_w_60','071709_w_23','110810_m_62','080209_w_26',
            '110614_m_42','101814_m_58','071313_m_41','102514_w_40','101114_w_37',
            '100509_w_43','082315_w_60','120614_w_61','101514_w_36','092813_w_24',
            '102309_m_61','081617_m_27','080609_w_27','111313_m_64','071614_m_20',
            '101809_m_59','082909_m_47','092014_m_56','101015_w_43','112909_w_20',
            '100117_w_36','111409_w_63','072714_m_23','112914_w_51','120514_w_56',
            '083109_m_60','110909_m_29','091814_m_37','071814_w_23','092509_w_51',
            '112809_w_23','100214_m_50','102214_w_36','082109_m_53','092808_m_51',
            '080309_m_29','102008_w_22','111914_w_63','082809_m_26','072514_m_27',
            '082814_w_46','072609_w_23','101216_m_40','091914_m_46','100914_m_39',
            '112209_m_51','092514_m_50','092009_m_54','082414_m_64','080614_m_24']

            # --- Random target subjects - 30
            target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28',
            '100909_w_65','081609_w_40', '082208_w_45','101908_m_61','082014_w_24','101916_m_40', '112016_m_25','100514_w_51','112310_m_20','092714_m_64',
            '102414_w_58','083114_w_55','101309_m_48','071911_w_24','102316_w_50','100417_m_44', '080714_m_23','101209_w_61','072414_m_23','111609_m_65',
            '080709_m_24','082714_m_22']
        
        elif args.target_sub_set == 40:
             # --- BioVid target subjects 47
            source_list_name = ['081714_m_36','112610_w_60','071709_w_23','110810_m_62','080209_w_26',
            '110614_m_42','101814_m_58','071313_m_41','102514_w_40','101114_w_37', '100509_w_43','082315_w_60','120614_w_61','101514_w_36','092813_w_24',
            '102309_m_61','081617_m_27','080609_w_27','111313_m_64','071614_m_20','101809_m_59','082909_m_47','092014_m_56','101015_w_43','112909_w_20', '100117_w_36','111409_w_63','072714_m_23','112914_w_51','120514_w_56',
            '083109_m_60','110909_m_29','091814_m_37','071814_w_23','072514_m_27','082814_w_46','072609_w_23','101216_m_40','091914_m_46','100914_m_39','112209_m_51','092514_m_50',
            '092009_m_54','082414_m_64','080614_m_24']

            # --- Random target subjects - 40
            target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28',
            '100909_w_65','081609_w_40', '082208_w_45','101908_m_61','082014_w_24','101916_m_40','112016_m_25','100514_w_51','112310_m_20','092714_m_64',
            '102414_w_58','083114_w_55','101309_m_48','071911_w_24','102316_w_50','100417_m_44','080714_m_23','101209_w_61','072414_m_23','111609_m_65',
            '080709_m_24','082714_m_22','092509_w_51','112809_w_23','100214_m_50','102214_w_36','082109_m_53','092808_m_51','080309_m_29','102008_w_22',
            '111914_w_63','082809_m_26']

        elif args.target_sub_set == 50:
             # --- BioVid target subjects 37
            source_list_name = ['081714_m_36','112610_w_60','071709_w_23','110810_m_62','080209_w_26',
            '110614_m_42','101814_m_58','071313_m_41','102514_w_40','101114_w_37',
            '100509_w_43','082315_w_60','120614_w_61','101514_w_36','092813_w_24',
            '102309_m_61','081617_m_27','080609_w_27','111313_m_64','071614_m_20',
            '101809_m_59','082909_m_47','092014_m_56','101015_w_43','112909_w_20',
            '100117_w_36','111409_w_63','072714_m_23','112914_w_51',
            '120514_w_56','083109_m_60','110909_m_29','091814_m_37',
            '071814_w_23','080614_m_24']

             # --- Random target subjects - 50
            target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28',
            '100909_w_65','081609_w_40', '082208_w_45','101908_m_61','082014_w_24','101916_m_40','112016_m_25','100514_w_51','112310_m_20','092714_m_64',
            '102414_w_58','083114_w_55','101309_m_48','071911_w_24','102316_w_50','100417_m_44','080714_m_23','101209_w_61','072414_m_23','111609_m_65',
            '080709_m_24','082714_m_22','092509_w_51','112809_w_23','100214_m_50','102214_w_36','082109_m_53','092808_m_51','080309_m_29','102008_w_22',
            '111914_w_63','082809_m_26','072514_m_27','082814_w_46','072609_w_23','101216_m_40','091914_m_46','100914_m_39','112209_m_51','092514_m_50',
            '092009_m_54','082414_m_64']
                        
       
    elif args.current_ds is config.STRESS:
        '''
        StressID Random Source subjects
        '''
        source_list_name = ['9j3o','5f7t','9t6n','71i5','v8mh','bfl5','m8g5','45lx','tmvd','4woj',  # 10
                            'b2l8','j9h8','d4n6','9txq','g9j5','w2t5','2z7d','8g4y','6g6y','j1u8',  # 20
                            'c3m7','h7j3','chdf','qw5t','t6v9','a1k9','6k5f','i9t9','cxj0','r5s8',  # 30
                            '2ea4','2hpu','y8c3','kkf5','h8r2','iqyg','y9z6','f6q3','e5p4','k67g',  # 40
                            '8i4i', 'k2v7','4e8r','wssm']
        # --- StressID target subjects
        target_subject_list = ["kycf","uymz","h8s1","ctzy","p9i3","7h5u","g7r2","b9w0","r3zm","x1q3"]
    elif args.current_ds is config.BAH:
        '''
        BAH_DB Random Source subjects
        '''
        source_list_name = load_bah_src_subs(config.BAH_PATH_TRAIN)
        target_subject_list = ["82711", "82687", "82585", "82592", "82598", "82632", "82681", "82683", "82708", "82714"]
    # selection_ps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    # for sel_p in selection_ps:
    # args.selection_p = sel_p
    print("===== Selected Conf Threshold: ", args.selection_p)

    file_name = 'logs/'+str(args.ctx_init)+'-au_topP='+str(args.au_topP)+'-stp='+str(args.tta_steps)+'-n_Ctx='+str(args.n_ctx)+'-landmarks='+str(args.use_landmarks)+'-sel_t='+str(args.selection_p)+'-num_land='+str(args.num_landmarks)+'.txt'
    # with open(file_name, "a") as f:  
    #     f.write("=> Model created: visual backbone {}".format(args.arch))

    # This codebase has only been tested under the single GPU setting
    if args.current_ds is config.BIOVID:
        # if args.load_t_adpt_cl_mod:
        #     args.srcs_file_name= 'lab_srcs77_biovid_ep10_bs'+args.batch_size+'_sql'+str(16)+'_str'+str(args.frame_stride)+'_vid'
        # args.srcs_file_name='lab_srcs77_biovid_ep10_bs8_sql16_str2_vid'
        # args.srcs_file_name='lab_srcs77_biovid_ep10_tar'+str(args.target_sub_set)+'_bs8_sql16_str2_vid'  # -- For Robustness using more Target set (20, 30, 40, ..)
        args.pain_db_root_path = config.BIOVID_PATH
        
        # args.test_sets = 'biosub0/biosub3/biosub6/biosub8'
        # create_source_files()
        # create_target_subject_files()
        if args.target_sub_set == 10:
            args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9'
            args.srcs_label_file_name= 'lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
            # args.srcs_label_file_name= 'source_57'
        elif args.target_sub_set == 20:
            args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9/biosub10/biosub11/biosub12/biosub13/biosub14/biosub15/biosub16/biosub17/biosub18/biosub19'
            args.srcs_label_file_name= 'source_67'
        elif args.target_sub_set == 30:
            args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9/biosub10/biosub11/biosub12/biosub13/biosub14/biosub15/biosub16/biosub17/biosub18/biosub19/biosub20/biosub21/biosub22/biosub23/biosub24/biosub25/biosub26/biosub27/biosub28/biosub29'
            args.srcs_label_file_name= 'source_57'
        elif args.target_sub_set == 40:
            args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9/biosub10/biosub11/biosub12/biosub13/biosub14/biosub15/biosub16/biosub17/biosub18/biosub19/biosub20/biosub21/biosub22/biosub23/biosub24/biosub25/biosub26/biosub27/biosub28/biosub29/biosub30/biosub31/biosub32/biosub33/biosub34/biosub35/biosub36/biosub37/biosub38/biosub39'
            # args.test_sets = 'biosub26/biosub27/biosub28/biosub29/biosub30/biosub31/biosub32/biosub33/biosub34/biosub35/biosub36/biosub37/biosub38/biosub39'
            args.srcs_label_file_name= 'source_47'
        elif args.target_sub_set == 50:
            args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9/biosub10/biosub11/biosub12/biosub13/biosub14/biosub15/biosub16/biosub17/biosub18/biosub19/biosub20/biosub21/biosub22/biosub23/biosub24/biosub25/biosub26/biosub27/biosub28/biosub29/biosub30/biosub31/biosub32/biosub33/biosub34/biosub35/biosub36/biosub37/biosub38/biosub39/biosub40/biosub41/biosub42/biosub43/biosub44/biosub45/biosub46/biosub47/biosub48/biosub49'
            # args.test_sets = 'biosub26/biosub27/biosub28/biosub29/biosub30/biosub31/biosub32/biosub33/biosub34/biosub35/biosub36/biosub37/biosub38/biosub39/biosub40/biosub41/biosub42/biosub43/biosub44/biosub45/biosub46/biosub47/biosub48/biosub49'
            args.srcs_label_file_name= 'source_37'

        args.srcs_label_val_file_name=None
        
    elif args.current_ds is config.STRESS:
        # if args.load_t_adpt_cl_mod:
        # args.srcs_file_name = f'lab_srcs44_stress_ep20_bs{args.batch_size}_sql{args.seq_len}_str{args.frame_stride}_fus{args.fus_type}_modAlign{args.is_mod_align}' 

        # args.srcs_file_name = f'lab_srcs44_stress_ep{args.t_adap_epoch}_bs8_sql{args.seq_len}_str{args.frame_stride}_fus{args.fus_type}_modAlign{args.is_mod_align}_newgoff' 
        # args.srcs_file_name='lab_srcs44_stress_ep20_bs8_sql16_str1_vid_mm_audio_dic'
        # args.srcs_file_name='lab_srcs44_stress_ep20_bs8_sql16_str1_vid_mm_audio_dic_mcs_3_ms_3_audio_fusion_alpha=True'
        args.srcs_file_name='lab_srcs44_stress_ep20_bs8_sql16_str1_vid_mm_audio_dic_og'
        args.pain_db_root_path = config.STRESS_PATH
        args.test_sets = 'stresssub0/stresssub1/stresssub2/stresssub3/stresssub4/stresssub5/stresssub6/stresssub7/stresssub8/stresssub9'        
        args.srcs_label_file_name= 'stress_source_sub_labels'
        args.srcs_label_val_file_name= None
    elif args.current_ds is config.BAH:
        # if args.load_t_adpt_cl_mod:
        #     args.srcs_file_name= 'lab_srcs44_stress_ep20_bs8_sql'+str(16)+'_str'+str(args.frame_stride)+'_vid'
        args.srcs_file_name='bah_src_ep10_bs8_sql6_str1_vid_mm_audio_dic_og'
        args.pain_db_root_path = config.BAH_DATASET_FRAMES_PATH
        args.test_sets = 'bahssub0/bahssub1/bahssub2/bahssub3/bahssub4/bahssub5/bahssub6/bahssub7/bahssub8/bahssub9'
        # args.test_sets = 'bahssub0'
        args.srcs_label_file_name= 'bah_source_sub_train_labels'
        args.srcs_label_val_file_name= 'bah_source_sub_val_labels'
    elif args.current_ds is config.FERV39k:
        args.pain_db_root_path = config.FERV39k_DATASET_FRAMES_PATH
        args.srcs_file_name='ferv39k_ep10_bs8_sql16_str2_vid'
        args.test_sets = 'ferv39kt'
        args.srcs_label_file_name= 'ferv39_train'
        args.srcs_label_val_file_name= None
        source_list_name = ['ferv39_train']
        target_subject_list = ['ferv39kt']
    elif args.current_ds is config.DFEW:
        args.pain_db_root_path = config.DFEW_DATASET_FRAMES_PATH
        args.srcs_file_name='dfew_ep10_bs8_sql16_str2_vid_set_5'
        args.test_sets = 'dfewt'
        args.srcs_label_file_name= 'dfew_train_set_1'
        args.srcs_label_val_file_name= None
        source_list_name = ['dfew_train_set_1']
        target_subject_list = ['dfewt']
    elif args.current_ds is config.MAFW:
        args.pain_db_root_path = config.MAFW_DATASET_FRAMES_PATH
        args.srcs_file_name='mafw_ep20_bs8_sql16_str2_vid_set_5'
        args.test_sets = 'mafwt'
        args.srcs_label_file_name= 'mafw_train_set_5'
        args.srcs_label_val_file_name= None
        source_list_name = ['mafw_train_set_5']
        target_subject_list = ['mafwt']
    assert args.gpu is not None
    with experiment.train():
        experiment.log_parameter("arch", args.arch)
        experiment.log_parameter("Load train text adpt and classifier model", args.load_t_adpt_cl_mod)
        experiment.log_parameter("Train text adpt and classifier model", args.train_t_adpt_cl)
        experiment.log_parameter("Video seq length", args.seq_len)
        experiment.log_parameter("Video frame stride", args.frame_stride)
        experiment.log_parameter("Loss", 'CrossEntropyLoss')
        experiment.log_parameter("srcs_file_name", args.srcs_file_name)
        experiment.log_parameter("current_ds", args.current_ds)
        experiment.log_parameter("batch size", args.batch_size)
        experiment.log_parameter("source epochs", args.t_adap_epoch)
        experiment.log_parameter("seq length", args.seq_len)
        experiment.log_parameter("frame_stride", args.frame_stride)
        experiment.log_parameter("key_frame_sel", args.key_frame_sel)
        experiment.log_parameter("Key frames", args.key_frames)

        main_worker(args.gpu, args, file_name, source_list_name, target_subject_list)


def main_worker(gpu, args, file_name, source_list_name, target_subject_list):
    args.gpu = gpu
    set_random_seed(get_default_seed())
    print("Use GPU: {} for training".format(args.gpu))
    print("Train text adpt and classifier model: ", args.train_t_adpt_cl)
    print("Load train text adpt and classifier model: ", args.load_t_adpt_cl_mod)
    
    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    elif args.current_ds is config.RAF_DB:
        classnames = raftrains_classes
    elif args.current_ds is config.STRESS:
        classnames = stresssub_classes 
    elif args.current_ds is config.BAH:
        classnames = bahssub_classes
    elif args.current_ds is config.FERV39k:
        classnames = ferv39k_classes
    elif args.current_ds is config.DFEW:
        classnames = dfew_classes
    elif args.current_ds is config.MAFW:
        classnames = mafw_classes
    else:
        classnames = biosub_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())

    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, 
                        num_aus=len(AU_PROMPTS), num_classes=len(classnames), au_prompts=AU_PROMPTS, 
                        is_video_clip=args.is_video_clip, frame_stride=args.frame_stride, save_audio_dict=(
                        args.audio_dict_dir + "/opensmile_feature_cluster_dictionary.npy"), 
                        opensmile_scaler_path=(args.audio_dict_dir + "/opensmile_scaler.pkl"), 
                        audio_cluster_labels_path=(args.audio_dict_dir + "/cluster_labels.npy"))
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
        if args.train_t_adpt_cl:
            # train the text adapter if present
            # if "text_adapter" in name:
            #     param.requires_grad_(True)

            # # train the AU classifier if present
            # if "au_classifier" in name:
            #     param.requires_grad_(True)

            # if any(k in name for k in ["text_adapter", "temporal_classifier", "temporal"]):
            if any(k in name for k in ["text_adapter", "temporal", "temporal_classifier", "audio_fusion_alpha", "visual_align_proj", "audio_align_proj"]):
                param.requires_grad_(True)
            # if any(k in name for k in ["au_classifier", "text_adapter", "temporal_classifier"]):
            #     param.requires_grad_(True)

        if args.adapt_tar_sub:
            # Freeze everything by default
            # for name, param in model.named_parameters():
            if any(k in name for k in ["au_prompt_learner", "subject_fusion_delta"]):
                param.requires_grad_(True)
            # if "temporal_classifier" in name:
            #     param.requires_grad_(True)
            # elif args.au_prompt_tune and "prompt_learner" in name:
            #     param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        if args.train_whole_clip_model:
            param.requires_grad_(True)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    print("=> Using TPT Augmentation {}".format(args.tpt))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        if args.train_t_adpt_cl:
            if not args.is_video_clip:
                trainable_params = (
                    list(model.text_adapter.parameters()) +
                    list(model.au_classifier.parameters())
                )
            else:
                # 🔹 Train AU adapter + AU classifier + temporal transformer for video FER
                # trainable_params = (
                #     list(model.text_adapter.parameters()) +
                #     list(model.temporal_classifier.parameters()) +
                #     list(model.temporal.parameters()) +
                #     list(model.temporal_proj.parameters())
                # )
                trainable_params = (
                    list(model.text_adapter.parameters()) +
                    list(model.temporal.parameters()) +
                    list(model.temporal_classifier.parameters()) +
                    [model.audio_fusion_alpha] +
                    list(model.visual_align_proj.parameters()) +
                    list(model.audio_align_proj.parameters())
                )
                print("[INFO] Training AU adapter, AU classifier, audio_adapter fusion_mlp, and temporal transformer (video mode).")
        elif args.adapt_tar_sub:
            if args.au_prompt_tune:
                trainable_params = (list(model.au_prompt_learner.parameters()) + [model.subject_fusion_delta])
                
                # trainable_params = (
                #     list(model.au_prompt_learner.parameters()) +
                #     list(model.temporal_proj.parameters())
                # )
                print("[INFO] Training Target subject-specific AU prompt tuning (personalization mode).")
            else:
                # trainable_params = model.temporal_classifier.parameters()
                trainable_params = (
                    list(model.text_adapter.parameters()) +
                    list(model.temporal_classifier.parameters()) 
                )
                print("[INFO] Training Target with AU classifier (video mode).")
                # print("[INFO] Training Target subject-specific adapter only (personalization mode).")
        elif args.train_whole_clip_model:
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            print("[INFO] Training a While CLIP Model.")
        else:
            # 🔹 Default: training CLIP prompt learner for image-based FER
            trainable_params = model.prompt_learner.parameters()
            print("[INFO] Training prompt learner only (image mode).")

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    '''
        Here Training AU Adapter and Classifier using Source Data
    '''
    if (args.train_t_adpt_cl or args.train_whole_clip_model) and args.current_ds is not config.RAF_DB:
        print(f"=== Source file name: {args.srcs_file_name}")
        # (dataset_path, label_path_train, label_path_val, batch_size, resolution, phase)
        srcs_loader, srcs_val_loader, srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, 
                                args.srcs_label_file_name+('.csv' if args.current_ds is config.BAH else '.txt'), 
                                (args.srcs_label_val_file_name+'.csv' if args.current_ds is config.BAH else None), 
                                args.batch_size, BICUBIC, args.resolution, phase='src', 
                                seq_len=args.seq_len, frame_stride=args.frame_stride)
        # source_model_path = os.path.join(config.WEIGHTS_FOLDER, args.current_ds, args.srcs_file_name+'.pth')
    # elif args.current_ds is config.RAF_DB:
    source_model_path = os.path.join(config.WEIGHTS_FOLDER, args.current_ds, args.srcs_file_name+'.pth')

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}

    for set_id in datasets:
        # comet create experiment name
        tar_subject_id = int(set_id[-1])
        set_comet_exp_name(experiment, len(source_list_name), True, len(source_list_name), str())
        if args.current_ds is config.FERV39k or args.current_ds is config.DFEW or args.current_ds is config.MAFW:
            tar_sub_list = target_subject_list[0]
        else:
            tar_sub_list = target_subject_list[int(set_id[-1])]
        if not args.train_t_adpt_cl:
            target_file_path, target_weight_path, timestamp = create_target_folders(config.CURRENT_DIR, config.WEIGHTS_FOLDER, 
                tar_sub_list, args.top_timestamp if args.target_evaluation_only else None)

        if (args.tpt) and not args.use_landmarks:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1)
            batchsize = 1
            # batchsize = args.batch_size
        elif args.use_landmarks:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ])
            batchsize = args.batch_size
        else:
            # -- this is added for ViT-B/32; it required to normalize the data   
            # data_transform = transforms.Compose([
            #     transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.CenterCrop(args.resolution),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=0, 
                                            augmix=len(set_id)>1)
            if args.adapt_per_video:
                batchsize = 1
            elif args.load_t_adpt_cl_mod and args.eval_au_tar_sb:
                batchsize = args.tar_batch_size
            else:
                batchsize = args.batch_size
        
        with open(file_name, "a") as f:  
            f.write("=> Model created: visual backbone {}".format(args.arch))
            f.write("\nPrompt: {}".format(args.ctx_init))
            f.write("\nSubject: {}".format(set_id))

        print("evaluating: {}".format(set_id))
        if 'bio' or 'stress' in set_id:
            # sub_id = set_id[-1]
            # set_id = set_id[:-1]
            set_id, sub_id = re.search(r'^(.*?)(\d*)$', set_id).groups()

        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = biosub_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        if args.is_video_clip:
            ''' Load Target data using file for Videos '''
            if args.current_ds is config.FERV39k:
                tar_file_name = config.FERV39k_PATH_TEST
                # tar_file_name = config.DFEW_PATH_TEST
                root_path = config.FERV39k_DATASET_FRAMES_PATH
            elif args.current_ds is config.DFEW:
                tar_file_name = config.DFEW_PATH_TEST
                root_path = config.DFEW_DATASET_FRAMES_PATH
                # tar_file_name = config.FERV39k_PATH_TEST
            elif args.current_ds is config.MAFW:
                # tar_file_name = config.MAFW_PATH_TEST
                root_path = config.FERV39k_DATASET_FRAMES_PATH
                tar_file_name = config.FERV39k_PATH_TEST
            else:
                tar_sub_id = target_subject_list[int(sub_id)]
                tar_file_name = os.path.join(config.WEIGHTS_FOLDER, str(int(sub_id)+1)+'-'+tar_sub_id, 'files', 
                                    tar_sub_id+('.csv' if args.current_ds is config.BAH else '.txt'))
                root_path = args.pain_db_root_path

            val_loader, _, _ = BaseDataset.load_pain_dataset(root_path, tar_file_name, 
                            None, batchsize, BICUBIC, args.resolution, phase='tar', seq_len=args.seq_len, frame_stride=args.frame_stride)
        else:
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, sub_id=sub_id if args.current_ds is config.BIOVID else '0', use_landmarks=args.use_landmarks, max_landmarks=args.num_landmarks)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=args.workers, pin_memory=True)     
        
        ''' 
            visualize dataset 
        '''
        if args.use_landmarks and args.visualize_img:
            visualizer = LandmarkVisualizer(val_dataset)
            # Visualize landmark focus for image 0
            visualizer.visualize_landmark_focus(image_idx=1000, save_path="visual/landmark_focus.png")
            
            # Show actual landmark crops
            visualizer.visualize_landmark_crops(image_idx=1000, max_crops_to_show=args.num_landmarks, save_path="visual/landmark_crops.png")
            
            # Compare different landmark selections
            visualizer.compare_landmark_selections(image_idx=1000, landmark_counts=[5, 15, 30, 50], save_path="visual/landmark_comparison.png")
        

        if args.create_audio_dictionary:
            if not args.infer_audio_dictionary:
                if args.current_ds is config.RAF_DB:
                    raise ValueError("create_audio_dictionary requires a video dataset with audio (not RAF_DB)")
                print("=== Creating OpenSMILE audio dictionary from source train data ===")
                create_opensmile_audio_dictionary(srcs_loader, val_loader)
                print("=== OpenSMILE audio dictionary created ===")
                return

            # if not args.infer_audio_dictionary:
            print("=== Inference OpenSMILE audio dictionary from source train data ===")
            # vis_result = visualize_opensmile_clusters_standalone_withoutnoise(
            #     dictionary_dir="outputs/opensmile_activ_dbscan_feat_dict",
            #     method="tsne",
            #     perplexity=30,
            #     save_dir="outputs/cluster_visualizations",
            # )
            # vis_result = visualize_opensmile_clusters_standalone_withoutnoise(
            #     dictionary_dir=args.audio_dict_dir,
            #     method="umap",
            #     perplexity=30,
            #     save_dir="outputs/cluster_visualizations_umap",
            # )
            infer_opensmile_dbsacn_feature_cluster_dictionary(test_loader=val_loader, dictionary_dir=args.audio_dict_dir, sub_id=tar_subject_id, tar_sub_code=tar_sub_list)
            # infer_opensmile_feature_cluster_dictionary(val_loader)
            # infer_opensmile_audio_dictionary(val_loader)
            print("=== Done audio dictionary Infer ===")
            continue

        emoclip_model = None
        # emoclip_model = VClip(args.arch, device)
        # model = load_partial_checkpoint(model, 'ViT_B_32_bah.pth')
        # emoclip_model = load_clip_encoders('1.pth', emoclip_model, device) # 1.pth
        if not args.create_audio_dictionary and args.load_t_adpt_cl_mod:
            saved_model = load_txt_adapter_classifier(source_model_path, model, args, device)
            print("[INFO] AU Adapter and Classifier Loaded: ", source_model_path)
            # saved_model.visualize_au_text_embeddings(AU_PROMPTS, device="cuda")

            # plot_top_au_clusters_diagonal(model, val_loader, AU_PROMPTS, classnames, args.gpu, device)
            # tsne_on_aus(model, val_loader, AU_PROMPTS, classnames, args.gpu, device)
            # plot_average_au_by_class(model, val_loader, AU_PROMPTS, classnames, args.gpu, device)
            # plot_tsne_au_vectors(model, val_loader, AU_PROMPTS, args.gpu, device)
            # plot_au_class_heatmap(model, AU_PROMPTS, classnames)
            if args.eval_au_adpt_cl:
                evaluate_txt_adapter_n_au_classifier(args, saved_model, val_loader if args.current_ds is config.RAF_DB else srcs_val_loader)
                break
            elif args.eval_au_tar_sb:
                # plot_positive_aus_per_class(model, AU_PROMPTS, classnames)
                TSNE_ALL_SUB_VIDEOS, TSNE_ALL_SUB_LABLES = evaluate_txt_adapter_n_au_classifier(args, model, val_loader, sub_id, tar_sub_list, args.eval_au_tar_sb)
                continue
        # continue
        if not args.create_audio_dictionary and (args.train_t_adpt_cl or args.train_whole_clip_model):
            model = train_txt_adapter_n_au_classifier(args, model, emoclip_model, val_loader if args.current_ds is config.RAF_DB else srcs_loader, srcs_val_loader,
                                            scaler, optimizer, optim_state, classnames, source_model_path, num_epochs=args.t_adap_epoch, 
                                            save_path=source_model_path, train_clip_model=args.train_whole_clip_model)
            evaluate_txt_adapter_n_au_classifier(args, model, val_loader if args.current_ds is config.RAF_DB else srcs_val_loader, set_id, tar_sub_list)
            break
        if not args.create_audio_dictionary and args.adapt_tar_sub:
            target_path = os.path.join(target_weight_path, 'adapt_model.pth')
            # if args.adapt_per_video:
            #     model = target_adapt_per_video( args, model, val_loader, scaler, source_model_path,
            #                         num_epochs=args.t_adap_epoch, save_path=target_path, train_clip_model=args.train_whole_clip_model)
            # else:
            model = train_txt_adapter_n_au_classifier(args, model, emoclip_model, val_loader, None, scaler, optimizer, optim_state, classnames, source_model_path, 
                                    num_epochs=args.t_adap_epoch, save_path=target_path, train_clip_model=args.train_whole_clip_model)
            continue
        # cal_sim_clprompts_auprompts(args, model, set_id, val_loader)
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: WAR {:.2f} | UAR {:.2f} | F1 {:.2f}".format(
                set_id,
                results[set_id]['war'],
                results[set_id]['uar'],
                results[set_id]['f1_macro']
            ))

            with open(file_name, "a") as f:  
                f.write("=> Acc. on testset [{}]: WAR {:.2f} | F1 {:.2f} | UAR {:.2f}\n".format(
                    set_id,
                    results[set_id]['war'],
                    results[set_id]['uar'],
                    results[set_id]['f1_macro']
                ))
        except KeyError:
            print("=> Acc. on testset [{}]: {:.2f}".format(set_id, results[set_id]))

    # Concatenate across subjects
    # TSNE_ALL_SUB_VIDEOS = torch.cat(TSNE_ALL_SUB_VIDEOS, dim=0)  # [N_total, 512]
    # TSNE_ALL_SUB_LABLES = torch.cat(TSNE_ALL_SUB_LABLES, dim=0)              # [N_total]
    # model.visualize_video_text_au_tsne_paper(TSNE_ALL_SUB_VIDEOS, AU_PROMPTS, args.adapt_tar_sub, args.key_frame_sel, args.key_frames, TSNE_ALL_SUB_LABLES, device="cuda")

    print("\n======== Result Summary ========")
    print("Zero-shot CLIP evaluation results:")
    print("\t[Dataset]\tWAR.\tUAR\tF1(macro)")

    for id in results.keys():
        r = results[id]
        print(f"{id}\t{r['war']:.2f}\t{r['uar']:.2f}\t{r['f1_macro']:.2f}")



def create_opensmile_audio_dictionary(
    train_loader,
    val_loader,
    output_dir="outputs/opensmile_audio_dictionary_4",
    sample_rate=16000,
    prototypes_per_class=4,
):
    """
    Create OpenSMILE audio dictionary from train_loader.

    Expected train_loader batch:
        feat, images, audio_arr, labels, video_path

    Args:
        train_loader: PyTorch DataLoader
        output_dir: folder to save dictionary files
        sample_rate: audio sample rate
        prototypes_per_class:
            1 = one prototype per emotion class
            >1 = multiple prototypes per emotion class using KMeans

    Returns:
        result dictionary with dictionary, labels, features, etc.
    """

    # builder = OpenSMILEAudioDictionaryBuilder(
    #     sample_rate=sample_rate,
    #     prototypes_per_class=prototypes_per_class,
    # )

    # result = builder.fit_from_loader(
    #     train_loader=train_loader,
    #     output_dir=output_dir,
    # )

    """
    Build an audio dictionary by clustering similar OpenSMILE feature patterns.

    Important:
        Labels are NOT used to create clusters.
        Labels are only used after clustering to assign a majority-vote class
        to each cluster for evaluation/prediction.

    Pipeline:
        audio_arr
        → OpenSMILE feature
        → normalize
        → KMeans clustering over all samples
        → dictionary = cluster centers
        → cluster label = majority GT label inside that cluster
    """

    min_cluster_size_values = [3, 5, 7, 10, 15]
    min_samples_values = [1, 2, 3, 5, 7]

    all_results = []

    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:

            # Optional: avoid overly strict settings
            if min_samples > min_cluster_size:
                continue

            output_dir = (
                "outputs/opensmile_activ_dbscan_feat_dict/"
                f"mcs_{min_cluster_size}_ms_{min_samples}"
            )

            os.makedirs(output_dir, exist_ok=True)

            print("=" * 80)
            print(f"Running HDBSCAN with:")
            print(f"min_cluster_size = {min_cluster_size}")
            print(f"min_samples      = {min_samples}")
            print(f"output_dir       = {output_dir}")
            print("=" * 80)

            cluster_dict = OpenSMILEActivatedFeatureClusterDictionary(
                sample_rate=16000,
                num_clusters=8,              # not used by HDBSCAN, but can keep it
                activation_threshold=0,
                positive_only=False,
            )

            result = cluster_dict.fit_hdbscan(
                train_loader=train_loader,
                output_dir=output_dir,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                include_noise_dictionary=False,
            )

            # Store useful run information
            run_info = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "output_dir": output_dir,
                "result": result,
            }

            all_results.append(run_info)

    # Save full sweep summary
    summary_path = "outputs/opensmile_activ_dbscan_feat_dict/hdbscan_sweep_summary.json"

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved HDBSCAN sweep summary to: {summary_path}")

    # cluster_dict = OpenSMILEActivatedFeatureClusterDictionary(
    #     sample_rate=16000,
    #     num_clusters=8,
    #     activation_threshold=0,
    #     positive_only=False,
    # )

    # result = cluster_dict.fit(
    #     train_loader=train_loader,
    #     output_dir="outputs/opensmile_activated_feature_cluster_dictionary"
    # )

    # result = cluster_dict.fit_hdbscan(
    #     train_loader=train_loader,
    #     output_dir="outputs/opensmile_activ_dbscan_feat_dict",
    #     min_cluster_size=7,
    #     min_samples=5,
    #     include_noise_dictionary=False,
    # )

    # train_metrics = cluster_dict.evaluate(train_loader)
    # print("Source Train: ", train_metrics)
    # test_metrics = cluster_dict.evaluate(val_loader)
    # print("Source Test: ", test_metrics)

    # vis_result = cluster_dict.visualize_clusters(
    #     dictionary_dir="outputs/opensmile_activ_dbscan_feat_dict",
    #     method="umap",
    #     save_dir="outputs/opensmile_activ_dbscan_feat_dict/visualization",
    # )

    # cluster_ids = result["cluster_ids"]

    # num_noise = np.sum(cluster_ids == -1)
    # total = len(cluster_ids)

    # print("Noise samples:", num_noise, "/", total)
    # print("Noise percentage:", 100 * num_noise / total)
    # print("Discovered clusters:", sorted(set(cluster_ids) - {-1}))

    # print_cluster_top_features(result, top_k=10)
    # return result

def infer_opensmile_audio_dictionary(train_loader, output_dir="outputs/opensmile_audio_dictionary", label_to_name=None):
    inferencer = OpenSMILEAudioDictionaryInferencer(
        dictionary_dir="outputs/opensmile_audio_dictionary_4",
        sample_rate=16000,
    )

    y_true = []
    y_pred = []

    for feat, audio_arr, labels, video_path in tqdm(train_loader, desc="Evaluating"):

        # Move to CPU numpy
        if isinstance(audio_arr, torch.Tensor):
            audio_arr = audio_arr.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # audio_arr: [B, T]
        # labels: [B]
        batch_size = audio_arr.shape[0]

        for i in range(batch_size):
            one_audio = audio_arr[i]      # [T]
            true_label = labels[i]

            # convert GT label if needed
            if label_to_name is not None:
                true_label = label_to_name[int(true_label)]

            # predict by closest dictionary prototype
            sim_result = inferencer.compute_similarity(one_audio)

            pred_label = sim_result["best_label"]

            if isinstance(pred_label, np.generic):
                pred_label = pred_label.item()

            y_true.append(true_label)
            y_pred.append(pred_label)

    # -------------------------
    # Final results
    # -------------------------
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    total = len(y_true)

    war = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n================ Audio Dictionary Results ================")
    print(f"Correct: {correct}/{total}")
    print(f"WAR / Accuracy: {war:.4f}")
    print(f"UAR / Macro Recall: {uar:.4f}")
    print(f"Macro F1: {f1:.4f}")

    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "correct": correct,
        "total": total,
        "war": war,
        "uar": uar,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_cluster_top_features(result, top_k=10):
    dictionary = result["dictionary"]
    feature_names = result["feature_names"]
    cluster_metadata = result["cluster_metadata"]

    for cluster_id, center in enumerate(dictionary):
        top_idx = np.argsort(np.abs(center))[::-1][:top_k]

        print(f"\nCluster {cluster_id}")
        print("Assigned label:", cluster_metadata[cluster_id]["assigned_label_majority_vote"])
        print("Num samples:", cluster_metadata[cluster_id]["num_samples"])
        print("Label distribution:", cluster_metadata[cluster_id]["label_distribution"])

        print("Top features:")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {center[idx]:.4f}")

def infer_opensmile_feature_cluster_dictionary(test_loader, label_to_name=None, print_report=True):

    cluster_dict = OpenSMILEFeatureClusterDictionary(sample_rate=16000).load(
        "outputs/opensmile_feature_cluster_dictionary"
    )

    """
    Inference/evaluation for OpenSMILE feature-cluster audio dictionary.

    This version assumes the dictionary was built by clustering samples
    based on similar OpenSMILE feature patterns, not by grouping labels first.

    Expected loader batch:
        feat, audio_arr, labels, video_path

    Expected audio_arr:
        [B, T]

    Expected labels:
        [B]

    Args:
        test_loader:
            DataLoader for evaluation.

        cluster_dict:
            A fitted OpenSMILEFeatureClusterDictionary object.
            It must already contain:
                cluster_dict.dictionary
                cluster_dict.cluster_labels
                cluster_dict.scaler
                cluster_dict.smile

        label_to_name:
            Optional dict to map numeric labels to names.
            Example:
                {
                    0: "anger",
                    1: "disgust",
                    2: "fear",
                    3: "happy",
                    4: "neutral",
                    5: "sad",
                    6: "surprise",
                }

        print_report:
            Whether to print classification report.

    Returns:
        Dictionary containing metrics and predictions.
    """

    y_true = []
    y_pred = []
    y_cluster = []
    y_score = []

    for feat, audio_arr, labels, video_path in tqdm(test_loader, desc="Evaluating feature-cluster dictionary"):

        # Move to CPU numpy
        if isinstance(audio_arr, torch.Tensor):
            audio_arr = audio_arr.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Expected:
        # audio_arr: [B, T]
        # labels:    [B]
        batch_size = audio_arr.shape[0]

        for i in range(batch_size):
            one_audio = audio_arr[i]
            true_label = labels[i]

            # Convert GT label if needed
            if label_to_name is not None:
                true_label = label_to_name[int(true_label)]

            # Prediction:
            # audio -> OpenSMILE -> normalize -> closest cluster center
            result = cluster_dict.predict_one(one_audio)

            pred_label = result["pred_label"]
            best_cluster = result["best_cluster"]
            best_score = result["best_score"]

            # Convert numpy scalar to Python scalar
            if isinstance(pred_label, np.generic):
                pred_label = pred_label.item()

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_cluster.append(best_cluster)
            y_score.append(best_score)

    # -------------------------
    # Final results
    # -------------------------
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)

    war = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels_order = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    print("\n================ Feature-Cluster Audio Dictionary Results ================")
    print(f"Correct: {correct}/{total}")
    print(f"WAR / Accuracy: {war:.4f}")
    print(f"UAR / Macro Recall: {uar:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nLabels order:")
    print(labels_order)

    print("\nConfusion Matrix:")
    print(cm)

    if print_report:
        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels_order,
                zero_division=0,
            )
        )

    return {
        "correct": correct,
        "total": total,
        "war": war,
        "uar": uar,
        "f1": f1,
        "labels_order": labels_order,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "predicted_cluster": np.asarray(y_cluster),
        "best_similarity_score": np.asarray(y_score),
    }

def infer_opensmile_dbsacn_feature_cluster_dictionary(
    test_loader,
    dictionary_dir="outputs/opensmile_feature_cluster_dictionary",
    label_to_name=None,
    print_report=True,
    sub_id=None, 
    tar_sub_code=None,
):
    """
    Inference/evaluation for OpenSMILE activation feature-cluster dictionary.

    Works with:
        KMeans activation dictionary
        HDBSCAN activation dictionary

    Expected loader batch:
        feat, audio_arr, labels, video_path

    or:
        feat, images, audio_arr, labels, video_path

    Prediction:
        audio
        -> OpenSMILE
        -> StandardScaler
        -> threshold activation
        -> L2 normalize
        -> cosine similarity to dictionary atoms
        -> closest dictionary atom
        -> predicted label = majority label of closest atom
    """

    cluster_dict = OpenSMILEActivatedFeatureClusterDictionary(sample_rate=16000).load(
        dictionary_dir
    )

    y_true = []
    y_pred = []
    y_cluster = []
    y_score = []
    y_active_count = []

    for batch in tqdm(test_loader, desc="Evaluating feature-cluster dictionary"):

        if len(batch) == 4:
            feat, audio_arr, labels, video_path = batch
        elif len(batch) == 5:
            feat, images, audio_arr, labels, video_path = batch
        else:
            raise ValueError(
                f"Expected batch length 4 or 5, got {len(batch)}"
            )

        # Move to CPU numpy
        if isinstance(audio_arr, torch.Tensor):
            audio_arr = audio_arr.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Expected audio_arr: [B, T]
        # Expected labels: [B]
        batch_size = audio_arr.shape[0]

        for i in range(batch_size):
            one_audio = audio_arr[i]
            true_label = labels[i]

            if label_to_name is not None:
                true_label = label_to_name[int(true_label)]

            result = cluster_dict.predict_one(one_audio)

            pred_label = result["pred_label"]
            best_cluster = result["best_cluster"]
            best_score = result["best_score"]

            num_active_features = result.get("num_active_features", None)

            # Convert numpy scalar to Python scalar
            if isinstance(pred_label, np.generic):
                pred_label = pred_label.item()

            if isinstance(true_label, np.generic):
                true_label = true_label.item()

            # Skip invalid predictions if any cluster has no assigned label
            if pred_label is None:
                continue

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_cluster.append(best_cluster)
            y_score.append(best_score)

            if num_active_features is not None:
                y_active_count.append(num_active_features)

    # -------------------------
    # Final results
    # -------------------------
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)

    war = accuracy_score(y_true, y_pred)
    uar = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    labels_order = sorted(
        np.unique(
            np.concatenate([y_true, y_pred])
        ).tolist()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_order,
    )

    print("\n================ Feature-Cluster Audio Dictionary Results ================")
    print(f"Dictionary path: {dictionary_dir}")
    print(f"Evaluated samples: {total}")
    print(f"Correct: {correct}/{total}")
    print(f"WAR / Accuracy: {war:.4f}")
    print(f"UAR / Macro Recall: {uar:.4f}")
    print(f"Macro F1: {f1:.4f}")

    if len(y_active_count) > 0:
        print(f"Average active features: {np.mean(y_active_count):.2f}")

    print("\nLabels order:")
    print(labels_order)

    print("\nConfusion Matrix:")
    print(cm)

    if print_report:
        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels_order,
                zero_division=0,
            )
        )

    min_cluster_size = dictionary_dir.split("mcs_")[1].split("_ms_")[0]
    min_samples = dictionary_dir.split("mcs_")[1].split("_ms_")[1]
    if str(sub_id).strip() in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        update_subject_result_xlsx(
            technique=f"mcs_{min_cluster_size}_ms_{min_samples}",
            filename="experiment_logs_audio_dict.xlsx", 
            bs=args.batch_size, 
            temp=1, 
            subject_name=f"Sub-{sub_id}", 
            subject_code=tar_sub_code, 
            war=f"{war:.4f}",
            uar=f"{uar:.4f}", 
            f1=f"{f1:.4f}",
            is_last_subject=True # Set to True only for your final subject (e.g., Sub-10)
        )

    return {
        "correct": correct,
        "total": total,
        "war": war,
        "uar": uar,
        "f1": f1,
        "labels_order": labels_order,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "predicted_cluster": np.asarray(y_cluster),
        "best_similarity_score": np.asarray(y_score),
        "active_count": np.asarray(y_active_count),
        "dictionary_dir": dictionary_dir,
    }

def train_txt_adapter_n_au_classifier(args, model, emoclip_model, train_loader, val_loader, scaler, optimizer, optim_state, classnames, source_model_path,
                                      num_epochs=10, save_path="clip_au_model_vit32_au46.pth", train_clip_model=False):
    # model.to(device)
    # optimizer = torch.optim.Adam(
    #     list(model.text_adapter.parameters()) + list(model.au_classifier.parameters()),
    #     lr=1e-3
    # )
    optimizer.load_state_dict(optim_state)

    if args.current_ds is config.BAH:
        class_prompt = CLASS_PROMPTS_AMBV
    else:
        class_prompt = CLASS_PROMPTS

    criterion = nn.CrossEntropyLoss()
    if args.adapt_tar_sub:
        model = load_txt_adapter_classifier(source_model_path, model, args, device)
        optimizer = rebuild_optimizer(model, args.lr)
    # with torch.no_grad():
    #     model.reset()
    # model.reset_states()
    # optimizer.zero_grad(set_to_none=True)

    # count_trainable_params(model)
    times = []
    w_align = 1

    save_dict = {}
    best_acc, best_uar, best_f1 = 0, 0, 0
    best_model = deepcopy(model.state_dict())
    logger = MetricLogger(save_dir="metrics", filename="comparison_tran.csv")

    # Initialize Grad-CAM for last conv layer in temporal CNN
    cam_extractor = GradCAM(model.temporal, target_layer="conv3")  # or "cnn.2" if Sequential
#   trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#   print("Trainable params:", trainable)
    # for nom, parametre in model.named_parameters():
    #     print(f"{nom}: {parametre.requires_grad}")
    for epoch in range(num_epochs):
        # model.train()
        running_loss = 0.0
        total_samples, itera = 0, 0

        all_labels = []
        all_preds = []

        # class_weights = [(epoch*0.05) + 0.05, 1 - ((epoch*0.05) + 0.05)]
        class_weights = None

        for images, audio_arr, labels, video_path in tqdm(train_loader):
            if args.adapt_per_video:
                model = load_txt_adapter_classifier(source_model_path, model, args, device)
                optimizer = rebuild_optimizer(model, args.lr)
            # images = images.to(device)
            images = images.cuda(args.gpu, non_blocking=True) if args.current_ds is not config.RAF_DB else images[0].cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            audio_arr = audio_arr.cuda(args.gpu, non_blocking=True)
            optimizer.zero_grad()

            # model.visualize_video_alignment_tsne(images, AU_PROMPTS, args.adapt_tar_sub, args.key_frame_sel, args.key_frames, device="cuda")
            # model.visualize_video_text_au_tsne(images, AU_PROMPTS, args.adapt_tar_sub, args.key_frame_sel, args.key_frames, labels, device="cuda")
            # gradcam_temporal_spatial(emoclip_model, images, CLASS_PROMPTS, device, itera)
            # top_aus, scores = visualize_au_gradcam(model, emoclip_model, images, AU_PROMPTS, device, itera, args, top_k=5, visualize_face=True)
            # visualize_cam(cam_extractor, images, device, itera)
            # print(top_aus, scores)

            # optimizer.zero_grad()

            # for j in range(images.size(0)):
            #     img_tensor = images[j].detach().cpu()

            #     # Unnormalize if you used transforms.Normalize
            #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            #     std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            #     img_tensor = img_tensor * std + mean

            #     img_pil = torchvision.transforms.functional.to_pil_image(img_tensor)

            #     # save file with label in name
            #     # label = target[j].item()
            #     # save_path = os.path.join(save_dir, f"batch{i}_img{j}_label{label}.png")
            #     img_pil.save(f"samples/au_batch_img{j}.png")

            # automatic mixed precision context
            with torch.cuda.amp.autocast():
                if args.adapt_tar_sub:
                    # start = time.time()
                    logits, au_sim, _, L_align = model(images, audio_arr, AU_PROMPTS, class_prompt if args.include_cls_prompt else None, mode=("temporal" if args.is_video_clip else "au"), 
                                adapt_target=args.adapt_tar_sub, key_frame_sel=args.key_frame_sel, train_whole_clip=train_clip_model, key_frames=args.key_frames, 
                                fus_type=args.fus_type, mod_align=args.is_mod_align, is_opensmile_dict=args.is_opensmile_dict)   # forward pass
                    
                    # feature_cam_visualization(model, au_sim, itera, device, target_class=None, top_k=5)

                    # AU-based prior
                    # au_prior = compute_class_au_prior(au_sim, model, AU_PROMPTS, classnames, top_k=5)
                    # au_prior = au_prior.cuda(args.gpu, non_blocking=True)

                    
                    # pred_prob = F.softmax(logits, dim=-1)
                    loss = entropy_loss(logits, class_weights)    # Use CrossEntropyLoss; when not using softmax prob of visual and audio

                    # loss = F.nll_loss(
                    #     torch.log(logits.clamp_min(1e-8)),
                    #     labels,
                    # )

                    # fused_probs = (prob * au_prior)
                    # fused_probs = fused_probs / fused_probs.sum()

                    # _, _ = visualize_au_gradcam(model, None, images, labels.cpu(), pred_prob, AU_PROMPTS, device, itera, epoch, args, top_k=45, visualize_face=True)


                    # conf, preds = prob.max(dim=-1)
                    # mask_conf = conf > 0.9 
                    # loss_pl = criterion(logits[mask_conf], preds[mask_conf]) if mask_conf.any() else 0

                    # loss = loss + loss_pl

                else:
                    # start = time.time()
                    logits, au_sim, _, L_align = model(images, audio_arr, AU_PROMPTS, CLASS_PROMPTS if args.include_cls_prompt else None, mode=("temporal" if args.is_video_clip else "au"), 
                                adapt_target=args.adapt_tar_sub, train_whole_clip=train_clip_model, fus_type=args.fus_type, mod_align=args.is_mod_align, is_opensmile_dict=args.is_opensmile_dict)   # forward pass
                    # loss = criterion(logits, labels)     # loss, DO NOT .item() here
                    
                    loss = F.nll_loss(
                        torch.log(logits.clamp_min(1e-8)),
                        labels,
                    )

                    if train_clip_model:
                        loss_t = criterion(logits, labels)
                        loss = (loss + loss_t) / 2
                
                    # Log the metrics
                    # logger.log(epoch, model_name="transformer", au_sim=au_sim, logits=logits)
            itera = itera + 1
            if L_align is not None:
                t_loss = loss + (w_align*L_align)
            else:
                t_loss = loss 
            # optimizer.zero_grad()
            scaler.scale(t_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # torch.cuda.synchronize()  # ensure timing accuracy
            # end = time.time()
            # times.append(end - start)

            # print(model.au_prompt_learner.ctx.grad.abs().mean().item())
            # print("Loss:", loss.item())
            # for name, p in model.named_parameters():
            #     if p.requires_grad and p.grad is not None:
            #         print(name, "grad:", p.grad.abs().mean().item())

            # for name, p in model.au_prompt_learner.named_parameters():
            #     print(name, p.requires_grad)


            batch_size = images.size(0)
            running_loss += t_loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            if itera == args.iter_limit:
                break

        # end of epoch metrics
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # avg_time = sum(times) / len(times)
        # print(f"Average per-batch time: {avg_time:.4f} sec")

        # per subject
        # print("Counts:", np.bincount(all_labels, minlength=2))
        # print("Confusion:\n", confusion_matrix(all_labels, all_preds, labels=[0,1]))
        # print("Per-class recall:", recall_score(all_labels, all_preds, average=None, labels=[0,1]))

        epoch_loss = running_loss / total_samples

        epoch_acc  = accuracy_score(all_labels, all_preds)
        epoch_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_uar  = recall_score(all_labels, all_preds, average='macro', zero_division=0)  # UAR


        # val_res = evaluate_txt_adapter_n_au_classifier(args, model, val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f}  "
              f"WAR: {epoch_acc:.4f}  "
              f"UAR: {epoch_uar:.4f} "
              f"F1: {epoch_f1:.4f}  ")

        experiment.log_metric("Loss:", epoch_loss, epoch=epoch)
        experiment.log_metric("WAR:", epoch_acc, epoch=epoch)
        experiment.log_metric("UAR:", epoch_uar, epoch=epoch)
        experiment.log_metric("WAF1R:", epoch_f1, epoch=epoch)

        if best_acc < epoch_acc or best_f1 < epoch_f1:
            if best_acc < epoch_acc: 
                best_acc = epoch_acc
            if best_uar < epoch_uar:
                best_uar = epoch_uar
            if best_f1 < epoch_f1: 
                best_f1 = epoch_f1
            best_model = deepcopy(model)
            if args.is_video_clip:
                if train_clip_model:
                    torch.save(best_model.state_dict(), save_path)
                    print(f"[INFO] Saving Complete CLIP model")
                elif args.adapt_tar_sub:
                    print("[INFO] SKIPPING Not saving the weights")
                    continue
                    # if using the new temporal transformer
                    if hasattr(model, "temporal_classifier"):
                        save_dict["temporal_classifier"] = model.temporal_classifier.state_dict()
                    print(f"[INFO] Saving Target video model Subject-Specific Adapter")
                    torch.save(save_dict, save_path)
                else:
                    save_dict["text_adapter"] = model.text_adapter.state_dict()

                    # if using the new temporal transformer
                    if hasattr(model, "temporal"):
                        save_dict["temporal"] = model.temporal.state_dict()
                    if hasattr(model, "temporal_proj"):
                        save_dict["temporal_proj"] = model.temporal_proj.state_dict()
                    if hasattr(model, "temporal_classifier"):
                        save_dict["temporal_classifier"] = model.temporal_classifier.state_dict()
                    if hasattr(model, "audio_fusion_alpha"):
                        save_dict["audio_fusion_alpha"] = model.audio_fusion_alpha.detach().cpu().clone()
                    
                    print(f"[INFO] Saving video model components: {list(save_dict.keys())}")
                    torch.save(save_dict, save_path)
            else:
                # Save only the trainable parts
                best_model = deepcopy(model)
                torch.save({
                    'text_adapter': model.text_adapter.state_dict(),
                    'au_classifier': model.au_classifier.state_dict()
                }, save_path)
                print(f"Model saved to {save_path}")
    
     # 🏁 Load best weights before returning
    # model.load_state_dict(best_model)
    save_best_metrics(war=best_acc, uar=best_uar, f1=best_f1, filename="best_metrics_"+str(args.target_sub_set)+".csv")
    return best_model

    criterion = nn.CrossEntropyLoss()
    best_overall_acc = 0.0
    best_overall_model = deepcopy(model.state_dict())
    logger = MetricLogger(save_dir="metrics", filename="comparison.csv")
    video_accuracies = []   # Track per-video accuracy
    video_uar = []   # Track per-video accuracy
    video_f1 = []   # Track per-video accuracy


    # Loop over videos
    for video_idx, video_loader in enumerate(train_loaders):
        print(f"\n🚀 Starting adaptation for video {video_idx}/{len(train_loaders)}")

        # --- 🔁 Reset model and optimizer for this video ---
        model = load_txt_adapter_classifier(source_model_path, model, args, device)
        optimizer = rebuild_optimizer(model, args.lr)

        best_video_acc = 0.0
        best_video_uar = 0.0
        best_video_f1 = 0.0
        best_video_model = deepcopy(model.state_dict())

        images, labels = video_loader

        # --- Train for N epochs on this single video ---
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            all_labels, all_preds = [], []

            # --- Data to device ---
            images = (
                images.cuda(args.gpu, non_blocking=True)
                if args.current_ds is config.BIOVID
                else images[0].cuda(args.gpu, non_blocking=True)
            )
            labels = labels.cuda(args.gpu, non_blocking=True)

            # --- Forward pass ---
            with torch.cuda.amp.autocast():
                if args.adapt_tar_sub:
                    logits = model(
                        images, AU_PROMPTS,
                        CLASS_PROMPTS if args.include_cls_prompt else None,
                        mode=("temporal" if args.is_video_clip else "au"),
                        adapt_target=args.adapt_tar_sub,
                        train_whole_clip=train_clip_model
                    )
                    loss = entropy_loss(logits)
                else:
                    logits, au_sim = model(
                        images, AU_PROMPTS,
                        CLASS_PROMPTS if args.include_cls_prompt else None,
                        mode=("temporal" if args.is_video_clip else "au"),
                        adapt_target=args.adapt_tar_sub,
                        train_whole_clip=train_clip_model
                    )
                    loss = criterion(logits, labels)
                    if train_clip_model:
                        loss_t = criterion(logits, labels)
                        loss = (loss + loss_t) / 2

            # --- Backprop ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Metrics ---
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            # --- End of epoch ---
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            epoch_loss = running_loss / total_samples
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            epoch_uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)

            print(f"Video {video_idx+1} | Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {epoch_loss:.4f}  WAR: {epoch_acc:.4f}  UAR: {epoch_uar:.4f}  F1: {epoch_f1:.4f}")

            # Log
            # logger.log(epoch, model_name="temporal_model", au_sim=None, logits=None)
            experiment.log_metric("Loss", epoch_loss, step=epoch)
            experiment.log_metric("WAR", epoch_acc, step=epoch)
            experiment.log_metric("UAR", epoch_uar, step=epoch)
            experiment.log_metric("F1", epoch_f1, step=epoch)

            # --- Save best model for this video ---
            if epoch_acc > best_video_acc:
                best_video_acc = epoch_acc
                best_video_uar = epoch_uar
                best_video_f1 = epoch_f1
                best_video_model = deepcopy(model.state_dict())

        # --- 🏁 End of video ---
        print(f"✅ Best accuracy for video {video_idx+1}: {best_video_acc:.4f}")
        video_accuracies.append(best_video_acc)  # <--- Add this
        video_uar.append(best_video_uar)
        video_f1.append(best_video_f1)

        # Optionally save this video-specific model
        # save_dict = {
        #     "text_adapter": model.text_adapter.state_dict(),
        #     "temporal": model.temporal.state_dict() if hasattr(model, "temporal") else None,
        #     "temporal_classifier": model.temporal_classifier.state_dict() if hasattr(model, "temporal_classifier") else None,
        # }
        # torch.save(save_dict, f"{save_path.replace('.pth', f'_video{video_idx+1}.pth')}")
        # print(f"💾 Saved adapted model for video {video_idx+1}")

        # --- Track best across all videos ---
        if best_video_acc > best_overall_acc:
            best_overall_acc = best_video_acc
            best_overall_model = deepcopy(best_video_model)

    if len(video_accuracies) > 0:
        avg_acc = sum(video_accuracies) / len(video_accuracies)
        avg_uar = sum(video_uar) / len(video_uar)
        avg_f1 = sum(video_f1) / len(video_f1)
        print(f"\n📊 Average accuracy across all {len(video_accuracies)} videos: {avg_acc:.4f} {avg_uar:.4f} {avg_f1:.4f}")
    else:
        print("⚠️ No videos processed, average accuracy not available.")
    # print(f"\n🏆 Best overall accuracy across all videos: {best_overall_acc:.4f}")
    model.load_state_dict(best_overall_model)
    return model

def get_effective_au_weights(model):
    classifier = model.temporal_classifier

    # Find the two linear layers
    linear_layers = [m for m in classifier if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) == 1:
        W_eff = linear_layers[0].weight.detach().cpu()
    else:
        # Combine W2 (num_classes × hidden) and W1 (hidden × num_aus)
        W1 = linear_layers[0].weight.detach().cpu()
        W2 = linear_layers[1].weight.detach().cpu()
        W_eff = W2 @ W1  # shape: [num_classes, num_aus]
    return W_eff.numpy()

def get_top_aus(au_sim, au_prompts, top_k=5):
    """
    au_sim: [B, num_AUs] tensor of similarity scores
    au_prompts: list of AU names
    """
    au_sim_np = au_sim.detach().cpu().numpy()

    for i, sim_vec in enumerate(au_sim_np):
        sim_vec = np.asarray(sim_vec).flatten()
        top_idx = sim_vec.argsort()[-top_k:][::-1]
        print(f"\n🎥 Video {i+1}: Top {top_k} AUs")
        for rank, idx in enumerate(top_idx):
            print(f"  {rank+1}. {au_prompts[idx]} (score: {sim_vec[idx]:.4f})")

    return top_idx, sim_vec

def compute_class_au_prior(au_sim, model, au_prompts, classnames, top_k=5, temp=1.0):
    """
    Compute AU-based class compatibility probability.
    Used to re-weight softmax logits.

    Args:
        au_sim (torch.Tensor): [num_AUs] similarity scores for this video
        W_eff (np.ndarray): [num_classes, num_AUs] effective classifier weights
        au_prompts (list[str]): AU names
        classnames (list[str]): class names
        top_k (int): number of top AUs to consider
        temp (float): temperature for softmax smoothing

    Returns:
        class_probs (torch.Tensor): [num_classes] AU-based probability prior
    """
    W_eff = get_effective_au_weights(model)

    top_idx, sim_vec = get_top_aus(au_sim, au_prompts, top_k=5)
    # Ensure shapes
    if isinstance(W_eff, torch.Tensor):
        W_eff = W_eff.detach().cpu().numpy()
    au_sim_np = au_sim.detach().cpu().numpy()

    # au_sim_np = au_sim_np.tolist()

    # ✅ ensure it's a flat 1D array
    # top_idx = np.argsort(au_sim_np)[-top_k:][::-1].astype(int)

    # ✅ top_idx is now list of ints
    top_aus = [au_prompts[int(i)] for i in top_idx]
    top_scores = sim_vec[top_idx]


    # 2️⃣ Compute compatibility of these AUs with each class
    class_scores = []
    for c_idx, cname in enumerate(classnames):
        class_w = W_eff[c_idx, top_idx]   # weights of top AUs for this class
        score = np.sum(top_scores * class_w) / (np.sum(np.abs(class_w)) + 1e-6)
        class_scores.append(score)

    # 3️⃣ Convert to probability distribution
    class_scores = torch.tensor(class_scores)
    class_probs = torch.softmax(class_scores / temp, dim=0)

    # 4️⃣ Optional: Print interpretable info
    print(f"\n🎥 Top-{top_k} AUs: {top_aus}")
    for cname, prob in zip(classnames, class_probs.tolist()):
        print(f"  {cname:<12}: {prob:.3f}")

    return class_probs

def count_trainable_params(model):

    # for name, module in [
    #     ("AU Adapter", model.text_adapter),
    #     ("Temporal (1D-CNN + GLU)", model.temporal),
    #     ("Emotion Classifier", model.temporal_classifier)
    # ]:

    for name, module in [
        ("AU Adapter", model.text_adapter),
        ("AU prompt_learner", model.au_prompt_learner)
    ]:
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {params/1e6:.3f} M params")

    # dummy_text = torch.randint(0, 100, (46,)).to(device)
    # flops_text, _ = profile(model.au_prompt_learner, inputs=(dummy_text,), verbose=False)
    # print(f"Text Adapter GFLOPs per batch: {flops_text / 1e9:.9f}")

    dummy_text = torch.randn(46, 512).to(device)
    flops_text, _ = profile(model.text_adapter, inputs=(dummy_text,), verbose=False)
    print(f"Text Adapter GFLOPs per batch: {flops_text / 1e9:.9f}")

    dummy_text = torch.randn(8, 512, 16).to(device)
    flops_temp, _ = profile(model.temporal, inputs=(dummy_text,), verbose=False)
    print(f"Temporal GFLOPs per batch: {flops_temp / 1e9:.9f}")

    dummy_video = torch.randn(8, 46).to(device)
    flops, _ = profile(model.temporal_classifier, inputs=(dummy_video,), verbose=False)
    print(f"Temporal_classifier GFLOPs per batch: {flops / 1e9:.9f}")


def evaluate_txt_adapter_n_au_classifier(args, model, data_loader, sub_id, tar_sub_code, eval_tar=False):
    # model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    TSNE_ALL_SUB_VIDEOS, TSNE_ALL_SUB_LABLES = [], []
    itr = 0
    with torch.no_grad():
        for images, audio_arr, labels, frame_paths in tqdm(data_loader):
            # top_aus, scores = visualize_au_gradcam(model, images, AU_PROMPTS, device, 000, args, top_k=5, visualize_face=True)
            # print(f"Video path: {frame_paths}")
            audio_arr = audio_arr.cuda(args.gpu, non_blocking=True)

            if args.tpt and not is_video_clip:
                images = images[0].cuda(args.gpu, non_blocking=True)  
            else:
                images = images.cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)

            logits, _, _, _ = model(images, audio_arr, au_prompts=AU_PROMPTS, class_prompts=CLASS_PROMPTS if args.include_cls_prompt else None,
                    mode=("temporal" if args.is_video_clip else "au"), fus_type=args.fus_type, mod_align=args.is_mod_align, is_opensmile_dict=args.is_opensmile_dict)
            # pred_prob = F.softmax(logits, dim=-1) # uncommneted when not using negative likelihood loss
            preds = torch.argmax(logits, dim=-1)
            # print(f"Probs: {pred_prob}")

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # model.visualize_video_text_au_tsne(images, AU_PROMPTS, args.adapt_tar_sub, args.key_frame_sel, args.key_frames, labels, device="cuda")
            
            # -- FOR TSNE
            # TSNE_ALL_SUB_VIDEOS.append(images)
            # TSNE_ALL_SUB_LABLES.append(labels)

            # top_aus, scores = visualize_au_gradcam(model, None, images, labels.cpu(), pred_prob, AU_PROMPTS, device, itr, 0, args, top_k=45, visualize_face=True)
            # itr = itr + 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc  = accuracy_score(all_labels, all_preds) * 100.0
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0
    uar  = recall_score(all_labels, all_preds, average='macro', zero_division=0)  * 100.0 # UAR

    print(f"Eval — WAR: {acc:.3f}  UAR: {uar:.3f}  F1(macro): {f1:.3f} ")

    save_best_metrics(war=acc, uar=uar, f1=f1, filename="clip_au_metrics_"+str(args.target_sub_set)+".csv")

    # At the end of each subject's training/evaluation:
    if str(sub_id).strip() in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        update_subject_result_xlsx(
            technique=f"fus_type:{args.fus_type}; mod_align:{args.is_mod_align}",
            filename="experiment_logs.xlsx", 
            bs=args.batch_size, 
            temp=1, 
            subject_name=f"Sub-{sub_id}", 
            subject_code=tar_sub_code, 
            war=f"{acc:.3f}",
            uar=f"{uar:.3f}", 
            f1=f"{f1:.3f}",
            is_last_subject=(True if sub_id == '9' else False) # Set to True only for your final subject (e.g., Sub-10)
        )

    # TSNE_ALL_SUB_VIDEOS = torch.cat(TSNE_ALL_SUB_VIDEOS, dim=0)  # [N_total, 512]
    # TSNE_ALL_SUB_LABLES = torch.cat(TSNE_ALL_SUB_LABLES, dim=0)              # [N_total]
    # model.visualize_video_text_au_tsne(TSNE_ALL_SUB_VIDEOS, AU_PROMPTS, args.adapt_tar_sub, args.key_frame_sel, args.key_frames, TSNE_ALL_SUB_LABLES, sub_id, device="cuda")


    return TSNE_ALL_SUB_VIDEOS, TSNE_ALL_SUB_LABLES
    return {'acc': acc, 'uar': uar, 'f1_macro': f1 }


def cal_sim_clprompts_auprompts(args, model, set_id, val_loader):
    # ==== STEP-1: AU–Class cosine similarity (text–text only) ====
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    # calculate image with class and AU prompts
    cal_sim_im_cl_au_prompts(args, model, val_loader)

    cls_features, au_features = create_cl_au_prompts(args, model)

    # # 3) Cosine similarity matrix (U x N)
    # au_class_sim = au_features @ cls_features.t()  # [U, N]

    # 3) Cosine similarity matrix U x N (because we normalized, dot = cosine)
    au_class_sim = au_features @ cls_features.t()                      # [U, N]

    # 4) For each class, pick Top-P AU prompts
    topP = args.au_topP
    U, N = au_class_sim.shape
    topP_indices = torch.topk(au_class_sim, k=min(topP, U), dim=0).indices # [topP, N]

    # 5) (Optional) Save CSV (rows = AU prompts, cols = class prompts)
    if args.dump_au_class_sim:
        out_dir = os.path.join("logs", "au_class_sims")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"AU_CLASS_sim_{set_id}.csv")
        save_matrix_csv(csv_path, header_cols=class_prompts, row_names=AU_PROMPTS, matrix_torch=au_class_sim)
        print(f"[AU↔Class] Saved similarity matrix to: {csv_path}")

    # 6) Print Top-P AUs per class for quick inspection
    for c_idx, cls_p in enumerate(class_prompts):
        idxs = topP_indices[:, c_idx].tolist()
        picks = [(AU_PROMPTS[i], float(au_class_sim[i, c_idx])) for i in idxs]
        print(f"\nTop-{topP} AU prompts for class: {cls_p}")
        for name, score in picks:
            print(f"  {score:+.4f}  {name}")

def create_cl_au_prompts(args, model):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")

    classnames = biosub_classes   # Set classes for diff ds

    # 1) Build class prompts from your template
    class_prompts = build_class_prompts_from_template(args.ctx_init, classnames)

    au_features = encode_text_prompts_with_model(args, model, AU_PROMPTS, device)  # [U, D]
    cls_features = encode_text_prompts_with_model(args, model, class_prompts, device)  # [N, D]

    return cls_features, au_features

def cal_sim_im_cl_au_prompts(args, model, val_loader):
    cls_features, au_features = create_cl_au_prompts(args, model)
    for i, (images, target) in enumerate(val_loader):
        images = images[0].cuda(args.gpu, non_blocking=True)

        # 1. Get image features
        img_feats = get_image_features(model, images)  # [B,D]

        # 2. Image ↔ Class similarity
        img_cls_sim = img_feats @ cls_features.t()       # [B,N]
        img_cls_probs = (img_cls_sim / model.logit_scale.exp()).softmax(dim=-1)

        # 3. Image ↔ AU similarity
        img_au_sim = img_feats @ au_features.t()         # [B,U]

        # pick top-K AUs actually present per image (optional)
        topK = 10
        topk_vals, topk_idx = torch.topk(img_au_sim, k=topK, dim=-1)
        # now topk_idx[b] gives indices of top-K AUs for image b

        if args.visualize_img:
            for j in range(images.size(0)):
                img_tensor = images[j].detach().cpu()

                # Unnormalize if you used transforms.Normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_tensor = img_tensor * std + mean

                img_pil = torchvision.transforms.functional.to_pil_image(img_tensor)

                # save file with label in name
                # label = target[j].item()
                # save_path = os.path.join(save_dir, f"batch{i}_img{j}_label{label}.png")
                img_pil.save(f"samples/AU_batch{i}_img{j}.png")

        # you can save or print these
        for b in range(images.size(0)):
            print(f"Image {i}-{b}:")
            print("  Class probs:", img_cls_probs[b].cpu().numpy())
            print("  Target Class: ", target)
            print("  Top-K AU indices:", topk_idx[b].cpu().numpy())
            print("  Top-K AU names:", [AU_PROMPTS[k] for k in topk_idx[b].cpu().tolist()])

        # Example usage:
        pain_top_au = [AU_PROMPTS[i] for i in topk_idx[b].cpu().tolist()]  # your top-10 AUs
        neutral_top_au = [AU_PROMPTS[i] for i in topk_idx[b].cpu().tolist()]

        pain_prompt   = build_hybrid_class_prompt("pain", pain_top_au)
        neutral_prompt = build_hybrid_class_prompt("neutral", neutral_top_au)

        print("Pain prompt:\n", pain_prompt)
        print("Neutral prompt:\n", neutral_prompt)

def build_hybrid_class_prompt(class_name, top_au_names, base_template="a_person_with_an_expression_of_[CLS]_with_features_like_"):
    """
    Combine class name with top AU prompts into one hybrid prompt string.
    """
    # strip "a_person_with_an_expression_of_" from each AU prompt
    stripped = [au.replace("a_person_with_an_expression_of_", "") for au in top_au_names]
    # join with underscores
    au_string = "_".join(stripped)
    # build full prompt
    prompt = base_template.replace("[CLS]", class_name) + au_string
    return prompt

@torch.no_grad()
def get_custom_prompts_for_image(args, model, image_tensor, gt_label, au_features, classnames, AU_PROMPTS, K=10):
    """
    image_tensor: 1x3xHxW tensor
    gt_label: integer index of the ground-truth class in classnames
    classnames: list of class names (e.g. ["neutral","pain"])
    AU_PROMPTS: list of AU prompt strings
    returns: dict with 'hybrid_prompt_gt' and 'default_prompt_other'
    """

    # image feature
    img_feat = model.image_encoder(image_tensor.type(model.dtype))  # [1,D]
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # image↔AU similarity
    img_au_sim = img_feat @ au_features.t()  # [1,U]
    topk_vals, topk_idx = torch.topk(img_au_sim, k=K, dim=-1)

    top_au_prompts = [AU_PROMPTS[i] for i in topk_idx[0].cpu().tolist()]

    # --- Get last N AU prompts
    # Sort all similarities in ascending order
    sorted_vals, sorted_idx = torch.sort(img_au_sim, dim=-1, descending=False)

    # Take the last N (lowest similarity) indices
    lastN_vals = sorted_vals[:, -K:]  # shape [1,N]
    lastN_idx = sorted_idx[:, -K:]    # shape [1,N]

    # Convert indices to actual prompt strings
    lastN_prompts = [AU_PROMPTS[idx] for idx in lastN_idx.squeeze(0).tolist()]

    # small helper to build hybrid prompt
    def build_hybrid_class_prompt(class_name, top_au_names, base_template="a_person_with_expression_of_[CLS]_with_features_"):
        stripped = [au.replace("a_person_with_expression_of_", "") for au in top_au_names]
        au_string = "_".join(stripped)
        return base_template.replace("[CLS]", class_name) + au_string

    # decide which class gets hybrid vs default
    if gt_label == 0:  # neutral is GT
        neutral_prompt = build_hybrid_class_prompt("neutral", top_au_prompts)
        pain_prompt = build_hybrid_class_prompt("pain", lastN_prompts)
        # pain_prompt = "a person with an expression of pain"
    else:  # pain is GT
        pain_prompt = build_hybrid_class_prompt("pain", top_au_prompts)
        neutral_prompt = build_hybrid_class_prompt("neutral", lastN_prompts)
        # neutral_prompt = "a person with an expression of neutral"

    # neutral_prompt = build_hybrid_class_prompt("neutral", top_au_prompts)
    # pain_prompt = build_hybrid_class_prompt("pain", top_au_prompts)
    return neutral_prompt, pain_prompt


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    war_meter = AverageMeter('WAR', ':6.2f', Summary.AVERAGE)
    uar_meter = AverageMeter('UAR', ':6.2f', Summary.AVERAGE)
    f1_meter = AverageMeter('F1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, war_meter, uar_meter, f1_meter],
        prefix='Test: ')

    cls_features, au_features = create_cl_au_prompts(args, model)
    correct = 0
    total = 0

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    # for i, (crops, target) in enumerate(val_loader):
    all_preds = []
    all_targets = []

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        # images = crops.squeeze(0)  # -> [N, C, H, W]
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)
        else:
            images = images[0].cuda(args.gpu, non_blocking=True)

        if args.visualize_img:
            for j in range(images.size(0)):
                img_tensor = images[j].detach().cpu()

                # Unnormalize if you used transforms.Normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img_tensor = img_tensor * std + mean

                img_pil = torchvision.transforms.functional.to_pil_image(img_tensor)

                # save file with label in name
                # label = target[j].item()
                # save_path = os.path.join(save_dir, f"batch{i}_img{j}_label{label}.png")
                img_pil.save(f"samples/batch{i}_img{j}.png")

        # optional: TTA adaptation
        if not args.cocoop:
            # if args.tta_steps > 0:
            #     with torch.no_grad():
            #         model.reset()
            model.train()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # actual inference
        if args.tpt and args.cocoop:
            image_feature = image_feature[0].unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output = model(image, au_prompts=AU_PROMPTS, mode=("adapt" if args.adapt_tar_sub else "clip"))

        # collect predictions
        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

        # compute metrics per batch
        batch_war = (preds == target).sum().item() / target.size(0) * 100.0
        batch_uar = recall_score(target.cpu().numpy(), preds.cpu().numpy(),
                                average='macro', zero_division=0) * 100.0
        batch_f1 = f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0) * 100.0

        # update meters
        war_meter.update(batch_war, n=target.size(0))
        uar_meter.update(batch_uar, n=target.size(0))
        f1_meter.update(batch_f1, n=target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    # concatenate for metrics
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # WAR (accuracy)
    war = (all_preds == all_targets).sum() / len(all_targets) * 100.0
    # UAR (macro recall)
    uar = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0
    # F1 macro
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0

    print(f"\nResults over {len(all_targets)} images:")
    print(f"WAR (Acc): {war:.2f}% | UAR: {uar:.2f}% | F1 (macro): {f1_macro:.2f}%")

    return {
        'war': war,
        'uar': uar,
        'f1_macro': f1_macro
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root', default=config.BIOVID_SOURCE_DATASET_PATH)# BIOVID_SOURCE_DATASET_PATH or RAF_DATASET_CLS_PATH
    
    # Stress
    parser.add_argument('--test_sets', type=str, default='stresssub0/stresssub1/stresssub2/stresssub3/stresssub4/stresssub5/stresssub6/stresssub7/stresssub8/stresssub9', help='test dataset (multiple datasets split by slash)')
    
    # biosubs
    # parser.add_argument('--test_sets', type=str, default='biosub9/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9', help='test dataset (multiple datasets split by slash)')
    # parser.add_argument('--test_sets', type=str, default='biosub6/biosub7/biosub8/biosub9', help='test dataset (multiple datasets split by slash)')

    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/32')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
    parser.add_argument('-tar_b', '--tar_batch-size', default=1, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr') # 5e-3, for biovid and stress = 0.001
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.5, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=10, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')

    parser.add_argument('--au_topP', type=int, default=8, help='Top-P AUs per class (by AU-class text similarity)')
    parser.add_argument('--dump_au_class_sim', action='store_true', default=True, help='Save AU-class similarity CSV')

    
    # -- prompts to try
    # - a_photo_of_a_[CLS]_face 
    # - a_photo_of_the_[CLS]_face 
    # - a_photo_of_one_[CLS]_face 
    # - a_close-up_photo_of_the_[CLS]_face
    # - a_low_resolution_photo_of_a_[CLS]_face
    # - a_good_photo_of_a_[CLS]_face 
    # - a_photo_of_my_[CLS]_face 
    # - a_cropped_photo_of_the_[CLS]_face
    # - a_photo_of_a_person_with_[CLS]_face
    # - a_person_with_an_expression_of_[CLS]
    parser.add_argument('--ctx_init', default='a_person_with_an_expression_of_[CLS]', type=str, help='init tunable prompts')

    # parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--use_landmarks', type=bool, default=False, help="to extract faical landmarks")
    parser.add_argument('--num_landmarks', type=int, default=12)
    parser.add_argument('--visualize_img', default=False, help='visualize_img')

    parser.add_argument('--train_whole_clip_model', type=bool, default=False, help="Train complete Clip model using au PROMPTS")

    parser.add_argument('--train_t_adpt_cl', type=bool, default=True, help="train_txt_adapt_classifer")
    parser.add_argument('--t_adap_epoch', default=10, type=int, help="Text and AU classifier training epochs")
    parser.add_argument('--iter_limit', default=8000, type=int, help='Limit loop')

    parser.add_argument('--load_t_adpt_cl_mod', type=bool, default=False, help="load_t_adpt_cl_mod")
    parser.add_argument('--eval_au_adpt_cl', type=bool, default=False, help="eval_au_adpt_cl")
    parser.add_argument('--eval_au_tar_sb', type=bool, default=False, help="eval_au_tar_sb")
    
    parser.add_argument('--include_cls_prompt', type=bool, default=False, help="include_cls_prompt")

    # =========================================================== Target Subject Video Adaptation
    parser.add_argument('--adapt_tar_sub', type=bool, default=False, help="adapt_tar_sub")
    parser.add_argument('--adapt_per_video', type=bool, default=False, help="adapt_per_video")
    parser.add_argument('--au_prompt_tune', type=bool, default=False, help="au_prompt_tune")
    parser.add_argument('--key_frame_sel', type=bool, default=False, help="key_frame_sel")
    parser.add_argument('--key_frames', type=int, default=16, help="key_frame_sel") # biovid=68
    parser.add_argument('--target_sub_set', type=int, default=20, help="target_sub_set")

    parser.add_argument('--current_ds', type=str, default=config.STRESS)  # BIOVID or STRESS or BAH or FERV39k or DFEW
    parser.add_argument('--pain_db_root_path', type=str, default=config.STRESS_PATH) # BIOVID_PATH or STRESS_PATH or 
    
    # parser.add_argument('--srcs_label_file_name', default='stress_source_sub_labels', type=str, help='Stress Source file name to train AU adapter and classifier')
    # parser.add_argument('--srcs_file_name', default='lab_srcs44_stress_ep20_bs8_sql16_str1_vid', type=str, help='Source file name to train AU adapter and classifier')


    # parser.add_argument('--srcs_label_file_name', default='lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only', type=str, help='Biovid Source file name to train AU adapter and classifier')
    # parser.add_argument('--srcs_file_name', default='lab_srcs77_biovid_ep5_bs512_newcode_tctoi', type=str, help='Source file name to train AU adapter and classifier')
    # parser.add_argument('--srcs_file_name', default='lab_srcs77_biovid_ep10_bs8_sql16_str2_vid', type=str, help='Source file name to train AU adapter and classifier')
    

    parser.add_argument('--srcs_label_file_name', default='bah_source_sub_labels', type=str, help='BAH Source file name to train AU adapter and classifier')
    # parser.add_argument('--srcs_file_name', default='lab_srcs77_biovid_ep5_bs512_newcode_tctoi', type=str, help='Source file name to train AU adapter and classifier')
    parser.add_argument('--srcs_file_name', default='biovid_src_ep10_tar20_bs8_sql16_str2_vid', type=str, help='Source file name to train AU adapter and classifier')
    parser.add_argument('--srcs_label_val_file_name', default='bah_source_sub_val_labels', type=str, help='Source file name to val AU adapter and classifier')
    

    # -- AU+CP srcs_file_name = lab_srcs77_biovid_ep10_bs8_sql16_str2_vid_auNcp_fused0.3_tclipfer
    
    # parser.add_argument('--srcs_file_name', default='rafdb_src_au_adapt_cl', type=str, help='Source file name to train AU adapter and classifier')

    # =========================================================== Audio dictionary
    parser.add_argument('--create_audio_dictionary', action='store_true', default=True,
                        help='Build OpenSMILE audio dictionary from source train_loader and exit (no train/eval)')

    parser.add_argument('--infer_audio_dictionary', action='store_true', default=False,
                        help='Infer OpenSMILE audio dictionary from source train_loader')

    parser.add_argument('--is_opensmile_dict', action='store_true', default=True,
                        help='Activate OpenSMILE audio dictionary')

    parser.add_argument('--audio_dict_dir', type=str, default="outputs/opensmile_activ_dbscan_feat_dict_bah", 
                        help="Audio dictionary directory")

    # =========================================================== Video
    parser.add_argument('--is_video_clip', type=bool, default=True, help="Train Clip on videos")
    parser.add_argument('--seq_len', default=16, type=int, help="Video / Sequence Length")
    parser.add_argument('--frame_stride', default=1, type=int, help="Frame stride")

    # ===========================================================
    parser.add_argument('--target_evaluation_only', type=bool, default=False)
    parser.add_argument('--top_timestamp', type=str, default='1754805485')

     # =========================================================== NEw FOr MM-VLM
    parser.add_argument('--fus_type', type=int, default=None, help="Fusion concatenation")
    # parser.add_argument('--is_fus_crossatten', type=bool, default=False, help="Fusion Cross-Attention")
    parser.add_argument('--is_mod_align', type=bool, default=False, help="Modality alignment")



    args = parser.parse_args()

    # base_audio_dict_dir = args.audio_dict_dir

    # min_cluster_size_values = [3, 5, 7, 10]
    # min_samples_values = [1, 2, 3, 5]

    # for min_cluster_size in min_cluster_size_values:
    #     for min_samples in min_samples_values:

    #         if min_samples > min_cluster_size:
    #             continue

    #         audio_dict_dir = os.path.join(
    #             base_audio_dict_dir,
    #             f"mcs_{min_cluster_size}_ms_{min_samples}",
    #         )

    #         if not os.path.exists(audio_dict_dir):
    #             print(f"Skipping missing directory: {audio_dict_dir}")
    #             continue

    #         run_args = copy.deepcopy(args)
    #         run_args.audio_dict_dir = audio_dict_dir

    #         print("=" * 80)
    #         print(f"Running with dictionary:")
    #         print(f"min_cluster_size = {min_cluster_size}")
    #         print(f"min_samples      = {min_samples}")
    #         print(f"audio_dict_dir   = {run_args.audio_dict_dir}")
    #         print("=" * 80)

    #         main(run_args)
    main(args)