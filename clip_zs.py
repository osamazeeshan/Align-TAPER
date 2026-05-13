import argparse
import os
import time
import urllib.request

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import clip
import config

from PIL import Image
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# CLIP model URLs
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

from data.imagnet_prompts import imagenet_classes
from data.biovid_prompts import biosub_classes
from data.datautils import build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from sklearn.metrics import accuracy_score, f1_score, recall_score


def load_clip_model(model_name, device):
    """Load CLIP model from specific URL"""
    if model_name not in _MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(_MODELS.keys())}")
    
    model_url = _MODELS[model_name]
    
    # Create cache directory
    cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download model if not cached
    model_path = os.path.join(cache_dir, f"{model_name.replace('/', '-')}.pt")
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded to {model_path}")
    
    # Load model
    model = torch.jit.load(model_path, map_location=device).eval()
    
    # Get preprocessing
    if "RN" in model_name:
        resolution = 224
    elif "ViT-L" in model_name:
        resolution = 224
    else:  # ViT-B models
        resolution = 224
    
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    preprocess = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize,
    ])
    
    return model, preprocess


# def create_text_prompts(classnames, template="a photo of a {}."):
#     """Create text prompts for zero-shot classification"""
#     prompts = []
#     for classname in classnames:
#         prompts.append(template.format(classname))
#     return prompts

def create_text_prompts(classnames, template="a photo of a {}."):
    """Create text prompts for zero-shot classification"""
    prompts = []
    for classname in classnames:
        if "[CLS]" in template:
            # Handle [CLS] token format and convert underscores to spaces
            prompt = template.replace("[CLS]", classname)
            prompt = prompt.replace("_", " ")
        else:
            # Handle standard {} format
            prompt = template.format(classname)
        prompts.append(prompt)
    print("===> Prompt: ", prompts)
    return prompts


def zero_shot_classify(model, images, text_features):
    """Perform zero-shot classification using CLIP"""
    with torch.no_grad():
        # Extract image features using the model's encode_image method
        # For torchscript models, we need to use the forward method differently
        image_features = model.encode_image(images)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    return similarities


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    file_name = 'logs/zero_shot_clip.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    with open(file_name, "w") as f:  
        f.write("=> Zero-shot CLIP evaluation\n")
        f.write("=> Model: {}\n".format(args.arch))

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args, file_name)


def main_worker(gpu, args, file_name):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for evaluation".format(args.gpu))

    # Load CLIP model using our custom loader
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocess = load_clip_model(args.arch, device)
        print("=> CLIP model loaded: {}".format(args.arch))
    except Exception as e:
        print(f"Failed to load custom model, falling back to clip.load(): {e}")
        model, preprocess = clip.load(args.arch, device=device)
        print("=> CLIP model loaded via clip.load(): {}".format(args.arch))
    
    model.eval()
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpu)

    cudnn.benchmark = True

    # Use custom preprocessing or fall back to the original
    if args.use_custom_preprocess:
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        
        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        data_transform = preprocess
    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    
    for set_id in datasets:
        print("evaluating: {}".format(set_id))
        
        # Handle bio datasets
        sub_id = None
        if 'bio' in set_id:
            sub_id = set_id[-1]
            set_id = set_id[:-1] 

        # Get appropriate classnames
        if args.test_sets in fewshot_datasets:
            classnames = eval("{}_classes".format(args.test_sets.lower()))
        elif len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = biosub_classes  # or imagenet_classes depending on your setup
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

        # Create text prompts and encode them
        if args.prompt_template:
            prompts = create_text_prompts(classnames, args.prompt_template)
        else:
            prompts = create_text_prompts(classnames, "a photo of a {}.")
        
        # Tokenize and encode text
        try:
            text_inputs = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        except AttributeError:
            # Handle torchscript models that might have different methods
            print("Using alternative text encoding method for torchscript model")
            text_tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        with open(file_name, "a") as f:  
            f.write("\nDataset: {}\n".format(set_id))
            f.write("Number of classes: {}\n".format(len(classnames)))

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, sub_id=sub_id)
        print("number of test samples: {}".format(len(val_dataset)))
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
            
        results[set_id] = evaluate_zero_shot(val_loader, model, text_features, args)
        
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

    print("\n======== Result Summary ========")
    print("Zero-shot CLIP evaluation results:")
    print("\t[Dataset]\tWAR.\tUAR\tF1(macro)")

    for id in results.keys():
        r = results[id]
        print(f"{id}\t{r['war']:.2f}\t{r['uar']:.2f}\t{r['f1_macro']:.2f}")


def evaluate_zero_shot(val_loader, model, text_features, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    end = time.time()

    all_preds = []
    all_targets = []

    for i, (images, target) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if len(images.size()) > 4:
            # when using ImageNet Sampler as the dataset
            assert images.size(0) == 1
            images = images.squeeze(0)

        # Zero-shot classification
        output = zero_shot_classify(model, images, text_features)  # [B,C]

        # Convert similarities to logits for accuracy calculation
        logits = torch.log(output + 1e-8)  # Add small epsilon to avoid log(0)

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, min(5, logits.size(1))))
        top1.update(acc1[0], images.size(0))
        if logits.size(1) >= 5:
            top5.update(acc5[0], images.size(0))

        # store predictions & labels for F1/UAR
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    # compute F1 and UAR
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0
    uar = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0

    # print(f"F1 (macro): {f1_macro:.4f}  UAR: {uar:.4f}")

    return {
        'war': top1.avg,
        'uar': uar,
        'f1_macro': f1_macro,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot CLIP Evaluation')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root', 
                        default=config.BIOVID_SOURCE_DATASET_PATH)
    
    parser.add_argument('--test_sets', type=str, 
                        default='biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9', 
                        help='test dataset (multiple datasets split by slash)')

    parser.add_argument('--dataset_mode', type=str, default='test', 
                        help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/32',
                        help='CLIP model architecture: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--prompt_template', default='a_person_with_an_expression_of_[CLS]', type=str, 
                        help='prompt template for zero-shot classification (e.g., "a photo of a {}.")')
    parser.add_argument('--use_custom_preprocess', action='store_true', default=False,
                        help='use custom preprocessing instead of CLIP default')
    parser.add_argument('--seed', type=int, default=0)

    main()