import os

import string
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import pandas as pd
from collections import defaultdict
from moviepy.editor import VideoFileClip

from PIL import Image
import torchaudio

import torchvision.transforms as T

import glob

import config
import re
import torch.nn.functional as F


class BSDS500(Dataset):

    def __init__(self):

        image_folder = config.CURRENT_DIR + config.DATASET_FOLDER + '/BSDS300/images/train'

        # self.image_files = img_files
        # temp = list(map(3, '/home/osamazeeshan/Downloads/PhD/FER/code/domain-adaptation-playground/data/BSDS300/images/train/187071.jpg'))
        # print(temp)
        # self.image_files = list(glob.glob('/*.jpg'))

        # get every image in the folder ending with .jpg and add to the image list
        self.image_files = glob.glob(image_folder + '/*.jpg')

    def __getitem__(self, i):
        image = cv2.imread(self.image_files[i], cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor

    def __len__(self):
        return len(self.image_files)


class MNISTM(Dataset):

    def __init__(self, train=True):
        super(MNISTM, self).__init__()
        self.mnist = datasets.MNIST(config.CURRENT_DIR + config.DATASET_FOLDER + '/mnist', train=train, download=True)
        self.bsds = BSDS500()

        # Fix RNG so the same images are used for blending
        self.rng = np.random.RandomState(42)

    def __getitem__(self, i):
        digit, label = self.mnist[i]
        digit = transforms.ToTensor()(digit)
        bsds_image = self._random_bsds_image()
        patch = self._random_patch(bsds_image)
        patch = patch.float() / 255
        blend = torch.abs(patch - digit)

        # print(blend)
        # print(label)
        return blend, label

    def _random_patch(self, image, size=(28, 28)):
        _, im_height, im_width = image.shape
        x = self.rng.randint(0, im_width-size[1])
        y = self.rng.randint(0, im_height-size[0])
        return image[:, y:y+size[0], x:x+size[1]]

    def _random_bsds_image(self):
        i = self.rng.choice(len(self.bsds))
        return self.bsds[i]

    def __len__(self):
        return len(self.mnist)   


class FerDatasets(Dataset):

    def __init__(self, imgs, labels, flag = 0):
        super(FerDatasets, self).__init__()

        self.img = imgs
        self.label = labels
        self.flag = flag

    def __getitem__(self, i):
        # img = read_image(self.img[i])

        # img = transforms.Resize(100)(img)

        img = cv2.imread(self.img[i], cv2.IMREAD_COLOR)
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[1]

        # print(img)
        # print(label)

        # digit, label = self.mnist[i]
        # digit = transforms.ToTensor()(digit)
        # bsds_image = self._random_bsds_image()
        # patch = self._random_patch(bsds_image)
        # patch = patch.float() / 255
        # blend = torch.abs(patch - digit)
        return tensor.float(), label

    def __len__(self):
        return len(self.img)


class PainDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(PainDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ") # for BAH sep="," else sep=" "
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()

    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])

        '''
        read_image is not working:
        UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory 
        warn(f"Failed to load image Python extension: {e}")
        '''
        # image = read_image(img_path)
        # image = self.convtensor(Image.open(img_path))
        # image = transforms.RandomResizedCrop(100)(image)

        image = (Image.open(img_path))

        label = self.img_labels.iloc[i, 1]

        # # display image
        # img = T.ToPILImage()(image)
        # img.show()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        # print(img)
        # print(label)

        # digit, label = self.mnist[i]
        # digit = transforms.ToTensor()(digit)
        # bsds_image = self._random_bsds_image()
        # patch = self._random_patch(bsds_image)
        # patch = patch.float() / 255
        # blend = torch.abs(patch - digit)


    def __len__(self):
        return len(self.img_labels)


class PainVideoDatasets(Dataset):
    """
    Video-level dataset where each sample corresponds to one video (a sequence of frames)
    belonging to a subject, with its class label.
    """

    def __init__(self, img_dir, label_path, transform=None, seq_len=None, target_transform=None):
        super(PainVideoDatasets, self).__init__()

        # Read label file: expected format "<subject>/<video> <label>"
        self.img_labels = pd.read_csv(label_path, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len  # number of frames to sample per video

        # Read label file (frame paths + labels)
        df = pd.read_csv(label_path, sep=" ", header=None)
        df.columns = ["frame_path", "label"]

        # Group frames by video (parent directory of frame)
        df["video_dir"] = df["frame_path"].apply(lambda p: os.path.dirname(p))
        grouped = df.groupby("video_dir")

        self.samples = []
        for video_dir, group in grouped:
            frames = group["frame_path"].tolist()
            label = int(group["label"].iloc[0])
            self.samples.append((video_dir, frames, label))

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        # Resolve absolute frame paths
        frame_paths = [os.path.join(self.img_dir, fp) for fp in frame_paths]

        # Sort frames by name to preserve temporal order
        frame_paths = sorted(frame_paths)

        # Sample or pad frames
        if self.seq_len is not None:
            if len(frame_paths) > self.seq_len:
                idxs = torch.linspace(0, len(frame_paths) - 1, self.seq_len).long()
                frame_paths = [frame_paths[i] for i in idxs]
            elif len(frame_paths) < self.seq_len:
                frame_paths += [frame_paths[-1]] * (self.seq_len - len(frame_paths))

        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        # Extract subject ID from path (first directory)
        subject_id = video_dir.split("/")[0]

         # 🔍 Sanity check for first few samples
        # if idx < 3:  # only print for first few to avoid spam
        #     print(f"\n[DEBUG] Sample {idx}:")
        #     print(f"  video_dir: {video_dir}")
        #     print(f"  num frames loaded: {len(frame_paths)}")
        #     print(f"  frames tensor shape: {frames.shape}")  # expect [T, C, H, W]
        #     print(f"  label: {label}")
        #     print(f"  subject_id: {subject_id}")

        #     # Optional: check temporal consistency (mean brightness over time)
        #     if isinstance(frames, torch.Tensor):
        #         print(f"  frame mean[0]: {frames[0].mean():.4f}, frame mean[-1]: {frames[-1].mean():.4f}")

        return frames, label

    def __len__(self):
        return len(self.samples)


class PainSeqDatasets(Dataset):
    """
    Handles label files that list *frame-level paths* instead of video dirs.
    Groups frames by video, then splits each into fixed-length clips.
    """

    def __init__(self, img_dir, label_path, transform=None, seq_len=24, target_transform=None):
        super(PainSeqDatasets, self).__init__()

        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(label_path, sep=" ")

        # Group frames by video
        video_groups = defaultdict(list)
        for _, row in df.iterrows():
            frame_rel_path, label = row.iloc[0], int(row.iloc[1])
            video_dir = os.path.dirname(frame_rel_path)  # e.g., S01/V001
            video_groups[video_dir].append((frame_rel_path, label))

        self.samples = []

        # Process each video group
        for video_dir, frames_info in video_groups.items():
            frame_paths = [os.path.join(self.img_dir, f) for f, _ in sorted(frames_info)]
            label = frames_info[0][1]  # same label for all frames in video

            n_frames = len(frame_paths)
            if n_frames > seq_len:
                num_clips = n_frames // seq_len
                for i in range(num_clips):
                    start = i * seq_len
                    end = start + seq_len
                    clip_frames = frame_paths[start:end]
                    self.samples.append((video_dir, clip_frames, label))

                # leftover frames
                if n_frames % seq_len != 0:
                    leftover = frame_paths[-seq_len:]
                    self.samples.append((video_dir, leftover, label))
            else:
                # pad if needed
                clip_frames = frame_paths.copy()
                if n_frames < seq_len:
                    clip_frames += [clip_frames[-1]] * (seq_len - n_frames)
                self.samples.append((video_dir, clip_frames, label))

        print(f"[INFO] Grouped {len(video_groups)} videos into {len(self.samples)} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        return frames, label



class TemporalPainDataset(Dataset):
    def __init__(self, img_dir, label_path, transform=None, seq_len=24, temporal_transform=None, target_transform=None, fps=5, audio_sr=16000):
        super().__init__()

        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.temporal_transform = temporal_transform  # NEW
        self.target_transform = target_transform
        self.fps = fps
        self.audio_sr = audio_sr
        df = pd.read_csv(label_path, sep=" ")

        # Group frames by video
        video_groups = defaultdict(list)
        for _, row in df.iterrows():
            frame_rel_path, label = row.iloc[0], int(row.iloc[1])
            video_dir = os.path.dirname(frame_rel_path)
            video_groups[video_dir].append((frame_rel_path, label))

        self.samples = []
        for video_dir, frames_info in video_groups.items():
            frame_paths = [os.path.join(self.img_dir, f) for f, _ in sorted(frames_info)]
            label = frames_info[0][1]
            self.samples.append((video_dir, frame_paths, label))

        print(f"[INFO] Loaded {len(self.samples)} videos with temporal transforms enabled.")

    def get_frame_index(self, frame_path):
        frame_name = os.path.basename(frame_path).split('.')[0]
        return int(frame_name.split('_')[1])

    def compute_audio_window(frame_indices, fps, seq_len):
        times = np.array(frame_indices) / fps
        start_time = times[0]
        # Compute temporal stride (robust to downsampling)
        if len(times) > 1:
            diffs = np.diff(times)
            stride = np.median(diffs)
        else:
            stride = 1.0 / fps
        # Estimate duration based on stride
        duration = stride * seq_len
        end_time = start_time + duration
        return start_time, end_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, frame_paths, label = self.samples[idx]

        # Apply temporal transforms BEFORE loading frames
        if self.temporal_transform:
            frame_paths, start = self.temporal_transform(frame_paths)

        # Load frames
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        # frame_indices = [self.get_frame_index(p) for p in frame_paths]
        # times = np.array(frame_indices) / self.fps
        # find positions of your selected window
        end_idx   = start + self.seq_len - 1
        # convert to time
        start_time = start
        # start_time = start / self.fps
        end_time   = end_idx

        folder, subfolder = frame_paths[0].split('/')[7:9]
        audio_path = os.path.join(self.img_dir, 'audio_all', folder, f'{subfolder}.wav')
        # --------------------------------------------------
        # Compute target duration (important for consistency)
        # --------------------------------------------------
        target_duration = max(0.0, end_time - start_time)

        # --------------------------------------------------
        # ✅ Case 1: Load from WAV (FAST PATH)
        # --------------------------------------------------
        if os.path.exists(audio_path):
            waveform, file_sr = torchaudio.load(audio_path)  # [C, T]
            # Resample if needed
            if file_sr != self.audio_sr:
                resampler = torchaudio.transforms.Resample(file_sr, self.audio_sr)
                waveform = resampler(waveform)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            total_len = waveform.shape[1]
            total_duration = total_len / self.audio_sr

            # 🔒 Clamp time using WAV duration
            start_time_clamped = max(0.0, min(start_time, total_duration - 1e-3))
            end_time_clamped   = max(0.0, min(end_time,   total_duration - 1e-3))
            
            actual_window = end_time_clamped - start_time_clamped
            if actual_window < target_duration:
                start_proxy = target_duration - actual_window
                start_time_clamped = max(0.0, min(start_time - start_proxy, total_duration - 1e-3))

            if end_time_clamped <= start_time_clamped:
                end_time_clamped = min(start_time_clamped + 1.0, total_duration)

            # Convert to samples
            start_sample = int(start_time_clamped * self.audio_sr)
            end_sample   = int(end_time_clamped   * self.audio_sr)

            audio_chunk = waveform[:, start_sample:end_sample]

            # Handle empty
            if audio_chunk.shape[1] == 0:
                audio_chunk = torch.zeros(1, int(self.audio_sr * target_duration))

            audio_array = audio_chunk.squeeze(0).numpy()
        # --------------------------------------------------
        # ❗ Case 2: Fallback → Extract from VIDEO
        # --------------------------------------------------
        else:
            video_path = os.path.join(self.img_dir, 'Videos', folder, f'{subfolder}.mp4')
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
            
                # 🔒 Clamp time using VIDEO duration
                start_time_clamped = max(0.0, min(start_time, duration - 1e-3))
                end_time_clamped   = max(0.0, min(end_time,   duration - 1e-3))
                
                # actual_window = end_time_clamped - start_time_clamped
                # if actual_window < target_duration:
                #     start_proxy = target_duration - actual_window 
                #     start_time_clamped = max(0.0, min(start_time - start_proxy, duration - 1e-3))

                if end_time_clamped <= start_time_clamped:
                    end_time_clamped = min(start_time_clamped + 1.0, duration)
                if clip.audio is None:
                    audio_array = np.zeros(
                        int(self.audio_sr * target_duration),
                        dtype=np.float32
                    )
                else:
                    try:
                        audio = clip.audio.subclip(start_time_clamped, end_time_clamped)
                        chunks = list(audio.iter_chunks(
                            fps=self.audio_sr,
                            chunksize=2048
                        ))

                        if len(chunks) == 0:
                            audio_array = np.zeros(
                                int(self.audio_sr * target_duration),
                                dtype=np.float32
                            )
                        else:
                            audio_array = np.concatenate(chunks, axis=0)
                    except Exception as e:
                        print(f"[WARN] Audio extraction failed: {e}")
                        audio_array = np.zeros(
                            int(self.audio_sr * target_duration),
                            dtype=np.float32
                        )
            except Exception as e:
                print(f"[WARN] Audio extraction failed for {video_path}: {e}")
                audio_array = np.zeros(
                        int(self.audio_sr * target_duration),
                        dtype=np.float32
                    )
            # Convert to mono
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

        # --------------------------------------------------
        # 🔧 Final normalization (COMMON)
        # --------------------------------------------------
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)

        audio_array = audio_array.astype(np.float32)

        if audio_array.shape[-1] < 100:
            print(f"Audio shape {audio.shape} is too short!")

        if audio_array.shape[-1] != 624000:     # for 16-frames= 240000; 20-frames= 304000; 30-frames= 464000; 40-frames= 624000; 
            print("ERROR")

        return frames, audio_array, label, video_dir



class TemporalBAHDataset(Dataset):
    def __init__(
        self,
        img_root,
        label_csv,
        transform=None,
        seq_len=16,
        temporal_transform=None,
        target_transform=None,
        fixed_window=500,
        fps=24,
        audio_sr=16000,
        audio_root=None,
        frame_index_base=0,
    ):
        """
        Args:
            img_root:
                Root folder containing extracted frames.

            label_csv:
                CSV file containing:
                [source_list, video, segment_id, label,
                 num_frames, start_frame, end_frame]

            transform:
                Image transform applied to each frame.

            seq_len:
                Model temporal sequence length.

            fixed_window:
                Fixed number of frames per sub-sequence.

            temporal_transform:
                Optional temporal transform on list of frame paths.

            target_transform:
                Optional transform on the label.

            fps:
                Frame rate used to map frame indices to audio time.

            audio_sr:
                Target audio sample rate.

            audio_root:
                Root folder for wav files. If None, the loader checks:
                    img_root/wav
                    dirname(img_root)/wav
                    wav

            frame_index_base:
                Use 0 if frame-0.jpg corresponds to time 0.
                Use 1 if frame-1.jpg corresponds to time 0.
        """
        super().__init__()

        self.img_root = img_root
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.seq_len = seq_len
        self.fixed_window = fixed_window
        self.fps = fps
        self.audio_sr = audio_sr
        self.audio_root = audio_root
        self.frame_index_base = frame_index_base

        df = pd.read_csv(label_csv)
        self.samples = []

        for _, row in df.iterrows():
            video_rel_path = row["video"]
            label = int(row["label"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            segment_id = int(row["segment_id"])

            video_frame_dir = os.path.join(
                img_root,
                video_rel_path,
            )

            frame_paths = []
            for frame_idx in range(start_frame, end_frame + 1):
                frame_file = f"frame-{frame_idx}.jpg"
                frame_path = os.path.join(
                    video_frame_dir,
                    frame_file,
                )

                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                else:
                    print(f"[WARN] Missing frame: {frame_path}")

            num_frames = len(frame_paths)

            if num_frames == 0:
                continue

            if num_frames <= fixed_window:
                self.samples.append({
                    "video": video_rel_path,
                    "segment_id": segment_id,
                    "frame_paths": frame_paths,
                    "label": label,
                })
            else:
                start = 0

                while start < num_frames:
                    end = start + fixed_window
                    subclip = frame_paths[start:end]

                    if len(subclip) < seq_len:
                        borrow = seq_len - len(subclip)

                        if start - borrow >= 0:
                            subclip = (
                                frame_paths[start - borrow:start]
                                + subclip
                            )
                        else:
                            subclip += [subclip[-1]] * borrow

                    self.samples.append({
                        "video": video_rel_path,
                        "segment_id": segment_id,
                        "frame_paths": subclip,
                        "label": label,
                    })

                    start += fixed_window

        print(
            f"[INFO] Loaded {len(self.samples)} fixed-window subsequences "
            f"(window={fixed_window}, seq_len={seq_len}) from {label_csv}"
        )

    def __len__(self):
        return len(self.samples)

    def get_frame_index(self, frame_path):
        """
        Extract frame index from names such as:
            frame-123.jpg
            frame_123.jpg
            frame123.jpg
        """
        frame_name = os.path.basename(frame_path)
        match = re.search(r"(\d+)", frame_name)

        if match is None:
            raise ValueError(
                f"Could not extract frame index from {frame_path}"
            )

        return int(match.group(1))

    def build_audio_path(self, video_rel_path):
        """
        Example:
            video_rel_path:
                Videos/82553/Visite_1/
                82553_Question_1_2024-08-22_12-11-55_Video.mp4

            audio path:
                wav/Videos/82553/Visite_1/
                82553_Question_1_2024-08-22_12-11-55_Video.mp4/
                82553_Question_1_2024-08-22_12-11-55_Video.wav
        """
        video_name = os.path.basename(video_rel_path)

        if video_name.endswith(".mp4"):
            wav_name = video_name[:-4] + ".wav"
        else:
            wav_name = video_name + ".wav"

        candidates = []

        if self.audio_root is not None:
            candidates.append(
                os.path.join(
                    self.audio_root,
                    video_rel_path,
                    wav_name,
                )
            )

        candidates.extend([
            os.path.join(
                self.img_root,
                "wav",
                video_rel_path,
                wav_name,
            ),
            os.path.join(
                os.path.dirname(self.img_root),
                "wav",
                video_rel_path,
                wav_name,
            ),
            os.path.join(
                "wav",
                video_rel_path,
                wav_name,
            ),
        ])

        for path in candidates:
            if os.path.exists(path):
                return path

        # Return the first candidate for warning/debugging.
        return candidates[0]

    def compute_audio_window_from_frames(self, frame_paths):
        """
        Convert selected visual frame indices into the corresponding
        audio time window.
        """
        frame_indices = [
            self.get_frame_index(p)
            for p in frame_paths
        ]

        start_frame = min(frame_indices)
        end_frame = max(frame_indices)

        start_time = (
            start_frame - self.frame_index_base
        ) / self.fps

        end_time = (
            end_frame - self.frame_index_base + 1
        ) / self.fps

        start_time = max(0.0, start_time)
        end_time = max(
            start_time + 1.0 / self.fps,
            end_time,
        )

        return start_time, end_time

    def load_audio_segment(
        self,
        audio_path,
        start_time,
        end_time,
    ):
        """
        Load the exact audio segment corresponding to the selected
        visual frame window.
        """
        target_duration = max(
            1.0 / self.fps,
            end_time - start_time,
        )

        target_num_samples = max(
            1,
            int(round(target_duration * self.audio_sr)),
        )

        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio file: {audio_path}")
            return np.zeros(
                target_num_samples,
                dtype=np.float32,
            )

        try:
            waveform, file_sr = torchaudio.load(audio_path)  # [C, S]

            if file_sr != self.audio_sr:
                resampler = torchaudio.transforms.Resample(
                    file_sr,
                    self.audio_sr,
                )
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(
                    dim=0,
                    keepdim=True,
                )

            total_samples = waveform.shape[1]
            total_duration = total_samples / self.audio_sr

            start_time = max(
                0.0,
                min(start_time, total_duration),
            )

            end_time = max(
                start_time,
                min(end_time, total_duration),
            )

            start_sample = int(round(start_time * self.audio_sr))
            end_sample = int(round(end_time * self.audio_sr))

            audio_chunk = waveform[
                :,
                start_sample:end_sample,
            ]

            if audio_chunk.shape[1] < target_num_samples:
                pad_len = target_num_samples - audio_chunk.shape[1]
                audio_chunk = F.pad(
                    audio_chunk,
                    (0, pad_len),
                )

            elif audio_chunk.shape[1] > target_num_samples:
                audio_chunk = audio_chunk[
                    :,
                    :target_num_samples,
                ]

            audio_array = audio_chunk.squeeze(0).numpy()

        except Exception as e:
            print(
                f"[WARN] Audio extraction failed for {audio_path}: {e}"
            )
            audio_array = np.zeros(
                target_num_samples,
                dtype=np.float32,
            )

        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / (
                np.max(np.abs(audio_array)) + 1e-8
            )

        audio_array = audio_array.astype(np.float32)

        return audio_array

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frame_paths = sample["frame_paths"]
        label = sample["label"]
        video_path = sample["video"]

        # --------------------------------------------------
        # Temporal transform
        # --------------------------------------------------
        if self.temporal_transform:
            temporal_output = self.temporal_transform(frame_paths)

            if isinstance(temporal_output, tuple):
                frame_paths = temporal_output[0]
            else:
                frame_paths = temporal_output

        # --------------------------------------------------
        # Subsample or pad frames
        # --------------------------------------------------
        if len(frame_paths) > self.seq_len:
            indices = torch.linspace(
                0,
                len(frame_paths) - 1,
                self.seq_len,
            ).long()

            frame_paths = [
                frame_paths[i]
                for i in indices
            ]

        elif len(frame_paths) < self.seq_len:
            frame_paths += [
                frame_paths[-1]
            ] * (self.seq_len - len(frame_paths))

        # --------------------------------------------------
        # Compute synchronized audio window
        # --------------------------------------------------
        start_time, end_time = (
            self.compute_audio_window_from_frames(frame_paths)
        )

        audio_path = self.build_audio_path(video_path)

        audio_array = self.load_audio_segment(
            audio_path=audio_path,
            start_time=start_time,
            end_time=end_time,
        )

        # --------------------------------------------------
        # Load frames
        # --------------------------------------------------
        frames = []

        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        frames = torch.stack(
            frames,
            dim=0,
        )  # [T, C, H, W]

        if self.target_transform:
            label = self.target_transform(label)

        return frames, audio_array, label, video_path


# class TemporalBAHDataset(Dataset):
#     def __init__(self, img_root, label_csv, transform=None, seq_len=16,
#              temporal_transform=None, target_transform=None,
#              fixed_window=500, fps=24, audio_sr=16000):
#         """
#         Args:
#             img_root (str): Root folder containing extracted frames for videos.
#             label_csv (str): CSV file containing:
#                 [source_list, video, segment_id, label, num_frames, start_frame, end_frame]
#             transform (callable): Image transform applied to each frame.
#             seq_len (int): Model temporal sequence length (used only for padding last chunk).
#             fixed_window (int): Fixed number of frames per sub-sequence (e.g. 75).
#             temporal_transform (callable): Optional temporal transform on list of frame paths.
#             target_transform (callable): Optional transform on the label.
#         """
#         super().__init__()

#         self.img_root = img_root
#         self.transform = transform
#         self.temporal_transform = temporal_transform
#         self.target_transform = target_transform
#         self.seq_len = seq_len
#         self.fixed_window = fixed_window

#         # --- Read CSV ---
#         df = pd.read_csv(label_csv)
#         self.samples = []

#         # --- Iterate through each segment ---
#         for _, row in df.iterrows():
#             video_rel_path = row["video"]
#             label = int(row["label"])
#             start_frame = int(row["start_frame"])
#             end_frame = int(row["end_frame"])
#             segment_id = int(row["segment_id"])

#             # Frame directory
#             video_frame_dir = os.path.join(img_root, video_rel_path)

#             # Collect frame paths
#             frame_paths = []
#             for frame_idx in range(start_frame, end_frame + 1):
#                 frame_file = f"frame-{frame_idx}.jpg"
#                 frame_path = os.path.join(video_frame_dir, frame_file)
#                 if os.path.exists(frame_path):
#                     frame_paths.append(frame_path)
#                 else:
#                     print(f"[WARN] Missing: {frame_path}")

#             num_frames = len(frame_paths)
#             if num_frames == 0:
#                 continue

#             # --- Split logic ---
#             if num_frames <= fixed_window:
#                 # short segment, keep as is
#                 self.samples.append({
#                     "video": video_rel_path,
#                     "segment_id": segment_id,
#                     "frame_paths": frame_paths,
#                     "label": label
#                 })
#             else:
#                 # Split into 75-frame chunks
#                 start = 0
#                 while start < num_frames:
#                     end = start + fixed_window
#                     subclip = frame_paths[start:end]

#                     # If last chunk smaller than seq_len → pad or borrow
#                     if len(subclip) < seq_len:
#                         # Borrow some frames from end of previous chunk if possible
#                         borrow = seq_len - len(subclip)
#                         if start - borrow >= 0:
#                             subclip = frame_paths[start - borrow:start] + subclip
#                         else:
#                             # fallback: pad by repeating last frame
#                             subclip += [subclip[-1]] * borrow

#                     self.samples.append({
#                         "video": video_rel_path,
#                         "segment_id": segment_id,
#                         "frame_paths": subclip,
#                         "label": label
#                     })

#                     # Move window forward
#                     start += fixed_window

#         print(f"[INFO] Loaded {len(self.samples)} fixed-window subsequences "
#             f"(window={fixed_window}, seq_len={seq_len}) from {label_csv}")


#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         frame_paths = sample["frame_paths"]
#         label = sample["label"]
#         video_path = sample["video"]

#         # --- Temporal transform ---
#         if self.temporal_transform:
#             frame_paths = self.temporal_transform(frame_paths)

#         # --- Subsample or pad frames ---
#         if len(frame_paths) > self.seq_len:
#             # Uniform sampling across segment
#             indices = torch.linspace(0, len(frame_paths) - 1, self.seq_len).long()
#             frame_paths = [frame_paths[i] for i in indices]
#         elif len(frame_paths) < self.seq_len:
#             # Repeat last frame to fill sequence
#             frame_paths += [frame_paths[-1]] * (self.seq_len - len(frame_paths))

#         # --- Load frames ---
#         frames = []
#         for frame_path in frame_paths:
#             img = Image.open(frame_path).convert("RGB")
#             if self.transform:
#                 img = self.transform(img)
#             frames.append(img)

#         frames = torch.stack(frames, dim=0)  # [T, C, H, W]

#         if self.target_transform:
#             label = self.target_transform(label)

#         return frames, label, video_path




class BAHDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(BAHDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, delimiter=',')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()
        self.pil_image = T.ToPILImage()
        # self.only_bio_mod = only_bio_mod

    def __getitem__(self, i):
        
        ''' for N-classifier Physio'''
        # while self.img_labels.iloc[i, 1] == -1:
        #     i = (i + 1) % len(self.img_labels)  # Wrap around if needed
        # label_id = self.id_labels.iloc[i, 1]
        # bio_path = os.path.join(self.img_dir,self.img_labels.iloc[i, 0])
        '''END'''
        
        # bio_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 2])
        # phy_signals = pd.read_csv(bio_path, sep='\t')
        # signal = phy_signals["gsr"] #Only use the EDA modality
        # signal = torch.tensor(signal.values)

        label = self.img_labels.iloc[i, 1]

        if self.target_transform:
            label = self.target_transform(label)

        try:
            # if 'RAF-DB_AffectNet_AFF-WILD2' in self.img_dir:
            #     img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0].split('/',1)[-1])
            # else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
            image = self.convtensor(Image.open(img_path))
            if self.transform:
                image = self.transform(self.pil_image(image))
        except:
            print(f"Skipping invalid image at index {img_path}")
            i = (i + 1) % len(self)
            image = torch.full((1, 3, 100, 100), float('nan'))
        
        # return image, label, img_path
        
        return image, label

    def __len__(self):
        return len(self.img_labels)
    
class FerImageFolder(Dataset):

    def __init__(self, imgs, labels, transform = None):
        super(FerImageFolder, self).__init__()

        self.img = imgs
        self.label = labels
        self.transform = transform

    def __getitem__(self, index):
        # img = read_image(self.img[i])

        # img = transforms.Resize(100)(img)

        # image = read_image(self.img[index])
        # image = transforms.RandomResizedCrop(100)(image)

        # if self.transform:
        # image = self.transform(image)

        img = cv2.imread(self.img[index], cv2.IMREAD_COLOR)
        self.transform = self.transform
        img1 = cv2.resize(img, (100, 100))
        tensor = torch.from_numpy(img1.transpose(2, 0, 1))

        label = self.label[index]

        return tensor.float(), label, index

    def __len__(self):
        return len(self.img)


# Define a custom iterator that restarts from the beginning
class TragetRestartableIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)