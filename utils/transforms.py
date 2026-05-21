from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import os


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class TemporalDownSample(object):
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        idx = [(i % self.factor) == 0 for i in range(clip.shape[0])]
        return clip[idx]


class RandomRoll(object):
    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, seq):
        if isinstance(seq, list):
            seq = np.asarray(seq)
        start_idx = np.random.randint(0, seq.shape[0])
        return np.concatenate([seq[start_idx:], seq[:start_idx]])


class RandomSequence_old(object): # for old version
    def __init__(self, seq_size, on_load=False, wrap=True):
        self.seq_size = seq_size
        self.on_load = on_load
        self.wrap = wrap

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        if self.on_load:
            return self.call_on_load(clip)
        else:
            return self.call_on_video(clip)

    def call_on_load(self, clip):
        T = len(clip)
        rnd_start = torch.randint(T, (1,)).item()
        end_idx = rnd_start + self.seq_size

        if end_idx < T:
            new_clip = clip[rnd_start:end_idx]
        elif self.wrap:
            end_idx -= T
            new_clip = np.concatenate((clip[rnd_start:], clip[:end_idx]))
        else:
            new_clip = clip[rnd_start:]

        # Safe padding for string arrays (frame paths)
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            new_clip = list(new_clip) + [new_clip[-1]] * pad

        return new_clip


class RandomSequence_1(object):
    def __init__(self, seq_size, on_load=False):
        self.seq_size = seq_size
        self.on_load  = on_load

    def __call__(self, clip):
        if isinstance(clip, list):
            clip = np.asarray(clip)
        if self.on_load:
            return self.call_on_load(clip)
        else:
            return self.call_on_video(clip)

    # ── helpers for path-based (on_load) mode ─────────────────────────────

    def get_frame_idx(self, path):
        name = os.path.basename(path)
        return int(os.path.splitext(name)[0].split('_')[-1])

    def split_continuous_segments(self, clip):
        # 🔥 STEP 1: sort by frame index
        clip = sorted(clip, key=self.get_frame_idx)
        indices = [self.get_frame_idx(p) for p in clip]

        segments = []
        current = [clip[0]]

        for i in range(1, len(indices)):
            gap = indices[i] - indices[i - 1]

            # 🔥 ONLY break if:
            # 1. reset (shouldn't happen after sorting but safe)
            # 2. HUGE gap (true discontinuity)
            if gap <= 0 or gap > 50:   # ← robust threshold (NOT median-based)
                segments.append(current)
                current = []

            current.append(clip[i])

        segments.append(current)
        return segments

    def call_on_load(self, clip):
        segments = self.split_continuous_segments(clip)

        # 🔥 FIX: ignore tiny segments completely
        segments = [s for s in segments if len(s) > 1]

        if len(segments) == 0:
            seg = sorted(clip, key=self.get_frame_idx)
        else:
            # 🔥 ALWAYS prefer longest segment (not random small one)
            seg = max(segments, key=len)

        return self._contiguous_window(seg)

    # ── video tensor mode ──────────────────────────────────────────────────

    def call_on_video(self, clip):
        """
        clip  : np.ndarray of shape (T, H, W, C) — already-loaded frames
                whose order is already temporal (frame 0, 5, 10, …).
        Returns a contiguous slice of seq_size frames from a random start.

        FIX: previously used np.random.choice which picked arbitrary indices.
             Now always picks start:start+seq_size — a single contiguous block.
        """
        return self._contiguous_window(clip)

    # ── shared core: pick a contiguous window of length seq_size ──────────

    def _contiguous_window(self, seq):
        """
        Given any sequence (list of paths OR np.ndarray axis-0),
        return seq_size consecutive elements starting at a random position.
        Pads by repeating the last element if the sequence is too short.
        """
        T = len(seq)

        if T >= self.seq_size:
            # random start, then take the next seq_size items in order
            start = np.random.randint(0, T - self.seq_size + 1)
            if isinstance(seq, np.ndarray):
                return seq[start : start + self.seq_size]
            else:
                return seq[start : start + self.seq_size]
        else:
            # sequence too short — take all frames + repeat the last one
            if isinstance(seq, np.ndarray):
                pad = np.stack([seq[-1]] * (self.seq_size - T), axis=0)
                return np.concatenate([seq, pad], axis=0)
            else:
                pad = [seq[-1]] * (self.seq_size - T)
                return list(seq) + pad


class RandomSequence(object):
    def __init__(self, seq_size, on_load=False):
        self.seq_size = seq_size
        self.on_load  = on_load

    def __call__(self, clip):
        # 🔥 FIX: numeric sort (CRITICAL)
        if isinstance(clip, list):
            clip = sorted(clip, key=self.get_frame_idx)
        else:
            # numpy array of paths
            clip = sorted(list(clip), key=self.get_frame_idx)

        return self._contiguous_window(clip)

    def get_frame_idx(self, path):
        name = os.path.basename(path)
        return int(os.path.splitext(name)[0].split('_')[-1])

    def _contiguous_window(self, seq):
        T = len(seq)

        if T >= self.seq_size:
            start = np.random.randint(0, T - self.seq_size + 1)
            return seq[start:start + self.seq_size], start
        else:
            return list(seq) + [seq[-1]] * (self.seq_size - T), 0