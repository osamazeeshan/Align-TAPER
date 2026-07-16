import os
import json
import joblib
import numpy as np
import torch
import opensmile

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import hdbscan

class OpenSMILEAudioDictionaryBuilder:
    """
    Build an OpenSMILE-based audio dictionary using labeled source data.

    Expected loader format:
        for feat, images, audio_arr, labels, video_path in train_loader:
            ...

    The dictionary can be:
        1. One prototype per class
        2. Multiple prototypes per class using KMeans
    """

    def __init__(
        self,
        sample_rate=16000,
        feature_set="eGeMAPSv02",
        feature_level="Functionals",
        prototypes_per_class=1,
        random_state=42,
    ):
        self.sample_rate = sample_rate
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state

        # OpenSMILE feature set
        if feature_set == "eGeMAPSv02":
            smile_feature_set = opensmile.FeatureSet.eGeMAPSv02
        elif feature_set == "ComParE_2016":
            smile_feature_set = opensmile.FeatureSet.ComParE_2016
        else:
            raise ValueError(f"Unsupported feature_set: {feature_set}")

        # OpenSMILE feature level
        if feature_level == "Functionals":
            smile_feature_level = opensmile.FeatureLevel.Functionals
        elif feature_level == "LowLevelDescriptors":
            smile_feature_level = opensmile.FeatureLevel.LowLevelDescriptors
        else:
            raise ValueError(f"Unsupported feature_level: {feature_level}")

        self.smile = opensmile.Smile(
            feature_set=smile_feature_set,
            feature_level=smile_feature_level,
        )

        self.scaler = StandardScaler()

        self.feature_names = None
        self.dictionary = None
        self.dictionary_labels = None
        self.prototype_metadata = None

    def _to_numpy_audio(self, audio):
        """
        Convert audio input to a 1D numpy array.

        Handles:
            - numpy array [T]
            - torch tensor [T]
            - torch tensor [1, T]
            - torch tensor [B, T] handled outside
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)

        # If shape is [1, T], squeeze channel
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        # If shape is [T, 1], squeeze last channel
        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Final safety
        audio = audio.reshape(-1).astype(np.float32)

        return audio

    def extract_feature(self, audio):
        """
        Extract OpenSMILE feature from one audio array.

        Args:
            audio: numpy array or torch tensor of shape [T]

        Returns:
            feat: numpy array of shape [D]
        """
        audio = self._to_numpy_audio(audio)

        if audio.size == 0:
            return None

        # Replace NaN/inf in raw audio
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # OpenSMILE extraction
        feature_df = self.smile.process_signal(audio, self.sample_rate)

        if self.feature_names is None:
            self.feature_names = list(feature_df.columns)

        feat = feature_df.values.squeeze().astype(np.float32)

        # Replace invalid values in OpenSMILE feature
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat

    def _unpack_batch_audio_and_labels(self, audio_arr, labels):
        """
        Normalize batch audio and labels into Python lists.

        Supports:
            audio_arr:
                - torch tensor [B, T]
                - torch tensor [T]
                - list of arrays
                - numpy array [B, T]
                - numpy array [T]

            labels:
                - torch tensor [B]
                - list
                - numpy array
                - scalar
        """

        # Handle labels
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        labels = np.asarray(labels)

        if labels.ndim == 0:
            labels_list = [labels.item()]
        else:
            labels_list = labels.tolist()

        # Handle audio
        if isinstance(audio_arr, torch.Tensor):
            audio_arr = audio_arr.detach().cpu()

            if audio_arr.ndim == 1:
                audio_list = [audio_arr]
            elif audio_arr.ndim == 2:
                audio_list = [audio_arr[i] for i in range(audio_arr.shape[0])]
            elif audio_arr.ndim == 3:
                # Example: [B, 1, T]
                audio_list = [audio_arr[i].squeeze(0) for i in range(audio_arr.shape[0])]
            else:
                raise ValueError(f"Unsupported audio tensor shape: {audio_arr.shape}")

        elif isinstance(audio_arr, np.ndarray):
            if audio_arr.ndim == 1:
                audio_list = [audio_arr]
            elif audio_arr.ndim == 2:
                audio_list = [audio_arr[i] for i in range(audio_arr.shape[0])]
            elif audio_arr.ndim == 3:
                audio_list = [audio_arr[i].squeeze(0) for i in range(audio_arr.shape[0])]
            else:
                raise ValueError(f"Unsupported audio numpy shape: {audio_arr.shape}")

        elif isinstance(audio_arr, (list, tuple)):
            audio_list = list(audio_arr)

        else:
            raise TypeError(f"Unsupported audio_arr type: {type(audio_arr)}")

        if len(audio_list) != len(labels_list):
            raise ValueError(
                f"Mismatch between audio batch and labels: "
                f"{len(audio_list)} audio samples vs {len(labels_list)} labels"
            )

        return audio_list, labels_list

    def collect_features_from_loader(self, train_loader):
        """
        Extract OpenSMILE features from your train_loader.

        Expected batch:
            feat, images, audio_arr, labels, video_path
        """

        all_features = []
        all_labels = []
        all_video_paths = []

        for batch in tqdm(train_loader, desc="Extracting OpenSMILE features"):
            feat, audio_arr, labels, video_path = batch

            audio_list, label_list = self._unpack_batch_audio_and_labels(
                audio_arr=audio_arr,
                labels=labels,
            )

            # video_path can be string or list
            if isinstance(video_path, (list, tuple)):
                video_path_list = list(video_path)
            else:
                video_path_list = [video_path] * len(audio_list)

            for audio, label, vp in zip(audio_list, label_list, video_path_list):
                audio_feat = self.extract_feature(audio)

                if audio_feat is None:
                    continue

                all_features.append(audio_feat)
                all_labels.append(label)
                all_video_paths.append(vp)

        X = np.stack(all_features, axis=0)
        y = np.asarray(all_labels)

        print("Collected OpenSMILE features:", X.shape)
        print("Collected labels:", y.shape)
        print("Unique labels:", np.unique(y))

        return X, y, all_video_paths

    def build_dictionary(self, X, y):
        """
        Build label-aware dictionary.

        If prototypes_per_class = 1:
            one mean prototype per label

        If prototypes_per_class > 1:
            KMeans prototypes inside each label
        """

        print("Normalizing OpenSMILE features...")
        X_norm = self.scaler.fit_transform(X)

        dictionary_vectors = []
        dictionary_labels = []
        prototype_metadata = []

        unique_labels = sorted(np.unique(y).tolist())

        for label in unique_labels:
            class_features = X_norm[y == label]
            num_samples = class_features.shape[0]

            if self.prototypes_per_class == 1:
                prototype = class_features.mean(axis=0)

                dictionary_vectors.append(prototype)
                dictionary_labels.append(label)

                prototype_metadata.append(
                    {
                        "label": str(label),
                        "prototype_index": 0,
                        "num_samples": int(num_samples),
                        "method": "class_mean",
                    }
                )

            else:
                k = min(self.prototypes_per_class, num_samples)

                if k == 1:
                    prototype = class_features.mean(axis=0)

                    dictionary_vectors.append(prototype)
                    dictionary_labels.append(label)

                    prototype_metadata.append(
                        {
                            "label": str(label),
                            "prototype_index": 0,
                            "num_samples": int(num_samples),
                            "method": "class_mean_fallback",
                        }
                    )
                else:
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=self.random_state,
                        n_init="auto",
                    )
                    kmeans.fit(class_features)

                    for proto_idx, center in enumerate(kmeans.cluster_centers_):
                        dictionary_vectors.append(center)
                        dictionary_labels.append(label)

                        cluster_size = int(np.sum(kmeans.labels_ == proto_idx))

                        prototype_metadata.append(
                            {
                                "label": str(label),
                                "prototype_index": int(proto_idx),
                                "num_samples": int(cluster_size),
                                "method": "class_kmeans",
                            }
                        )

        self.dictionary = np.stack(dictionary_vectors, axis=0)
        self.dictionary_labels = np.asarray(dictionary_labels)
        self.prototype_metadata = prototype_metadata

        print("Dictionary shape:", self.dictionary.shape)
        print("Dictionary labels:", self.dictionary_labels)

        return self.dictionary, self.dictionary_labels, X_norm

    def save(self, output_dir, X_raw=None, X_norm=None, y=None):
        """
        Save dictionary and supporting files.
        """

        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "opensmile_audio_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "dictionary_labels.npy"),
            self.dictionary_labels,
        )

        joblib.dump(
            self.scaler,
            os.path.join(output_dir, "opensmile_scaler.pkl"),
        )

        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(os.path.join(output_dir, "prototype_metadata.json"), "w") as f:
            json.dump(self.prototype_metadata, f, indent=2)

        summary = {
            "sample_rate": self.sample_rate,
            "num_dictionary_atoms": int(self.dictionary.shape[0]),
            "feature_dim": int(self.dictionary.shape[1]),
            "prototypes_per_class": int(self.prototypes_per_class),
            "unique_labels": sorted(list(set(map(str, self.dictionary_labels.tolist())))),
        }

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if X_raw is not None:
            np.save(os.path.join(output_dir, "opensmile_features_raw.npy"), X_raw)

        if X_norm is not None:
            np.save(os.path.join(output_dir, "opensmile_features_norm.npy"), X_norm)

        if y is not None:
            np.save(os.path.join(output_dir, "labels.npy"), y)

        print(f"Saved OpenSMILE audio dictionary to: {output_dir}")

    def fit_from_loader(self, train_loader, output_dir=None):
        """
        Complete pipeline:
            1. collect OpenSMILE features
            2. normalize features
            3. build dictionary
            4. optionally save files
        """

        X_raw, y, video_paths = self.collect_features_from_loader(train_loader)

        D_audio, dictionary_labels, X_norm = self.build_dictionary(X_raw, y)

        if output_dir is not None:
            self.save(
                output_dir=output_dir,
                X_raw=X_raw,
                X_norm=X_norm,
                y=y,
            )

        return {
            "dictionary": D_audio,
            "dictionary_labels": dictionary_labels,
            "X_raw": X_raw,
            "X_norm": X_norm,
            "y": y,
            "video_paths": video_paths,
            "feature_names": self.feature_names,
            "prototype_metadata": self.prototype_metadata,
        }


class OpenSMILEAudioDictionaryInferencer:
    """
    Load a saved OpenSMILE audio dictionary and compute similarity
    between a new audio array and dictionary prototypes.
    """

    def __init__(self, dictionary_dir, sample_rate=16000):
        self.dictionary_dir = dictionary_dir
        self.sample_rate = sample_rate

        self.dictionary = np.load(
            os.path.join(dictionary_dir, "opensmile_audio_dictionary.npy")
        )

        self.dictionary_labels = np.load(
            os.path.join(dictionary_dir, "dictionary_labels.npy"),
            allow_pickle=True,
        )

        self.scaler = joblib.load(
            os.path.join(dictionary_dir, "opensmile_scaler.pkl")
        )

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def _to_numpy_audio(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)

        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        audio = audio.reshape(-1).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        return audio

    def extract_feature(self, audio):
        audio = self._to_numpy_audio(audio)

        feature_df = self.smile.process_signal(audio, self.sample_rate)
        feat = feature_df.values.squeeze().astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat

    def compute_similarity(self, audio):
        """
        Args:
            audio: one audio array, shape [T]

        Returns:
            result dictionary containing:
                - similarity vector
                - dictionary labels
                - best label
                - best score
        """

        feat = self.extract_feature(audio)

        feat_norm = self.scaler.transform(feat.reshape(1, -1))

        sim = cosine_similarity(feat_norm, self.dictionary).squeeze(0)

        best_idx = int(np.argmax(sim))
        best_label = self.dictionary_labels[best_idx]
        best_score = float(sim[best_idx])

        return {
            "similarity": sim,
            "dictionary_labels": self.dictionary_labels,
            "best_label": best_label,
            "best_score": best_score,
        }



class OpenSMILEFeatureClusterDictionary:
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

    def __init__(
        self,
        sample_rate=16000,
        num_clusters=32,
        random_state=42,
        activation_threshold=1.0,
        positive_only=False,
    ):
        self.sample_rate = sample_rate
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.activation_threshold = activation_threshold
        self.positive_only = positive_only

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=random_state,
            n_init="auto",
        )

        self.dictionary = None
        self.cluster_labels = None
        self.cluster_metadata = None
        self.feature_names = None

    def build_threshold_activation(self, X_std):
        """
        Build sparse activation vectors using a threshold, not top-M.

        Args:
            X_std:
                Standardized OpenSMILE features, shape [N, 88].

        Returns:
            X_act:
                Sparse activation vectors, shape [N, 88].

        Logic:
            positive_only=False:
                keep feature if abs(z-score) >= activation_threshold

            positive_only=True:
                keep feature if z-score >= activation_threshold

        No fixed number of active features is enforced.
        Each sample can have a different number of active OpenSMILE features.
        """

        X_std = np.asarray(X_std, dtype=np.float32)
        X_act = np.zeros_like(X_std, dtype=np.float32)

        if self.positive_only:
            mask = X_std >= self.activation_threshold
        else:
            mask = np.abs(X_std) >= self.activation_threshold

        X_act[mask] = X_std[mask]

        return X_act

    def load(self, dictionary_dir):
        """
        Load saved OpenSMILE feature-cluster dictionary from path.

        Expected files:
            opensmile_feature_cluster_dictionary.npy
            cluster_labels.npy
            opensmile_scaler.pkl
            kmeans.pkl
            feature_names.json
            cluster_metadata.json
        """

        self.dictionary = np.load(
            os.path.join(dictionary_dir, "opensmile_feature_cluster_dictionary.npy")
        )

        self.cluster_labels = np.load(
            os.path.join(dictionary_dir, "cluster_labels.npy"),
            allow_pickle=True,
        )

        self.scaler = joblib.load(
            os.path.join(dictionary_dir, "opensmile_scaler.pkl")
        )

        self.kmeans = joblib.load(
            os.path.join(dictionary_dir, "kmeans.pkl")
        )

        feature_names_path = os.path.join(dictionary_dir, "feature_names.json")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                self.feature_names = json.load(f)

        cluster_metadata_path = os.path.join(dictionary_dir, "cluster_metadata.json")
        if os.path.exists(cluster_metadata_path):
            with open(cluster_metadata_path, "r") as f:
                self.cluster_metadata = json.load(f)

        self.num_clusters = self.dictionary.shape[0]

        print(f"Loaded feature-cluster dictionary from: {dictionary_dir}")
        print(f"Dictionary shape: {self.dictionary.shape}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Cluster labels: {self.cluster_labels}")

        return self

    def _to_numpy_audio(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)

        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        audio = audio.reshape(-1).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        return audio

    def extract_feature(self, audio):
        audio = self._to_numpy_audio(audio)

        feature_df = self.smile.process_signal(audio, self.sample_rate)

        if self.feature_names is None:
            self.feature_names = list(feature_df.columns)

        feat = feature_df.values.squeeze().astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat

    def collect_features_from_loader(self, train_loader):
        """
        Expected batch:
            feat, images, audio_arr, labels, video_path

        Expected:
            audio_arr: [B, T]
            labels:    [B]
        """

        X = []
        y = []
        paths = []

        for feat, audio_arr, labels, video_path in tqdm(
            train_loader,
            desc="Extracting OpenSMILE features"
        ):
            if isinstance(audio_arr, torch.Tensor):
                audio_arr = audio_arr.detach().cpu().numpy()

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            batch_size = audio_arr.shape[0]

            for i in range(batch_size):
                one_audio = audio_arr[i]
                one_label = labels[i]

                audio_feat = self.extract_feature(one_audio)

                X.append(audio_feat)
                y.append(one_label)

                if isinstance(video_path, (list, tuple)):
                    paths.append(video_path[i])
                else:
                    paths.append(video_path)

        X = np.stack(X, axis=0)
        y = np.asarray(y)

        print("Collected OpenSMILE features:", X.shape)
        print("Collected labels:", y.shape)
        print("Unique GT labels:", np.unique(y))

        return X, y, paths

    def fit(self, train_loader, output_dir=None):
        """
        Create dictionary by clustering feature-similar samples.

        Labels are not used during KMeans.
        """

        X_raw, y, paths = self.collect_features_from_loader(train_loader)

        print("Normalizing OpenSMILE features...")
        X_norm = self.scaler.fit_transform(X_raw)

        print(f"Clustering into {self.num_clusters} feature-based clusters...")
        cluster_ids = self.kmeans.fit_predict(X_norm)

        self.dictionary = self.kmeans.cluster_centers_

        cluster_labels = []
        cluster_metadata = []

        for cluster_id in range(self.num_clusters):
            idx = np.where(cluster_ids == cluster_id)[0]

            cluster_gt_labels = y[idx]

            if len(cluster_gt_labels) == 0:
                majority_label = None
                label_counts = {}
            else:
                counts = Counter(cluster_gt_labels.tolist())
                majority_label = counts.most_common(1)[0][0]
                label_counts = dict(counts)

            cluster_labels.append(majority_label)

            cluster_metadata.append(
                {
                    "cluster_id": int(cluster_id),
                    "assigned_label_majority_vote": None if majority_label is None else str(majority_label),
                    "num_samples": int(len(idx)),
                    "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
                }
            )

        self.cluster_labels = np.asarray(cluster_labels)
        self.cluster_metadata = cluster_metadata

        print("Dictionary shape:", self.dictionary.shape)
        print("Cluster labels by majority vote:", self.cluster_labels)

        if output_dir is not None:
            self.save(
                output_dir=output_dir,
                X_raw=X_raw,
                X_norm=X_norm,
                y=y,
                cluster_ids=cluster_ids,
            )

        return {
            "dictionary": self.dictionary,
            "cluster_labels": self.cluster_labels,
            "cluster_ids": cluster_ids,
            "X_raw": X_raw,
            "X_norm": X_norm,
            "y": y,
            "paths": paths,
            "cluster_metadata": self.cluster_metadata,
            "feature_names": self.feature_names,
        }

    def predict_one(self, audio):
        """
        Predict class of one audio sample.

        Prediction:
            audio → OpenSMILE → normalize → closest cluster center
            → predicted class = majority label of closest cluster
        """

        feat = self.extract_feature(audio)
        feat_norm = self.scaler.transform(feat.reshape(1, -1))

        sim = cosine_similarity(feat_norm, self.dictionary).squeeze(0)

        best_cluster = int(np.argmax(sim))
        pred_label = self.cluster_labels[best_cluster]

        return {
            "pred_label": pred_label,
            "best_cluster": best_cluster,
            "similarity": sim,
            "best_score": float(sim[best_cluster]),
        }

    def evaluate(self, data_loader):
        y_true = []
        y_pred = []

        for feat, audio_arr, labels, video_path in tqdm(
            data_loader,
            desc="Evaluating feature-cluster dictionary"
        ):
            if isinstance(audio_arr, torch.Tensor):
                audio_arr = audio_arr.detach().cpu().numpy()

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            batch_size = audio_arr.shape[0]

            for i in range(batch_size):
                one_audio = audio_arr[i]
                true_label = labels[i]

                result = self.predict_one(one_audio)
                pred_label = result["pred_label"]

                y_true.append(true_label)
                y_pred.append(pred_label)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        correct = int(np.sum(y_true == y_pred))
        total = len(y_true)

        war = accuracy_score(y_true, y_pred)
        uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        labels_order = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        print("\n================ Feature-Cluster Dictionary Results ================")
        print(f"Correct: {correct}/{total}")
        print(f"WAR / Accuracy: {war:.4f}")
        print(f"UAR / Macro Recall: {uar:.4f}")
        print(f"Macro F1: {f1:.4f}")

        print("\nLabels order:")
        print(labels_order)

        print("\nConfusion Matrix:")
        print(cm)

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
            "confusion_matrix": cm,
            "labels_order": labels_order,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save(self, output_dir, X_raw=None, X_norm=None, y=None, cluster_ids=None):
        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "opensmile_feature_cluster_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "cluster_labels.npy"),
            self.cluster_labels,
        )

        joblib.dump(
            self.scaler,
            os.path.join(output_dir, "opensmile_scaler.pkl"),
        )

        joblib.dump(
            self.kmeans,
            os.path.join(output_dir, "kmeans.pkl"),
        )

        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(os.path.join(output_dir, "cluster_metadata.json"), "w") as f:
            json.dump(self.cluster_metadata, f, indent=2)

        summary = {
            "sample_rate": self.sample_rate,
            "num_clusters": self.num_clusters,
            "dictionary_shape": list(self.dictionary.shape),
        }

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if X_raw is not None:
            np.save(os.path.join(output_dir, "opensmile_features_raw.npy"), X_raw)

        if X_norm is not None:
            np.save(os.path.join(output_dir, "opensmile_features_norm.npy"), X_norm)

        if y is not None:
            np.save(os.path.join(output_dir, "labels.npy"), y)

        if cluster_ids is not None:
            np.save(os.path.join(output_dir, "cluster_ids.npy"), cluster_ids)

        print(f"Saved feature-cluster dictionary to: {output_dir}")


class OpenSMILEActivatedFeatureClusterDictionary:
    """
    Activation-based OpenSMILE audio dictionary.

    This class creates clusters based on activated OpenSMILE feature patterns.

    Pipeline:
        audio_arr
        -> OpenSMILE eGeMAPSv02 features [88]
        -> StandardScaler
        -> threshold activation
        -> L2 normalization
        -> KMeans clustering
        -> dictionary atoms = cluster centers
        -> cluster label = majority GT label inside cluster

    Important:
        Labels are NOT used to create clusters.
        Labels are only used after clustering to assign a majority-vote label
        to each cluster for evaluation/inference.
    """

    def __init__(
        self,
        sample_rate=16000,
        num_clusters=32,
        activation_threshold=1.0,
        positive_only=False,
        random_state=42,
    ):
        self.sample_rate = sample_rate
        self.num_clusters = num_clusters
        self.activation_threshold = activation_threshold
        self.positive_only = positive_only
        self.random_state = random_state

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.scaler = StandardScaler()

        self.kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            n_init=20,
        )

        self.dictionary = None
        self.cluster_labels = None
        self.cluster_metadata = None
        self.feature_names = None

    # -------------------------------------------------------
    # Audio / feature extraction
    # -------------------------------------------------------

    def _to_numpy_audio(self, audio):
        """
        Convert audio to 1D numpy float32 array.

        Handles:
            torch tensor [T]
            torch tensor [1, T]
            numpy array [T]
            numpy array [1, T]
        """

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)

        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        audio = audio.reshape(-1).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        return audio

    def extract_feature(self, audio):
        """
        Extract OpenSMILE eGeMAPSv02 functionals from one audio array.

        Args:
            audio: array-like, shape [T]

        Returns:
            feat: numpy array, shape [88]
        """

        audio = self._to_numpy_audio(audio)

        feature_df = self.smile.process_signal(
            audio,
            self.sample_rate,
        )

        if self.feature_names is None:
            self.feature_names = list(feature_df.columns)

        feat = feature_df.values.squeeze().astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat

    # -------------------------------------------------------
    # Activation representation
    # -------------------------------------------------------

    def build_threshold_activation(self, X_std):
        """
        Build sparse activation vectors using thresholding.

        Args:
            X_std:
                Standardized OpenSMILE features, shape [N, 88]

        Returns:
            X_act:
                Sparse activation matrix, shape [N, 88]

        Logic:
            positive_only=False:
                keep feature if abs(z-score) >= activation_threshold

            positive_only=True:
                keep feature if z-score >= activation_threshold

        No top-M selection is used.
        Each sample can have a different number of active features.
        """

        X_std = np.asarray(X_std, dtype=np.float32)
        X_act = np.zeros_like(X_std, dtype=np.float32)

        if self.positive_only:
            mask = X_std >= self.activation_threshold
        else:
            mask = np.abs(X_std) >= self.activation_threshold

        X_act[mask] = X_std[mask]

        return X_act

    def transform_to_cluster_space(self, X_raw, fit_scaler=False):
        """
        Convert raw OpenSMILE features into clustering representation.

        Steps:
            raw OpenSMILE features
            -> standardize
            -> threshold activation
            -> L2 normalize

        Args:
            X_raw:
                Raw OpenSMILE features, shape [N, 88]

            fit_scaler:
                True during training/source dictionary creation.
                False during inference.

        Returns:
            X_std:
                standardized OpenSMILE features

            X_act:
                threshold activation features

            X_cluster:
                L2-normalized activation features used for clustering/similarity
        """

        if fit_scaler:
            X_std = self.scaler.fit_transform(X_raw)
        else:
            X_std = self.scaler.transform(X_raw)

        X_act = self.build_threshold_activation(X_std)

        active_count = np.sum(X_act != 0, axis=1)

        X_cluster = normalize(X_act, norm="l2", axis=1)

        return X_std, X_act, X_cluster, active_count

    # -------------------------------------------------------
    # Data collection from source loader
    # -------------------------------------------------------

    def collect_features_from_loader(self, train_loader):
        """
        Collect OpenSMILE features and labels from source train_loader.

        Supports batch format:
            feat, audio_arr, labels, video_path

        or:
            feat, images, audio_arr, labels, video_path

        Expected:
            audio_arr: [B, T]
            labels:    [B]
        """

        X_raw = []
        y = []
        paths = []

        for batch in tqdm(train_loader, desc="Extracting OpenSMILE features"):
            if len(batch) == 4:
                feat, audio_arr, labels, video_path = batch
            elif len(batch) == 5:
                feat, images, audio_arr, labels, video_path = batch
            else:
                raise ValueError(
                    f"Expected batch length 4 or 5, got {len(batch)}"
                )

            if isinstance(audio_arr, torch.Tensor):
                audio_arr = audio_arr.detach().cpu().numpy()

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            batch_size = audio_arr.shape[0]

            for i in range(batch_size):
                one_audio = audio_arr[i]
                one_label = labels[i]

                audio_feat = self.extract_feature(one_audio)

                X_raw.append(audio_feat)
                y.append(one_label)

                if isinstance(video_path, (list, tuple)):
                    paths.append(video_path[i])
                else:
                    paths.append(video_path)

        X_raw = np.stack(X_raw, axis=0)
        y = np.asarray(y)

        print("Raw OpenSMILE feature shape:", X_raw.shape)
        print("Labels shape:", y.shape)
        print("Unique labels:", np.unique(y))

        return X_raw, y, paths

    # -------------------------------------------------------
    # Fit dictionary
    # -------------------------------------------------------

    def fit(self, train_loader, output_dir=None):
        """
        Create activation-based OpenSMILE feature-cluster dictionary.

        This is the main function you call.

        Example:
            cluster_dict = OpenSMILEFeatureClusterDictionary(
                sample_rate=16000,
                num_clusters=32,
            )

            result = cluster_dict.fit(
                train_loader=train_loader,
                output_dir="outputs/opensmile_feature_cluster_dictionary"
            )

        Clustering does NOT use labels.
        Labels are only used after clustering to assign majority labels.
        """

        # 1. Extract raw OpenSMILE features from source data
        X_raw, y, paths = self.collect_features_from_loader(train_loader)

        # 2. Convert raw features to activation-pattern clustering space
        X_std, X_act, X_cluster, active_count = self.transform_to_cluster_space(
            X_raw,
            fit_scaler=True,
        )

        print("\nActivation statistics:")
        print(f"Activation threshold: {self.activation_threshold}")
        print(f"Positive only: {self.positive_only}")
        print(f"Average active features per sample: {active_count.mean():.2f}")
        print(f"Min active features: {active_count.min()}")
        print(f"Max active features: {active_count.max()}")

        num_zero_rows = int(np.sum(active_count == 0))
        if num_zero_rows > 0:
            print(
                f"Warning: {num_zero_rows} samples have zero active features. "
                f"Consider lowering activation_threshold."
            )

        # 3. Cluster based on sparse activation feature patterns
        print(
            f"\nClustering {X_cluster.shape[0]} samples into "
            f"{self.num_clusters} activation-pattern clusters..."
        )

        min_active_features = 3
        valid_idx = active_count >= min_active_features
        X_cluster_fit = X_cluster[valid_idx]
        y_fit = y[valid_idx]
        paths_fit = [paths[i] for i in np.where(valid_idx)[0]]

        print(f"Using {X_cluster_fit.shape[0]}/{X_cluster.shape[0]} samples for clustering")

        cluster_ids = self.kmeans.fit_predict(X_cluster)

        # 4. Dictionary atoms = cluster centers in activation space
        self.dictionary = normalize(
            self.kmeans.cluster_centers_,
            norm="l2",
            axis=1,
        )

        # 5. Assign each cluster a majority label using source GT labels
        cluster_labels = []
        cluster_metadata = []

        for cluster_id in range(self.num_clusters):
            idx = np.where(cluster_ids == cluster_id)[0]
            cluster_gt_labels = y[idx]

            if len(cluster_gt_labels) == 0:
                majority_label = None
                label_counts = {}
                mean_active = 0.0
            else:
                counts = Counter(cluster_gt_labels.tolist())
                majority_label = counts.most_common(1)[0][0]
                label_counts = dict(counts)
                mean_active = float(active_count[idx].mean())

            cluster_labels.append(majority_label)

            cluster_metadata.append(
                {
                    "cluster_id": int(cluster_id),
                    "assigned_label_majority_vote": None
                    if majority_label is None
                    else str(majority_label),
                    "num_samples": int(len(idx)),
                    "label_distribution": {
                        str(k): int(v) for k, v in label_counts.items()
                    },
                    "mean_active_features": mean_active,
                }
            )

        self.cluster_labels = np.asarray(cluster_labels)
        self.cluster_metadata = cluster_metadata

        print("\nDictionary created.")
        print("Dictionary shape:", self.dictionary.shape)
        print("Cluster labels:", self.cluster_labels)

        # 6. Save all files
        if output_dir is not None:
            self.save(
                output_dir=output_dir,
                X_raw=X_raw,
                X_std=X_std,
                X_act=X_act,
                X_cluster=X_cluster,
                y=y,
                cluster_ids=cluster_ids,
                active_count=active_count,
                paths=paths,
            )

        return {
            "dictionary": self.dictionary,
            "cluster_labels": self.cluster_labels,
            "cluster_ids": cluster_ids,
            "X_raw": X_raw,
            "X_std": X_std,
            "X_act": X_act,
            "X_cluster": X_cluster,
            "active_count": active_count,
            "y": y,
            "paths": paths,
            "cluster_metadata": self.cluster_metadata,
            "feature_names": self.feature_names,
        }

    def fit_hdbscan(
        self,
        train_loader,
        output_dir=None,
        min_cluster_size=10,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        include_noise_dictionary=False):
        
        """
        Create activation-based OpenSMILE dictionary using HDBSCAN.

        This does NOT predefine the number of clusters.

        Clustering logic:
            1. Extract OpenSMILE features.
            2. Standardize features.
            3. Threshold activation.
            4. L2 normalize activation vectors.
            5. HDBSCAN discovers natural clusters.
            6. Samples that do not belong to any stable cluster are labeled -1.
            7. Dictionary atoms are mean vectors of discovered clusters.
            8. Each cluster gets a majority label using source labels.

        Args:
            train_loader:
                Source dataloader.

            output_dir:
                Folder to save dictionary outputs.

            min_cluster_size:
                Minimum number of samples required to form a cluster.
                Larger value = fewer, more stable clusters.
                Smaller value = more small clusters.

            min_samples:
                Controls how conservative outlier detection is.
                If None, HDBSCAN uses min_cluster_size.
                Larger value = more samples marked as noise.

            cluster_selection_epsilon:
                Allows nearby clusters to merge.
                Usually keep 0.0 first.

            include_noise_dictionary:
                If True, creates one dictionary atom for noise samples labeled -1.
                Usually False.
        """

        # 1. Extract raw OpenSMILE features
        X_raw, y, paths = self.collect_features_from_loader(train_loader)

        # 2. Convert to activation-pattern space
        X_std, X_act, X_cluster, active_count = self.transform_to_cluster_space(
            X_raw,
            fit_scaler=True,
        )

        print("\nActivation statistics:")
        print(f"Activation threshold: {self.activation_threshold}")
        print(f"Positive only: {self.positive_only}")
        print(f"Average active features per sample: {active_count.mean():.2f}")
        print(f"Min active features: {active_count.min()}")
        print(f"Max active features: {active_count.max()}")
        print(f"Zero-active samples: {int(np.sum(active_count == 0))}")

        # 3. HDBSCAN clustering
        print("\nRunning HDBSCAN activation-pattern clustering...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method="eom",
            prediction_data=True,
        )

        cluster_ids = clusterer.fit_predict(X_cluster)

        unique_cluster_ids = sorted([c for c in np.unique(cluster_ids) if c != -1])
        num_noise = int(np.sum(cluster_ids == -1))

        print(f"Discovered clusters: {len(unique_cluster_ids)}")
        print(f"Noise/outlier samples: {num_noise}/{len(cluster_ids)}")

        # 4. Build dictionary atoms from discovered clusters
        dictionary_vectors = []
        dictionary_cluster_ids = []
        cluster_labels = []
        cluster_metadata = []

        for cluster_id in unique_cluster_ids:
            idx = np.where(cluster_ids == cluster_id)[0]
            cluster_vectors = X_cluster[idx]

            # mean activation vector for this natural cluster
            center = cluster_vectors.mean(axis=0, keepdims=True)
            center = normalize(center, norm="l2", axis=1).squeeze(0)

            dictionary_vectors.append(center)
            dictionary_cluster_ids.append(cluster_id)

            cluster_gt_labels = y[idx]
            counts = Counter(cluster_gt_labels.tolist())
            majority_label = counts.most_common(1)[0][0]

            cluster_labels.append(majority_label)

            cluster_metadata.append(
                {
                    "cluster_id": int(cluster_id),
                    "dictionary_index": int(len(dictionary_vectors) - 1),
                    "assigned_label_majority_vote": str(majority_label),
                    "num_samples": int(len(idx)),
                    "label_distribution": {
                        str(k): int(v) for k, v in counts.items()
                    },
                    "mean_active_features": float(active_count[idx].mean()),
                    "min_active_features": int(active_count[idx].min()),
                    "max_active_features": int(active_count[idx].max()),
                }
            )

        # Optional: include noise as one special dictionary atom
        if include_noise_dictionary and num_noise > 0:
            noise_idx = np.where(cluster_ids == -1)[0]
            noise_center = X_cluster[noise_idx].mean(axis=0, keepdims=True)
            noise_center = normalize(noise_center, norm="l2", axis=1).squeeze(0)

            dictionary_vectors.append(noise_center)
            dictionary_cluster_ids.append(-1)

            noise_labels = y[noise_idx]
            counts = Counter(noise_labels.tolist())
            majority_label = counts.most_common(1)[0][0]

            cluster_labels.append(majority_label)

            cluster_metadata.append(
                {
                    "cluster_id": -1,
                    "dictionary_index": int(len(dictionary_vectors) - 1),
                    "assigned_label_majority_vote": str(majority_label),
                    "num_samples": int(len(noise_idx)),
                    "label_distribution": {
                        str(k): int(v) for k, v in counts.items()
                    },
                    "mean_active_features": float(active_count[noise_idx].mean()),
                    "note": "noise_dictionary_atom",
                }
            )

        if len(dictionary_vectors) == 0:
            raise RuntimeError(
                "HDBSCAN found no valid clusters. "
                "Try lowering min_cluster_size or activation_threshold."
            )

        self.dictionary = np.stack(dictionary_vectors, axis=0)
        self.cluster_labels = np.asarray(cluster_labels)
        self.cluster_metadata = cluster_metadata

        # Save extra fields
        self.hdbscan_clusterer = clusterer
        self.hdbscan_cluster_ids = np.asarray(dictionary_cluster_ids)

        print("\nDictionary created.")
        print("Dictionary shape:", self.dictionary.shape)
        print("Dictionary cluster IDs:", self.hdbscan_cluster_ids)
        print("Dictionary labels:", self.cluster_labels)

        if output_dir is not None:
            self.save_hdbscan_dictionary(
                output_dir=output_dir,
                X_raw=X_raw,
                X_std=X_std,
                X_act=X_act,
                X_cluster=X_cluster,
                y=y,
                cluster_ids=cluster_ids,
                active_count=active_count,
                paths=paths,
                hdbscan_cluster_ids=self.hdbscan_cluster_ids,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                include_noise_dictionary=include_noise_dictionary,
            )

        return {
            "dictionary": self.dictionary,
            "cluster_labels": self.cluster_labels,
            "cluster_ids": cluster_ids,
            "hdbscan_cluster_ids": self.hdbscan_cluster_ids,
            "X_raw": X_raw,
            "X_std": X_std,
            "X_act": X_act,
            "X_cluster": X_cluster,
            "active_count": active_count,
            "y": y,
            "paths": paths,
            "cluster_metadata": self.cluster_metadata,
            "feature_names": self.feature_names,
            "num_discovered_clusters": len(unique_cluster_ids),
            "num_noise": num_noise,
        }


    def save_hdbscan_dictionary(
        self,
        output_dir,
        X_raw,
        X_std,
        X_act,
        X_cluster,
        y,
        cluster_ids,
        active_count,
        paths=None,
        hdbscan_cluster_ids=None,
        min_cluster_size=None,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        include_noise_dictionary=False):

        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "opensmile_feature_cluster_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "opensmile_hdbscan_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "cluster_labels.npy"),
            self.cluster_labels,
        )

        np.save(
            os.path.join(output_dir, "cluster_ids.npy"),
            cluster_ids,
        )

        if hdbscan_cluster_ids is not None:
            np.save(
                os.path.join(output_dir, "hdbscan_dictionary_cluster_ids.npy"),
                hdbscan_cluster_ids,
            )

        np.save(
            os.path.join(output_dir, "opensmile_features_raw.npy"),
            X_raw,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_std.npy"),
            X_std,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_activation.npy"),
            X_act,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_cluster_input.npy"),
            X_cluster,
        )

        np.save(
            os.path.join(output_dir, "active_count.npy"),
            active_count,
        )

        np.save(
            os.path.join(output_dir, "labels.npy"),
            y,
        )

        joblib.dump(
            self.scaler,
            os.path.join(output_dir, "opensmile_scaler.pkl"),
        )

        if hasattr(self, "hdbscan_clusterer"):
            joblib.dump(
                self.hdbscan_clusterer,
                os.path.join(output_dir, "hdbscan_clusterer.pkl"),
            )

        with open(os.path.join(output_dir, "cluster_metadata.json"), "w") as f:
            json.dump(self.cluster_metadata, f, indent=2)

        if self.feature_names is not None:
            with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
                json.dump(self.feature_names, f, indent=2)

        if paths is not None:
            with open(os.path.join(output_dir, "paths.json"), "w") as f:
                json.dump([str(p) for p in paths], f, indent=2)

        summary = {
            "dictionary_type": "hdbscan_threshold_activation_opensmile_dictionary",
            "sample_rate": int(self.sample_rate),
            "feature_dim": int(self.dictionary.shape[1]),
            "num_dictionary_atoms": int(self.dictionary.shape[0]),
            "activation_threshold": float(self.activation_threshold),
            "positive_only": bool(self.positive_only),
            "num_samples": int(len(y)),
            "avg_active_features": float(active_count.mean()),
            "min_active_features": int(active_count.min()),
            "max_active_features": int(active_count.max()),
            "min_cluster_size": None if min_cluster_size is None else int(min_cluster_size),
            "min_samples": None if min_samples is None else int(min_samples),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
            "include_noise_dictionary": bool(include_noise_dictionary),
            "num_noise_samples": int(np.sum(cluster_ids == -1)),
        }

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved HDBSCAN OpenSMILE dictionary to: {output_dir}")

    # -------------------------------------------------------
    # Inference
    # -------------------------------------------------------

    def predict_one(self, audio):
        """
        Predict using the activation-based dictionary.

        Steps:
            audio
            -> OpenSMILE feature
            -> StandardScaler
            -> threshold activation
            -> L2 normalize
            -> cosine similarity to dictionary atoms
            -> closest cluster
            -> predicted label = majority label of closest cluster
        """

        feat = self.extract_feature(audio)
        feat = feat.reshape(1, -1)

        X_std, X_act, X_cluster, active_count = self.transform_to_cluster_space(
            feat,
            fit_scaler=False,
        )

        sim = cosine_similarity(X_cluster, self.dictionary).squeeze(0)

        best_cluster = int(np.argmax(sim))
        pred_label = self.cluster_labels[best_cluster]

        return {
            "pred_label": pred_label,
            "best_cluster": best_cluster,
            "similarity": sim,
            "best_score": float(sim[best_cluster]),
            "num_active_features": int(active_count[0]),
            "activation_vector": X_act.squeeze(0),
            "cluster_input_vector": X_cluster.squeeze(0),
        }

    # -------------------------------------------------------
    # Save / load
    # -------------------------------------------------------

    def save(
        self,
        output_dir,
        X_raw,
        X_std,
        X_act,
        X_cluster,
        y,
        cluster_ids,
        active_count,
        paths=None,
    ):
        """
        Save dictionary and all intermediate representations.
        """

        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(output_dir, "opensmile_feature_cluster_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "opensmile_activation_dictionary.npy"),
            self.dictionary,
        )

        np.save(
            os.path.join(output_dir, "cluster_labels.npy"),
            self.cluster_labels,
        )

        np.save(
            os.path.join(output_dir, "cluster_ids.npy"),
            cluster_ids,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_raw.npy"),
            X_raw,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_std.npy"),
            X_std,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_activation.npy"),
            X_act,
        )

        np.save(
            os.path.join(output_dir, "opensmile_features_cluster_input.npy"),
            X_cluster,
        )

        np.save(
            os.path.join(output_dir, "active_count.npy"),
            active_count,
        )

        np.save(
            os.path.join(output_dir, "labels.npy"),
            y,
        )

        joblib.dump(
            self.scaler,
            os.path.join(output_dir, "opensmile_scaler.pkl"),
        )

        joblib.dump(
            self.kmeans,
            os.path.join(output_dir, "kmeans.pkl"),
        )

        with open(os.path.join(output_dir, "cluster_metadata.json"), "w") as f:
            json.dump(self.cluster_metadata, f, indent=2)

        if self.feature_names is not None:
            with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
                json.dump(self.feature_names, f, indent=2)

        if paths is not None:
            with open(os.path.join(output_dir, "paths.json"), "w") as f:
                json.dump([str(p) for p in paths], f, indent=2)

        summary = {
            "dictionary_type": "threshold_activation_based_opensmile_dictionary",
            "sample_rate": int(self.sample_rate),
            "num_clusters": int(self.num_clusters),
            "feature_dim": int(self.dictionary.shape[1]),
            "activation_threshold": float(self.activation_threshold),
            "positive_only": bool(self.positive_only),
            "num_samples": int(len(y)),
            "avg_active_features": float(active_count.mean()),
            "min_active_features": int(active_count.min()),
            "max_active_features": int(active_count.max()),
        }

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved activation-based OpenSMILE dictionary to: {output_dir}")

    def load(self, dictionary_dir):
        """
        Load saved activation-based OpenSMILE dictionary.
        """

        dict_path_1 = os.path.join(
            dictionary_dir,
            "opensmile_feature_cluster_dictionary.npy",
        )

        dict_path_2 = os.path.join(
            dictionary_dir,
            "opensmile_activation_dictionary.npy",
        )

        if os.path.exists(dict_path_1):
            self.dictionary = np.load(dict_path_1)
        elif os.path.exists(dict_path_2):
            self.dictionary = np.load(dict_path_2)
        else:
            raise FileNotFoundError(
                "Could not find dictionary file in directory."
            )

        self.cluster_labels = np.load(
            os.path.join(dictionary_dir, "cluster_labels.npy"),
            allow_pickle=True,
        )

        self.scaler = joblib.load(
            os.path.join(dictionary_dir, "opensmile_scaler.pkl"),
        )

        kmeans_path = os.path.join(dictionary_dir, "kmeans.pkl")
        if os.path.exists(kmeans_path):
            self.kmeans = joblib.load(kmeans_path)

        metadata_path = os.path.join(dictionary_dir, "cluster_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.cluster_metadata = json.load(f)

        feature_names_path = os.path.join(dictionary_dir, "feature_names.json")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                self.feature_names = json.load(f)

        summary_path = os.path.join(dictionary_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)

            self.sample_rate = summary.get("sample_rate", self.sample_rate)
            self.num_clusters = summary.get("num_clusters", self.num_clusters)
            self.activation_threshold = summary.get(
                "activation_threshold",
                self.activation_threshold,
            )
            self.positive_only = summary.get(
                "positive_only",
                self.positive_only,
            )

        print(f"Loaded dictionary from: {dictionary_dir}")
        print("Dictionary shape:", self.dictionary.shape)
        print("Activation threshold:", self.activation_threshold)
        print("Positive only:", self.positive_only)
        print("Cluster labels:", self.cluster_labels)

        return self

    # -------------------------------------------------------
    # Optional helper for interpretation
    # -------------------------------------------------------

    def print_cluster_active_features(self, top_k=10):
        """
        Print dominant features for each dictionary atom.
        """

        if self.dictionary is None:
            raise ValueError("Dictionary is not created or loaded yet.")

        if self.feature_names is None:
            raise ValueError("feature_names are missing.")

        for cluster_id, atom in enumerate(self.dictionary):
            top_idx = np.argsort(np.abs(atom))[::-1][:top_k]

            print("\n" + "=" * 70)
            print(f"Cluster {cluster_id}")

            if self.cluster_labels is not None:
                print(f"Majority label: {self.cluster_labels[cluster_id]}")

            if self.cluster_metadata is not None:
                print(
                    "Label distribution:",
                    self.cluster_metadata[cluster_id].get(
                        "label_distribution",
                        {},
                    ),
                )

            print("Dominant OpenSMILE features:")
            for idx in top_idx:
                print(f"  {self.feature_names[idx]}: {atom[idx]:.4f}")
    
    def evaluate(self, data_loader, label_to_name=None, print_report=True):
        """
        Evaluate activation-based OpenSMILE dictionary.

        Expected batch:
            feat, audio_arr, labels, video_path

        or:
            feat, images, audio_arr, labels, video_path

        Prediction:
            audio -> OpenSMILE -> threshold activation -> closest dictionary atom
            -> predicted label = majority label of closest cluster
        """

        y_true = []
        y_pred = []
        y_cluster = []
        y_score = []
        y_active_count = []

        for batch in tqdm(data_loader, desc="Evaluating OpenSMILE dictionary"):
            if len(batch) == 4:
                feat, audio_arr, labels, video_path = batch
            elif len(batch) == 5:
                feat, images, audio_arr, labels, video_path = batch
            else:
                raise ValueError(
                    f"Expected batch length 4 or 5, got {len(batch)}"
                )

            if isinstance(audio_arr, torch.Tensor):
                audio_arr = audio_arr.detach().cpu().numpy()

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            batch_size = audio_arr.shape[0]

            for i in range(batch_size):
                one_audio = audio_arr[i]
                true_label = labels[i]

                if label_to_name is not None:
                    true_label = label_to_name[int(true_label)]

                result = self.predict_one(one_audio)

                pred_label = result["pred_label"]

                if isinstance(pred_label, np.generic):
                    pred_label = pred_label.item()

                y_true.append(true_label)
                y_pred.append(pred_label)
                y_cluster.append(result["best_cluster"])
                y_score.append(result["best_score"])
                y_active_count.append(result["num_active_features"])

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

        print("\n================ OpenSMILE Dictionary Evaluation ================")
        print(f"Correct: {correct}/{total}")
        print(f"WAR / Accuracy: {war:.4f}")
        print(f"UAR / Macro Recall: {uar:.4f}")
        print(f"Macro F1: {f1:.4f}")
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
        }

    def _compute_2d_cluster_centers(self, Z_samples, cluster_ids):
        """
        Compute 2D visual cluster centers after t-SNE/UMAP.

        This is better than projecting 88D cluster centers directly,
        because t-SNE/UMAP are nonlinear.
        """

        unique_clusters = np.unique(cluster_ids)
        centers_2d = []

        for cluster_id in unique_clusters:
            idx = cluster_ids == cluster_id
            center = Z_samples[idx].mean(axis=0)
            centers_2d.append(center)

        return np.asarray(centers_2d), unique_clusters


    def visualize_clusters(
        self,
        dictionary_dir=None,
        X_cluster=None,
        cluster_ids=None,
        labels=None,
        method="tsne",
        save_dir=None,
        perplexity=30,
        n_neighbors=15,
        min_dist=0.1,
        max_points=None,
        random_state=42,
    ):
        """
        Visualize OpenSMILE activation clusters using t-SNE or UMAP.

        Recommended call after fit:
            vis_result = cluster_dict.visualize_clusters(
                dictionary_dir="outputs/opensmile_feature_cluster_dictionary",
                save_dir="outputs/opensmile_feature_cluster_dictionary/visualization",
                method="tsne",
            )

        Or using fit result:
            vis_result = cluster_dict.visualize_clusters(
                X_cluster=result["X_cluster"],
                cluster_ids=result["cluster_ids"],
                labels=result["y"],
                save_dir="outputs/vis",
            )
        """

        if dictionary_dir is not None:
            X_cluster = np.load(
                os.path.join(dictionary_dir, "opensmile_features_cluster_input.npy")
            )
            cluster_ids = np.load(
                os.path.join(dictionary_dir, "cluster_ids.npy")
            )
            labels = np.load(
                os.path.join(dictionary_dir, "labels.npy"),
                allow_pickle=True,
            )

        if X_cluster is None or cluster_ids is None or labels is None:
            raise ValueError(
                "Provide either dictionary_dir or X_cluster, cluster_ids, and labels."
            )

        X_cluster = np.asarray(X_cluster)
        cluster_ids = np.asarray(cluster_ids)
        labels = np.asarray(labels)

        # Optional subsampling for speed
        if max_points is not None and X_cluster.shape[0] > max_points:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X_cluster.shape[0], size=max_points, replace=False)

            X_vis = X_cluster[idx]
            cluster_ids_vis = cluster_ids[idx]
            labels_vis = labels[idx]
        else:
            X_vis = X_cluster
            cluster_ids_vis = cluster_ids
            labels_vis = labels

        print("Visualization input:", X_vis.shape)

        if method.lower() == "tsne":
            pca_dim = min(50, X_vis.shape[1], X_vis.shape[0])
            X_pca = PCA(
                n_components=pca_dim,
                random_state=random_state,
            ).fit_transform(X_vis)

            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            )

            Z_samples = reducer.fit_transform(X_pca)
            method_name = "t-SNE"

        elif method.lower() == "umap":
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )

            Z_samples = reducer.fit_transform(X_vis)
            method_name = "UMAP"

        else:
            raise ValueError("method must be 'tsne' or 'umap'")

        Z_centers_2d, unique_clusters = self._compute_2d_cluster_centers(
            Z_samples,
            cluster_ids_vis,
        )

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # -------------------------------------------------------
        # Plot 1: color by feature cluster
        # -------------------------------------------------------
        plt.figure(figsize=(11, 8))

        for cluster_id in unique_clusters:
            idx = cluster_ids_vis == cluster_id
            plt.scatter(
                Z_samples[idx, 0],
                Z_samples[idx, 1],
                s=12,
                alpha=0.65,
                label=f"C{cluster_id}",
            )

        plt.scatter(
            Z_centers_2d[:, 0],
            Z_centers_2d[:, 1],
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
            label="2D cluster centers",
        )

        for j, cluster_id in enumerate(unique_clusters):
            label_text = f"C{cluster_id}"

            if self.cluster_labels is not None and cluster_id < len(self.cluster_labels):
                label_text += f":{self.cluster_labels[cluster_id]}"

            plt.text(
                Z_centers_2d[j, 0],
                Z_centers_2d[j, 1],
                label_text,
                fontsize=8,
                weight="bold",
            )

        plt.title(f"{method_name} of OpenSMILE activation clusters")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=7,
            ncol=1,
        )
        plt.tight_layout()

        cluster_plot_path = None
        if save_dir is not None:
            cluster_plot_path = os.path.join(
                save_dir,
                f"{method.lower()}_by_cluster.png",
            )
            plt.savefig(cluster_plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {cluster_plot_path}")

        plt.show()

        # -------------------------------------------------------
        # Plot 2: color by GT label
        # -------------------------------------------------------
        plt.figure(figsize=(11, 8))

        unique_labels = np.unique(labels_vis)

        for label in unique_labels:
            idx = labels_vis == label
            plt.scatter(
                Z_samples[idx, 0],
                Z_samples[idx, 1],
                s=12,
                alpha=0.65,
                label=str(label),
            )

        plt.scatter(
            Z_centers_2d[:, 0],
            Z_centers_2d[:, 1],
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
            label="2D cluster centers",
        )

        for j, cluster_id in enumerate(unique_clusters):
            plt.text(
                Z_centers_2d[j, 0],
                Z_centers_2d[j, 1],
                f"C{cluster_id}",
                fontsize=8,
                weight="bold",
            )

        plt.title(f"{method_name} of OpenSMILE samples by GT label")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )
        plt.tight_layout()

        label_plot_path = None
        if save_dir is not None:
            label_plot_path = os.path.join(
                save_dir,
                f"{method.lower()}_by_label.png",
            )
            plt.savefig(label_plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {label_plot_path}")

        plt.show()

        return {
            "Z_samples": Z_samples,
            "Z_centers_2d": Z_centers_2d,
            "cluster_ids": cluster_ids_vis,
            "labels": labels_vis,
            "unique_clusters": unique_clusters,
            "cluster_plot_path": cluster_plot_path,
            "label_plot_path": label_plot_path,
        }