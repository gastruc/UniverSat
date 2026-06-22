import gc
import io
import math
import time
from pathlib import Path
from typing import Optional, Union

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

from data.datamodule import DataModule
from data.transforms.transform import Identity
from src import utils

log = utils.get_pylogger(__name__)

try:
    import importlib
    _stopit = importlib.import_module("stopit")
    Timeout = getattr(_stopit, "SignalTimeout")  # ThreadingTimeout?
    USE_TIMEOUT = True
except Exception:
    Timeout = None  # type: ignore
    USE_TIMEOUT = False

SOLVER = 'lbfgs'
MAX_ITER = 1000
TOL=3e-3
TIMEOUT=40*60
WAVELENGTHS = {
    "aerial": [0.665, 0.56, 0.49, "NIR"],
    "aerial-flair": [0.665, 0.56, 0.49, "NIR", "DSM"],
    "spot": [0.665, 0.56, 0.49],
    "naip": [0.665, 0.56, 0.49, "NIR"],
    "s2": [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19],
    "s1": ["VV", "VH", "Ratio_VV_VH"],
    "l8": 10,
    "l7": 30,
    "alos": ["HH", "HV", "Ratio_HH_HV"],
    "modis": 250
}
SUBPATCHES = {
    "s1": 1,
    "s2": 1,
    "l8": 1,
    "l7": 1,
    "alos": 1,
    "modis": 1,
    "aerial": 10,
    "aerial-flair": 10,
    "spot": 10,
    "naip": 10
}
INPUT_RES = {
    "aerial": 0.2,
    "aerial-flair": 0.2,
    "spot": 1,
    "naip": 1.25,
    "s2": 10,
    "s1": 10,
    "l8": 10,
    "l7": 30,
    "alos": 30,
    "modis": 250
}
EVAL_DATASETS = {
    # "PASTIS-R-patch": {
    #     "config": "Pastis",
    #     "train_augmentation": Identity(),
    #     "test_augmentation": Identity(),
    #     "modalities": ["spot", "s2", "s1"],
    #     "scale": 4,
    #     "task_type": "semseg",
    #     "num_workers": 2,
    #     "batch_size": 1,
    #     "max_iter_train": 200,
    #     "max_iter_test": 200,
    #     "semseg_drop_rate": 0.9,
    #     "output_grid": "patch",
    #     "compute_tSNE": False,
    #     'overrides': {'classif':False},
    #     'ignore_labels': [19],
    # },
    "PASTIS-Full": {
        "config": "Pastis",
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["spot", "s2", "s1"],
        "scale": 4,
        "task_type": "semseg",
        "num_workers": 2,
        "batch_size": 1,
        "max_iter_train": 400,
        "max_iter_test": 400,
        "semseg_drop_rate": 0.75,
        "output_grid": "dense",
        "compute_tSNE": True,
        "n_full_samples": 4,
        "full_samples_modality": 'spot',
        'overrides': {'classif':False},
        'ignore_labels': [19],
    },
    # "PASTIS-S2": {
    #     "config": "Pastis",
    #     "train_augmentation": Identity(),
    #     "test_augmentation": Identity(),
    #     "modalities": ["s2"],
    #     "scale": 4,
    #     "task_type": "semseg",
    #     "num_workers": 2,
    #     "batch_size": 1,
    #     "max_iter_train": 400,
    #     "max_iter_test": 400,
    #     "semseg_drop_rate": 0.75,
    #     "output_grid": "dense",
    #     "compute_tSNE": True,
    #     "n_full_samples": 4,
    #     "full_samples_modality": 's2',
    #     'overrides': {'classif':False},
    #     'ignore_labels': [19],
    # },
    # "PASTIS-S1": {
    #     "config": "Pastis",
    #     "train_augmentation": Identity(),
    #     "test_augmentation": Identity(),
    #     "modalities": ["s1"],
    #     "scale": 4,
    #     "task_type": "semseg",
    #     "num_workers": 2,
    #     "batch_size": 1,
    #     "max_iter_train": 200,
    #     "max_iter_test": 200,
    #     "semseg_drop_rate": 0.75,
    #     "output_grid": "dense",
    #     "compute_tSNE": False,
    #     "n_full_samples": 4,
    #     "full_samples_modality": 's1',
    #     'overrides': {'classif':False},
    #     'ignore_labels': [19],
    # },
        "so2sat": {
        "config": "So2Sat",
        "train_augmentation": Identity(),
        "test_augmentation": Identity(),
        "modalities": ["s2", "s1"],
        "scale": 8,
        "task_type": "classif",
        "num_workers": 0,
        "batch_size": 8,
        "max_iter_train": 1000,
        "max_iter_test": 1000,
        "semseg_drop_rate": 0,
        "output_grid": "patch",
        "compute_tSNE": False,
        "n_full_samples": 0,
        "full_samples_modality": None,
        'overrides': {'train_dataset.max_samples': 2000,},
        'ignore_labels': [],
    },
}

def train_LR_sklearn(features_train, labels_train, device=None):
    t_LP_start = time.time()
    clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, n_jobs=None, solver=SOLVER)
    clf.fit(features_train, labels_train)
    t_LP_end = time.time()
    def inference(Xte):
        pred_val = clf.predict(Xte)
        return pred_val
    t_LP = t_LP_end - t_LP_start
    return inference, t_LP

def train_LR_torch(features_train: np.ndarray,
                   labels_train: np.ndarray,
                   device: Optional[Union[str, torch.device]] = None):
    """
    Drop-in GPU alternative to train_LR_sklearn.

    Implements multinomial logistic regression (softmax regression) with a
    single linear layer trained on GPU. Uses full-batch LBFGS when feasible
    for rapid convergence; otherwise falls back to Adam with mini-batching.

    Args:
        features_train: (N, D) float numpy array
        labels_train:   (N,)   int numpy array with class indices
        features_val:   (M, D) float numpy array
        device:         torch device or string; if None, tries cuda then cpu

    Returns:
        pred_val: (M,) numpy array of predicted class indices
        t_LP:     float, training + inference elapsed time in seconds
    """
    # Choose device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Convert inputs
    Xtr = torch.from_numpy(features_train).to(device=device, dtype=torch.float32)
    ytr = torch.from_numpy(labels_train).to(device=device, dtype=torch.long)

    N, D = Xtr.shape
    # Compute number of classes from labels
    if ytr.ndim != 1:
        ytr = ytr.view(-1)
    classes = torch.unique(ytr)
    K = int(classes.numel())
    # Map labels to 0..K-1 if not already contiguous
    if not torch.equal(classes, torch.arange(K, device=classes.device)):
        cls2idx = {int(c.item()): i for i, c in enumerate(classes.tolist())}
        ytr = torch.tensor([cls2idx[int(c.item())] for c in ytr], device=device, dtype=torch.long)

    # Model: linear logits XW + b
    model = nn.Linear(D, K, bias=True).to(device)

    # Initialize weights small (helps LBFGS)
    with torch.no_grad():
        nn.init.normal_(model.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.bias)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Regularization similar to sklearn's default C=1.0 (L2)
    # weight_decay approximates L2; scaling differs slightly across optimizers
    C = 1.0
    weight_decay = 1.0 / C

    t_start = time.time()
    # Enable gradients even if called under a no_grad() context
    with torch.enable_grad():
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITER, tolerance_grad=TOL, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xtr)
            loss = criterion(logits, ytr)
            # Manual L2 to match sklearn style regardless of optimizer's wd handling
            l2 = 0.0
            for p in model.parameters():
                l2 = l2 + p.pow(2).sum()
            loss = loss + 0.5 * weight_decay * l2 / N
            loss.backward()
            return loss

        optimizer.step(closure)
    del optimizer, criterion, Xtr, ytr, classes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Inference
    def inference(Xte):
        with torch.no_grad():
            Xte = torch.from_numpy(Xte).to(device=device, dtype=torch.float32)
            logits_val = model(Xte)
            pred_val = torch.argmax(logits_val, dim=-1).detach().cpu().numpy()
        return pred_val

    t_end = time.time()
    t_LP = t_end - t_start
    return inference, t_LP

def detect_and_fix_NAN(x:np.array, array_name:str):
    """
    Detect and fix NaN values in the input array.
    """
    if np.isnan(x).any():
        log.warning("NaN values detected in the input array: {}".format(array_name))
        x[np.isnan(x)] = 0
        print("NaN values replaced with 0.")
    if np.isinf(x).any():
        log.warning("Inf values detected in the input array: {}".format(array_name))
        x[np.isinf(x)] = 0
        print("Inf values replaced with 0.")
    return x

def compute_and_plot_tSNE(features_val, labels_val, dataset_name, compute_tSNE, centroid=None, verbose=False):
    """
    Compute and plot t-SNE visualization.
    """
    if not compute_tSNE:
        return None
    t_tsne_start = time.time()
    # if more than 10k samples, use only 10k
    if features_val.shape[0] > 10000:
        indices = np.random.choice(features_val.shape[0], 10000, replace=False)
        features_val_tsne = features_val[indices]
        labels_val_tsne = labels_val[indices]
    else:
        features_val_tsne = features_val
        labels_val_tsne = labels_val
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, metric='cosine')
    if centroid is not None:
        # Compute t-SNE for centroid
        if centroid.shape[1] != features_val_tsne.shape[1]:
            features_val_tsne = features_val_tsne[:,centroid.shape[1]:]
        features_val_tsne = np.concatenate([features_val_tsne, centroid], axis=0)
    features_val_2d = tsne.fit_transform(features_val_tsne)
    if centroid is not None:
        features_val_2d_centroid = features_val_2d[-centroid.shape[0]:]
        features_val_2d = features_val_2d[:-centroid.shape[0]]
    t_tsne_end = time.time()

    if verbose:
        print(f"t-SNE computation time for {dataset_name}: {t_tsne_end - t_tsne_start:.2f} seconds")

    # Plot t-SNE with labels
    plt.figure(figsize=(10, 8))
    plt.scatter(features_val_2d[:, 0], features_val_2d[:, 1], c=labels_val_tsne, cmap='gist_rainbow', s=1)
    if centroid is not None:
        plt.scatter(features_val_2d_centroid[:, 0], features_val_2d_centroid[:, 1], c='black', marker='x', s=5, label='Centroid')
    plt.title(f"t-SNE Visualization of {dataset_name}")
    fig = plt.gcf()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    if verbose:
        plt.savefig(f"tSNE_{dataset_name}.png", format='png')
    buf.seek(0)
    t_SNE = PIL.Image.open(buf)
    plt.close(fig)

    # plot tSNE with cluster
    if centroid is not None:
        cluster_assignment = np.argmax(features_val_tsne[:-centroid.shape[0]]@centroid.T, axis=1)
        plt.figure(figsize=(10, 8))
        plt.scatter(features_val_2d[:, 0], features_val_2d[:, 1], c=cluster_assignment, cmap='gist_rainbow', s=1, vmin=0, vmax=centroid.shape[0])
        plt.scatter(features_val_2d_centroid[:, 0], features_val_2d_centroid[:, 1], c=np.arange(centroid.shape[0]), cmap='gist_rainbow', marker='*', s=20, label='Centroid')
        plt.title(f"t-SNE Visualization of {dataset_name} with Centroid")
        fig = plt.gcf()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        if verbose:
            plt.savefig(f"tSNE_{dataset_name}_cluster.png", format='png')
        buf.seek(0)
        t_SNE_cluster = PIL.Image.open(buf)
        plt.close(fig)
    else:
        t_SNE_cluster = None
    return t_SNE, t_SNE_cluster

def plot_input_feature_label_pred_grid(data_list, label_cmap='gist_rainbow', dataset_name="", verbose=False):
    """
    Plot a grid where each row contains: input (RGB), feature (PCA to 3D), label, prediction.
    Each element in data_list is [input, feature, label, prediction].
    Returns a PIL image of the plot.
    """
    n_rows = data_list[0].shape[0]
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)  # Ensure axes is 2D

    for i in range(n_rows):
        input_img, feature, label, pred = data_list[0][i], data_list[1][i], data_list[2][i], data_list[3][i]
        # Input: take first 3 channels if more
        if len(input_img.shape) > 3: #Temporal
            input_img = input_img[input_img.shape[0]//2]
        input_img = input_img.transpose(1,2,0)
        if input_img.shape[-1] > 3:
            input_img = input_img[:,:,:3]
        input_img = (input_img - input_img.min()) / (np.ptp(input_img) + 1e-8)
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        # Feature: PCA to 3D, then show as RGB
        if feature.shape[0] > 3:
            h, w = feature.shape[0], feature.shape[1]
            feat_flat = feature.reshape(-1, feature.shape[-1]) # (H*W, C)
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat)
            feat_pca = (feat_pca - feat_pca.min()) / (np.ptp(feat_pca) + 1e-8)
            feat_pca = feat_pca.reshape(h, w, 3)
        else:
            feat_pca = feature
            feat_pca = (feat_pca - feat_pca.min()) / (np.ptp(feat_pca) + 1e-8)
        axes[i, 1].imshow(feat_pca)
        axes[i, 1].set_title("Feature (PCA)")
        axes[i, 1].axis('off')

        # Label
        axes[i, 2].imshow(label, cmap=label_cmap, interpolation='nearest')
        axes[i, 2].set_title("Label")
        axes[i, 2].axis('off')

        # Prediction.shape)
        axes[i, 3].imshow(pred, cmap=label_cmap, interpolation='nearest')
        axes[i, 3].set_title("Prediction")
        axes[i, 3].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = PIL.Image.open(buf)
    if verbose:
        plt.savefig(f"input_feature_label_pred_grid_{dataset_name}.png", format='png')
    plt.close(fig)
    return img

@torch.no_grad()
def train_LP(dataloader_train,
             dataloader_val,
             dataset_name,
             scale,
             type,
             wavelengths,
             input_res,
             latent_grid,
             output_grid,
             subpatches,
             model,
             device,
             max_iter_train,
             max_iter_test,
             semseg_drop_rate,
             compute_tSNE,
             centroid=None,
             ignore_labels=[],
             labels_map=None,
             n_full_samples=0,
             full_samples_modality=None,
             verbose=True):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    if type == 'semseg':
        output_type= 'dense'
    elif type == 'classif':
        output_type= 'tile'

    model.eval()
    semseg_drop_in_sampler = type == 'semseg' and semseg_drop_rate > 0

    def sample_data(dataloader, max_iter, mode, n_full_samples=0, full_samples_modality=None):
        features_list, labels_list = [], []
        full_samples=[[],[],[]]
        if verbose:
            print(f"Sampling {mode} set")
        for i, batch in enumerate(dataloader):
            if verbose:
                print(f"{i}/{max_iter}", end='\r')
            label = batch.pop('label')
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            if hasattr(model, 'forward_release'):
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model.forward_release(batch, scale, output=output_type, output_modality='')
                if output_type == 'dense':
                    out = out.flatten(1,2)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out, _ = model.forward(batch, wavelengths, input_res, scale, latent_grid, output_grid, subpatches, keep_subpatch=False)
                if type == 'classif':
                    out = out[:,model.n_registers:,:].mean(dim=1)
                else:
                    out = out[:,model.n_registers:,:]
            if i == 0 and verbose:
                print("First batch output shape:")
                print(out.shape)
                print(label.shape)
            #sample full samples
            if n_full_samples > 0 and full_samples_modality is not None:
                assert full_samples_modality in batch.keys()
                #input
                full_samples[0].append(batch[full_samples_modality].cpu().numpy())
                #features
                grid_size = math.ceil(math.sqrt(out.shape[1]))
                features_2d = out.reshape(out.shape[0], grid_size, grid_size, out.shape[-1])

                if grid_size == 32:
                    features_2d = F.interpolate(features_2d.permute(0, 3, 1, 2),
                                        scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False
                                    ).permute(0,2,3,1)

                features_2d = features_2d.cpu().numpy()
                full_samples[1].append(features_2d)
                #labels
                full_samples[2].append(label.cpu().numpy())
                n_full_samples -= label.shape[0]

            if type == 'semseg':
                B, N, D = out.shape
                num_patches = int(N**(1/2))

                out = out.view(B, N, 1, D).permute(0, 2, 1, 3)
                out = out.view(B, 1, num_patches, num_patches, D).flatten(0, 1)

                if num_patches == 32:
                    out = F.interpolate(out.permute(0, 3, 1, 2),
                                        scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False
                                    ).permute(0,2,3,1)

                out = out.flatten(1, 2)
                label = label.flatten(1, 2)

                if mode == "train" and semseg_drop_in_sampler:
                    keep_mask = torch.rand(label.shape, device=label.device) > semseg_drop_rate
                    out = out[keep_mask.to(out.device)]
                    label = label[keep_mask]

            features_list.append(out.cpu())
            labels_list.append(label.cpu())
            if i >= max_iter:
                break
        return features_list, labels_list, full_samples

    def predict_data(dataloader, max_iter, n_full_samples=0, full_samples_modality=None):
        pred_list, labels_list = [], []
        tsne_features, tsne_labels = [], []
        full_samples=[[],[],[]]
        tsne_remaining = 10000 if compute_tSNE else 0
        for i, batch in enumerate(dataloader):
            if verbose:
                print(f"{i}/{max_iter}", end='\r')
            label = batch.pop('label')
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            if hasattr(model, 'forward_release'):
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model.forward_release(batch, scale, output=output_type, output_modality='')
                if output_type == 'dense':
                    out = out.flatten(1,2)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out, _ = model.forward(batch, wavelengths, input_res, scale, latent_grid, output_grid, subpatches, keep_subpatch=False)
                if type == 'classif':
                    out = out[:,model.n_registers:,:].mean(dim=1)
                else:
                    out = out[:,model.n_registers:,:]

            if n_full_samples > 0 and full_samples_modality is not None:
                assert full_samples_modality in batch.keys()
                full_samples[0].append(batch[full_samples_modality].cpu().numpy())
                grid_size = math.ceil(math.sqrt(out.shape[1]))
                features_2d = out.reshape(out.shape[0], grid_size, grid_size, out.shape[-1])
                if grid_size == 32:
                    features_2d = F.interpolate(features_2d.permute(0, 3, 1, 2),
                                        scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False
                                    ).permute(0,2,3,1)
                full_samples[1].append(features_2d.cpu().numpy())
                full_samples[2].append(label.cpu().numpy())
                n_full_samples -= label.shape[0]

            if type == 'semseg':
                B, N, D = out.shape
                num_patches = int(N**(1/2))
                out = out.view(B, N, 1, D).permute(0, 2, 1, 3)
                out = out.view(B, 1, num_patches, num_patches, D).flatten(0, 1)
                if num_patches == 32:
                    out = F.interpolate(out.permute(0, 3, 1, 2),
                                        scale_factor=4,
                                        mode='bilinear',
                                        align_corners=False
                                    ).permute(0,2,3,1)
                out = out.flatten(1, 2)
                label = label.flatten(1, 2)

            features = out.cpu().numpy()
            labels = label.cpu().numpy()
            if len(ignore_labels)>0:
                mask = np.isin(labels, ignore_labels, invert=True)
                features = features[mask]
                labels = labels[mask]
            features = detect_and_fix_NAN(features, f"features_val_{dataset_name}_{i}")
            labels = detect_and_fix_NAN(labels, f"labels_val_{dataset_name}_{i}")

            if type == 'classif':
                while len(labels.shape)>1:
                    labels = np.argmax(labels, axis=1)
            elif type == 'semseg':
                if len(labels.shape)==3:
                    pos = labels.shape[2]//2
                    labels = labels[:,:,pos]
                features = features.reshape(-1, features.shape[-1])
                labels = labels.reshape(-1)

            if labels_map is not None:
                labels = np.vectorize(labels_map.get)(labels)

            features = (features - mean) / std
            preds = inference(features)
            pred_list.append(preds)
            labels_list.append(labels)

            if tsne_remaining > 0:
                take = min(tsne_remaining, features.shape[0])
                tsne_features.append(features[:take])
                tsne_labels.append(labels[:take])
                tsne_remaining -= take

            if i >= max_iter:
                break

        pred_val = np.concatenate(pred_list, axis=0)
        labels_val = np.concatenate(labels_list, axis=0)
        if tsne_features:
            features_val_tsne = np.concatenate(tsne_features, axis=0)
            labels_val_tsne = np.concatenate(tsne_labels, axis=0)
        else:
            features_val_tsne = np.empty((0, mean.shape[0]), dtype=mean.dtype)
            labels_val_tsne = np.empty((0,), dtype=labels_val.dtype)
        return pred_val, labels_val, features_val_tsne, labels_val_tsne, full_samples


    if verbose:
        print("Start evaluating on", dataset_name)
    t_sample_start = time.time()
    features_train, labels_train, _ = sample_data(dataloader_train, max_iter_train, "train", n_full_samples=0)
    t_train_sample_end = time.time()
    if verbose:
        print("End sampling, Training Linear regression")
    features_train = torch.cat(features_train, dim=0).numpy()
    labels_train = torch.cat(labels_train, dim=0).numpy()

    #remove ignore labels
    if len(ignore_labels)>0:
        mask = np.isin(labels_train, ignore_labels, invert=True)
        features_train = features_train[mask]
        labels_train = labels_train[mask]


    # Detect and fix NaN values
    features_train = detect_and_fix_NAN(features_train, f"features_train_{dataset_name}")
    labels_train = detect_and_fix_NAN(labels_train, f"labels_train_{dataset_name}")

    if type == 'classif':
        while len(labels_train.shape)>1:
            labels_train = np.argmax(labels_train, axis=1)
    elif type == 'semseg':
        if len(labels_train.shape)==3:
            #patch case
            pos = labels_train.shape[2]//2
            labels_train = labels_train[:,:,pos]
        # Flatten the features and labels
        features_train = features_train.reshape(-1, features_train.shape[-1])
        labels_train = labels_train.reshape(-1)

        # Select semseg_drop_rate of the training pixel to drop
        if semseg_drop_rate > 0 and not semseg_drop_in_sampler:
            keep_mask = np.random.rand(*labels_train.shape) > semseg_drop_rate
            labels_train = labels_train[keep_mask]
            features_train = features_train[keep_mask]

    #normalize features
    mean = np.mean(features_train, axis=0)
    std = np.std(features_train, axis=0)

    if labels_map is not None:
        labels_train = np.vectorize(labels_map.get)(labels_train)

    features_train = (features_train - mean) / std

    if verbose:
        print(f"{features_train.shape} train samples")

    # inference, t_LP = train_LR_sklearn(features_train, labels_train, device=device)
    inference, t_LP = train_LR_torch(features_train, labels_train, device=device)
    del features_train, labels_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_val_sample_start = time.time()
    pred_val, labels_val, features_val_tsne, labels_val_tsne, full_samples_val = predict_data(
        dataloader_val,
        max_iter_test,
        n_full_samples=n_full_samples,
        full_samples_modality=full_samples_modality,
    )
    t_val_sample_end = time.time()
    t_sample_end = t_sample_start + (t_train_sample_end - t_sample_start) + (t_val_sample_end - t_val_sample_start)
    full_samples_val[1] = detect_and_fix_NAN(np.array(full_samples_val[1]), f"full_samples_features_val_{dataset_name}")

    if verbose:
        print(f"{labels_val.shape} validation samples")

    out = {}
    if compute_tSNE:
        t_SNE_label, t_SNE_cluster = compute_and_plot_tSNE(
            features_val_tsne, labels_val_tsne, dataset_name, compute_tSNE, centroid=centroid, verbose=verbose
        )
        if t_SNE_label is not None:
            out['tSNE'] = t_SNE_label
        if t_SNE_cluster is not None:
            out['tSNE_cluster'] = t_SNE_cluster
    if n_full_samples:
        full_samples_val = [np.concatenate(x, axis=0) for x in full_samples_val] # [input, feature, label]
        # compute prediction from features
        features_full = full_samples_val[1].reshape(-1, full_samples_val[1].shape[-1])
        # normalize with the same stats
        features_full = (features_full - mean) / std
        pred_full = inference(features_full)
        pred_full = pred_full.reshape(full_samples_val[1].shape[:3])
        full_samples_val.append(pred_full)

        # plot input, feature, label, prediction
        out['samples'] = plot_input_feature_label_pred_grid(full_samples_val, label_cmap='gist_rainbow', dataset_name=dataset_name, verbose=verbose)


    if type == 'classif':
        acc = accuracy_score(labels_val, pred_val)
        f1 = f1_score(labels_val, pred_val, average='macro')

        if verbose:
            print("End of evaluation on", dataset_name)
        out.update({'accuracy': acc, 'f1_score': f1, 'time_LP': t_LP, 'time_sample': t_sample_end - t_sample_start})

    elif type == 'semseg':
        acc = accuracy_score(labels_val, pred_val)
        jaccard_score_val = jaccard_score(labels_val, pred_val, average='macro')

        if verbose:
            print("End of evaluation on", dataset_name)
        out.update({
            'accuracy': acc,
            'jaccard_score': jaccard_score_val,
            'time_LP': t_LP,
            'time_sample': t_sample_end - t_sample_start,
        })
    return out

def eval_model_FT(datasets_config, model, device, centroid=None, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    # L.seed_everything(42, workers=True)
    metrics= {}
    for dataset in datasets_config:

        #clean memory
        torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            print(f"Evaluating on dataset: {dataset['out_name']}")

        dataloader_train = dataset['train_dataloader']
        dataloader_val = dataset['val_dataloader']
        assert dataset['output_grid'] in ['dense','patch']
        output_grid = dataset['num_patches'] if dataset['output_grid'] == 'dense' else dataset['num_patches']//dataset['scale']**2
        params = {
            'dataloader_train': dataloader_train,
            'dataloader_val': dataloader_val,
            'dataset_name': dataset['name'],
            'scale': dataset['scale'],
            'wavelengths': WAVELENGTHS,
            'input_res': INPUT_RES,
            'latent_grid': dataset['num_patches']//dataset['scale']**2,
            'output_grid': output_grid,
            'subpatches': SUBPATCHES,
            'model': model,
            'type': dataset['task_type'],
            'device': device,
            'max_iter_train': dataset['max_iter_train'],
            'max_iter_test': dataset['max_iter_test'],
            'semseg_drop_rate': dataset['semseg_drop_rate'],
            'labels_map': dataset['labels_map'],
            'compute_tSNE': dataset['compute_tSNE'],
            'n_full_samples': dataset['n_full_samples'],
            'full_samples_modality': dataset['full_samples_modality'],
            'centroid': centroid,
            'ignore_labels': dataset['ignore_labels'],
            'verbose': verbose
        }
        try:
            if USE_TIMEOUT:
            # Use Timeout to limit the execution time
                with Timeout(TIMEOUT) as timeout_manager:
                    result = train_LP(**params)
                if timeout_manager.state == Timeout.TIMED_OUT:
                    print(f"Timeout occurred during evaluation on {dataset['out_name']}")
                    result = {'time_LP': np.NAN, 'time_sample':np.NAN}
            else:
            # If Timeout is not available, run without timeout
                result = train_LP(**params)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA out of memory during evaluation on {dataset['out_name']}: {e}")
                result = {'time_LP': np.nan, 'time_sample': np.nan}
            else:
                raise
        except Exception as e:
            result = {}
            print(f"Error during evaluation on {dataset['out_name']}: {e}")
            raise e
        metrics.update({f"{dataset['out_name']}_{k}": v for k, v in result.items()})
        if verbose:
            print(f"Results for {dataset['out_name']}: {result}")

    return metrics


def resolve_lp_eval_model(module):
    if hasattr(module, 'model') and hasattr(module.model, 'encoder'):
        return module.model.encoder
    if hasattr(module, 'model') and hasattr(module.model, 'model'):
        return module.model.model
    if hasattr(module, 'target_encoder'):
        return module.target_encoder
    raise ValueError(f"Unable to resolve LP evaluation model from module {type(module).__name__}")

def eval_model_from_path(model_path, model_config, device = 'cuda', overwrite_data_dir=None, verbose=False):
    """
    Evaluate the model on the validation set using linear probing(sklearn).
    """
    OmegaConf.register_new_resolver("eval", eval)

    config = OmegaConf.load(model_config)
    module = hydra.utils.instantiate(config['model']['instance'])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    centroid = None
    if 'target_head.layer.weight' in model_state:
        centroid = model_state['target_head.layer.weight'].detach().cpu().numpy()

    missing, unexpected = module.load_state_dict(model_state, strict=False)
    if verbose and (missing or unexpected):
        print(f"Missing keys while loading LP eval module: {missing}")
        print(f"Unexpected keys while loading LP eval module: {unexpected}")

    model = resolve_lp_eval_model(module)
    model.to(device)

    if overwrite_data_dir is not None:
        data_dir = overwrite_data_dir
    elif 'paths' in config and 'data_dir' in config['paths']:
        data_dir = config['paths']['data_dir']
    else:
        data_dir = config.get('data_dir')

    if data_dir is None:
        raise ValueError(f"Unable to resolve data_dir from config {model_config}")

    data_dir = str(Path(data_dir))
    list_dataconfig = get_dataset_config(data_dir=data_dir, verbose=verbose)
    metrics = eval_model_FT(list_dataconfig, model, device, centroid=centroid, verbose=verbose)
    return metrics

def get_dataset_config_metadata(data_dir):
    """Lightweight view of ``EVAL_DATASETS`` for LP-eval orchestration.

    Returns a list of small dicts with just the fields needed for load
    balancing (``max_iter_train``, ``max_iter_test``, ``batch_size``,
    ``modalities``, ``compute_tSNE``) and identification (``out_name``).

    Does NOT instantiate datasets or dataloaders, so callers can decide
    which items to materialize per rank instead of building every one on
    every rank (which otherwise pins 8x train+val+test dataset objects and
    their worker processes in RAM and routinely OOM-kills the node during
    ``on_validation_epoch_end``).
    """
    metadata = []
    for dataset_name, dataset_config in EVAL_DATASETS.items():
        metadata.append({
            "out_name": dataset_name,
            "modalities": list(dataset_config.get("modalities", [])),
            "batch_size": int(dataset_config.get("batch_size", 1)),
            "max_iter_train": int(dataset_config.get("max_iter_train", 0) or 0),
            "max_iter_test": int(dataset_config.get("max_iter_test", 0) or 0),
            "compute_tSNE": bool(dataset_config.get("compute_tSNE", False)),
            "task_type": dataset_config.get("task_type"),
        })
    return metadata


def build_dataset_config(dataset_name, data_dir, verbose=False):
    """Build the full hydra-instantiated config + dataloaders for a single
    ``EVAL_DATASETS`` entry.

    Split out of :func:`get_dataset_config` so callers can materialize one
    dataset at a time (and release it between datasets) rather than keeping
    all of them alive for the whole LP eval pass.
    """
    if dataset_name not in EVAL_DATASETS:
        raise KeyError(
            f"{dataset_name!r} is not in EVAL_DATASETS "
            f"(known: {list(EVAL_DATASETS.keys())})."
        )
    dataset_config = EVAL_DATASETS[dataset_name]
    if verbose:
        print(f"Configuring dataset: {dataset_name}")

    dataset_config_path = f"configs/dataset/{dataset_config['config']}.yaml"
    dataconfig = OmegaConf.load(dataset_config_path)
    dataconfig['modalities'] = dataset_config['modalities']
    dataconfig['scale'] = dataset_config['scale']
    dataconfig['data_dir'] = data_dir + "/${dataset.name}/"
    dataconfig['paths'] = {"data_dir": data_dir + "/"}
    dataconfig['out_name'] = dataset_name

    dataconfig['num_workers'] = dataset_config['num_workers']
    dataconfig['batch_size'] = dataset_config['batch_size']
    dataconfig['max_iter_train'] = dataset_config['max_iter_train']
    dataconfig['max_iter_test'] = dataset_config['max_iter_test']
    dataconfig['semseg_drop_rate'] = dataset_config['semseg_drop_rate']
    dataconfig['output_grid'] = dataset_config['output_grid']
    dataconfig['compute_tSNE'] = dataset_config['compute_tSNE'] if 'compute_tSNE' in dataset_config else False
    dataconfig['ignore_labels'] = dataset_config['ignore_labels'] if 'ignore_labels' in dataset_config else []
    dataconfig['n_full_samples'] = dataset_config['n_full_samples'] if 'n_full_samples' in dataset_config else 0
    dataconfig['full_samples_modality'] = dataset_config['full_samples_modality'] if 'full_samples_modality' in dataset_config else None

    for k, v in dataset_config['overrides'].items():
        ks = k.split('.')
        target = dataconfig
        for k in ks[:-1]:
            target = target[k]
        target[ks[-1]] = v

    dataconfig['train_dataset']['transform'] = None
    dataconfig['val_dataset']['transform'] = None
    dataconfig['test_dataset']['transform'] = None
    dataconfig['dataset'] = dataconfig
    dataconfig = OmegaConf.to_container(dataconfig, resolve=True)
    dataconfig['labels_map'] = dataset_config['labels_map'] if 'labels_map' in dataset_config else None
    dataconfig['train_dataset']['transform'] = dataset_config['train_augmentation']
    dataconfig['val_dataset']['transform'] = dataset_config['test_augmentation']
    dataconfig['test_dataset']['transform'] = dataset_config['test_augmentation']
    dataconfig['task_type'] = dataset_config['task_type']

    datamodule = DataModule(
        train_dataset=hydra.utils.instantiate(dataconfig['train_dataset']),
        val_dataset=hydra.utils.instantiate(dataconfig['val_dataset']),
        test_dataset=hydra.utils.instantiate(dataconfig['test_dataset']),
        global_batch_size=dataconfig['batch_size'],
        num_nodes=1,
        num_devices=1,
        num_workers=dataconfig['num_workers'],
        verbose=verbose,
    )
    datamodule.setup()
    dataconfig['train_dataloader'] = datamodule.train_dataloader()
    dataconfig['val_dataloader'] = datamodule.val_dataloader()
    # Keep the datamodule reachable so its datasets aren't GC'd while the
    # dataloaders are in use; callers should drop the whole dataconfig to
    # release everything together.
    dataconfig['_datamodule'] = datamodule
    return dataconfig


def get_dataset_config(data_dir, verbose=False):
    """Back-compat shim: build every dataset eagerly.

    Kept for scripts like :func:`eval_model_from_path` that run outside the
    distributed training loop. For the DDP LP-eval path prefer
    :func:`get_dataset_config_metadata` + :func:`build_dataset_config`.
    """
    return [build_dataset_config(name, data_dir, verbose=verbose)
            for name in EVAL_DATASETS.keys()]

if __name__ == "__main__":
    model_config = "logs/JZ/train/runs/20250501-13:54-GEOm-Ut-Capi-16gpu/0_/.hydra/config.yaml"
    model_path = "logs/JZ/train/runs/20250501-13:54-GEOm-Ut-Capi-16gpu/0_/checkpoints/last.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = '/home/yperron/code/AnySat/data/'
    metrics = eval_model_from_path(model_path, model_config, device, overwrite_data_dir=data_dir, verbose=True)
    print(metrics)
