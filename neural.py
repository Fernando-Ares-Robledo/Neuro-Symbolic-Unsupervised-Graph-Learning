# neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np
import optuna
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
import pandas as pd
import time
import gc
import config
from sklearn.preprocessing import StandardScaler


from utiles import *
from plots import *


feature_scaler = StandardScaler()



def prepare_graph(
    G: nx.Graph,
    *,
    scaler: Optional[StandardScaler] = None,
    descriptor_vector: Optional[np.ndarray] = None,
    drop_attrs: Tuple[str, ...] = ("Timestamp",),
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    nodes = list(G.nodes())
    n = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)

    if add_self_loops:
        A = A + np.eye(n, dtype=np.float32)

    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + eps))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    numeric_keys = []
    if n > 0:
        attrs0 = G.nodes[nodes[0]]
        numeric_keys = [k for k, v in attrs0.items() if isinstance(v, (int, float)) and k not in drop_attrs]
    if numeric_keys:
        X_raw = np.zeros((n, len(numeric_keys)), dtype=np.float32)
        for i, node in enumerate(nodes):
            for j, k in enumerate(numeric_keys):
                X_raw[i, j] = float(G.nodes[node].get(k, 0.0))
    else:
        X_raw = np.ones((n, 1), dtype=np.float32)

    if descriptor_vector is not None:
        desc = np.asarray(descriptor_vector, dtype=np.float32).reshape(1, -1)
        X_raw = np.hstack([X_raw, np.tile(desc, (n, 1))])

    if scaler is not None:
        if descriptor_vector is not None:
            node_dim = X_raw.shape[1] - len(descriptor_vector)
            X_node = X_raw[:, :node_dim]
            desc_part = X_raw[:, node_dim:]
            X_node_scaled = scaler.transform(X_node)
            X_out = np.hstack([X_node_scaled, desc_part])
        else:
            X_out = scaler.transform(X_raw)
    else:
        X_out = X_raw

    return torch.from_numpy(A_norm), torch.from_numpy(X_out)


def fit_feature_scaler(
    raw_graphs: List[Dict],
    *,
    drop_attrs: Tuple[str, ...] = ("Timestamp",),
    use_descriptors: bool = False,
) -> StandardScaler:
    """
    Parameters
    ----------
    raw_graphs : list[dict]

    drop_attrs : tuple[str], default ("Timestamp",)
    use_descriptors : bool, default False
     
    Returns
    -------
    scaler : sklearn.preprocessing.StandardScaler
    """
    feature_blocks: List[np.ndarray] = []

    for i, entry in enumerate(raw_graphs):
        G: nx.Graph = entry["graph"]

        desc_vec: Optional[np.ndarray] = None
        if use_descriptors and "descriptors" in entry:
            desc = entry["descriptors"]
            desc_vec = np.array([float(desc[k]) for k in sorted(desc)], dtype=np.float32)

        _, X = prepare_graph(
            G,
            scaler=None,
            descriptor_vector=desc_vec,
            drop_attrs=drop_attrs,
        )
        feature_blocks.append(X.numpy())          

        del G, X, desc_vec
        # if (i + 1) % 50 == 0:                     
        #     gc.collect()

    X_all = np.vstack(feature_blocks)
    scaler = StandardScaler()
    scaler.fit(X_all)

    del X_all, feature_blocks
    # gc.collect()
    return scaler


# ---------- 2. Encoder ----------
class Encoder(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, num_layers:int=2, activation:str='relu', dropout:float=0.0):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim, bias=False)
                                     for i in range(num_layers)])
        self.act = nn.ReLU() if activation=='relu' else (nn.Tanh() if activation=='tanh' else nn.LeakyReLU(0.1))
        self.drop = nn.Dropout(dropout)
    def forward(self, A_norm, X):
        H = X
        for lin in self.layers:
            H = A_norm @ H
            H = lin(H)
            H = self.act(H)
            H = self.drop(H)
        return H


# ---------- 3. Variational Layer ----------
class VarLatent(nn.Module):
    def __init__(self, hidden_dim:int, latent_dim:int):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, H):
        mu = self.fc_mu(H)
        logvar = self.fc_logvar(H)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        Z = mu + eps*std
        return Z, mu, logvar


# ---------- 4. Decoder ----------
class Decoder(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, Z):
        return torch.sigmoid(Z @ Z.t())


# ---------- 5. VGAE  ----------
class VGAE(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int=64, latent_dim:int=64,
                 num_layers:int=2, activation:str='relu', dropout:float=0.0):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, num_layers, activation, dropout)
        self.varlat  = VarLatent(hidden_dim, latent_dim)
        self.decoder = Decoder()
        self.cls = nn.Linear(latent_dim, 1)
    def encode(self, A_norm, X):
        H = self.encoder(A_norm, X)
        Z, mu, logvar = self.varlat(H)
        return Z, mu, logvar
    def decode(self, Z):
        return self.decoder(Z)
    def forward(self, A_norm, X):
        Z, mu, logvar = self.encode(A_norm, X)
        A_hat = self.decode(Z)
        zG = Z.mean(dim=0)  
        p = torch.sigmoid(self.cls(zG)).squeeze()
        return A_hat, mu, logvar, p


def vgae_loss(
    A_true : torch.Tensor,  
    A_hat  : torch.Tensor,   
    mu     : torch.Tensor,   
    logvar : torch.Tensor,   
    beta   : float = 1.0,
    eps    : float = 1e-10,
) -> torch.Tensor:
    """
    vgae loss
    
    Parameters
    ----------
    A_true : Tensor  (n × n)  
    A_hat  : Tensor  (n × n) )
    mu     : Tensor  (n × z)  
    logvar : Tensor  (n × z)  
    beta   : float            
    eps    : float            

    Returns
    -------
    loss : torch.Tensor (escalar)
    """
    # -------- 1. Binary‑Cross‑Entropy ------------------
    A_hat_clamped = torch.clamp(A_hat, eps, 1.0 - eps)
    bce = F.binary_cross_entropy(A_hat_clamped, A_true, reduction="mean")

    # -------- 2.  KL variacional --------------------------
    kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.mean(torch.sum(kl_per_node, dim=1))  

    # -------- 3. loss ---------------------------------------
    loss = bce + beta * kl

    del A_hat_clamped, kl_per_node
    return loss

def total_loss(A_true, A_hat, mu, logvar, p_hat, y_sym, beta, gamma):
    A_hat_clamped = torch.clamp(A_hat, eps, 1.0 - eps)
    bce = F.binary_cross_entropy(A_hat_clamped, A_true, reduction="mean")
    kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.mean(torch.sum(kl_per_node, dim=1))
    ce  = F.binary_cross_entropy(p_hat, y_sym.float())
    return bce + beta*kl + gamma*ce

# ──────────────────────────────────────────────────────────────────────
def reconstruction_error(A: torch.Tensor, A_hat: torch.Tensor) -> float:
    err = torch.norm(A - A_hat, p="fro").item()
    del A, A_hat  
    return err


# ──────────────────────────────────────────────────────────────────────
def latent_distance(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    d = mu.norm(p=2, dim=1).mean().item()
    del mu, logvar
    return d


# ──────────────────────────────────────────────────────────────────────
def score_normalizer(raw_errors: list[float]) -> tuple[float, float]:
    arr = np.asarray(raw_errors, dtype=np.float32)
    mu_err = float(arr.mean())
    sigma_err = float(arr.std(ddof=0) + 1e-12)   # 
    del arr
    # gc.collect()
    return mu_err, sigma_err


# ──────────────────────────────────────────────────────────────────────
def neural_score(
    err: float,
    mu_err: float,
    sigma_err: float,
    *,
    alpha: float = 0.3,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> float:
    z = (err - mu_err) / sigma_err
    s = 0.5 * (1.0 + np.tanh(alpha * z))
    s_clipped = float(np.clip(s, clip_min, clip_max))
    return s_clipped


def mcc(
    pred_scores: list[float],
    labels      : list[int] | list[str],
    thr         : float,
) -> float:
    """
    Parameters
    ----------
    pred_scores : list[float]
    labels      : list[int] | list[str]
    thr         : float
        
    Returns
    -------
    mcc_value : float
    """
    preds = np.array([1 if s >= thr else 0 for s in pred_scores], dtype=np.int8)

    if isinstance(labels[0], str):
        y_true = np.array([0 if str(l).lower() == "normal" else 1 for l in labels],
                          dtype=np.int8)
    else:
        y_true = np.asarray(labels, dtype=np.int8)

    mcc_value = matthews_corrcoef(y_true, preds)

    del preds, y_true
    gc.collect()
    return float(mcc_value)
