# dataset.py

import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from symbolic import *
from neural import *
class GraphDataset(Dataset):
    """
    returns por muestra:
      A_norm   : Tensor (n×n)
      X        : Tensor (n×d)
      y_sym    : Tensor scalar 
      y_real   : Tensor scalar 
      gid      : int
    """
    def __init__(
        self,
        raw_graphs: List[Dict[str, Any]],
        scaler: StandardScaler,
        rule_stats: Dict[str, Tuple[float, float]],
        k_sigma: float,
        tau: float,
        *,
        drop_attrs: Tuple[str, ...] = ("Timestamp",),
        include_descriptors: bool = False,
    ):
        self.raw_graphs        = raw_graphs
        self.scaler            = scaler
        self.rule_stats        = rule_stats
        self.k_sigma           = k_sigma
        self.tau               = tau
        self.drop_attrs        = drop_attrs
        self.include_descriptors = include_descriptors

    def __len__(self) -> int:
        return len(self.raw_graphs)

    def __getitem__(self, idx: int):
        entry = self.raw_graphs[idx]
        G: nx.Graph = entry["graph"]

        desc_vec: Optional[np.ndarray] = None
        if self.include_descriptors and "descriptors" in entry:
            raw_desc = entry["descriptors"]
            num_items = [(k, v) for k, v in raw_desc.items() if isinstance(v,(int,float))]
            if num_items:
                keys = sorted(k for k,_ in num_items)
                desc_vec = np.array([float(raw_desc[k]) for k in keys], dtype=np.float32)

        deltas = []
        for key,(mu,sigma) in sorted(self.rule_stats.items()):
            x = entry.get("descriptors",{}).get(key, 0.0)
            delta = compute_deviation(float(x), mu, sigma, self.k_sigma)
            deltas.append(delta)
        s_sym = probabilistic_s_norm(deltas)
        y_sym = 1 if s_sym > self.tau else 0

        rule_vec = np.array(deltas + [s_sym], dtype=np.float32)
        full_vec  = np.hstack([desc_vec, rule_vec]) if desc_vec is not None else rule_vec


        A_norm, X = prepare_graph(
            G,
            scaler            = self.scaler,
            descriptor_vector = full_vec,
            drop_attrs        = self.drop_attrs,
        )


        cat    = str(entry.get("cat","")).lower()
        y_real = 0 if cat=="normal" else 1
        gid = idx

        return A_norm, X, torch.tensor(y_sym,dtype=torch.float32), torch.tensor(y_real,dtype=torch.long), gid


def collate_small_graphs(batch):
    A_list, X_list, y_sym_list, y_real_list, ids = [],[],[],[],[]
    for A,X,y_sym,y_real,gid in batch:
        A_list.append(A);  X_list.append(X)
        y_sym_list.append(y_sym);  y_real_list.append(y_real)
        ids.append(gid)
    y_sym_tensor  = torch.stack(y_sym_list)   
    y_real_tensor = torch.stack(y_real_list)  
    return A_list, X_list, y_sym_tensor, y_real_tensor, ids
