# symbolic.py
import logging
import math
import gc
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm
import time
import config
from utiles import record_memory_usage

def compute_descriptors_statistics(graphs: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    descriptor_values: Dict[str, List[float]] = {}
    for graph in graphs:
        descriptors = graph.get("descriptors", {})
        for key, value in descriptors.items():
            if isinstance(value, (int, float)):
                descriptor_values.setdefault(key, []).append(float(value))
    statistics: Dict[str, Tuple[float, float]] = {}
    for key, values in descriptor_values.items():
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        statistics[key] = (mean_value, std_value)
        logging.info("Descriptor '%s': mean = %.4f, std = %.4f", key, mean_value, std_value)
    del descriptor_values, values, key, mean_value, std_value
    gc.collect()
    return statistics

def compute_deviation(x: float, mu: float, sigma: float, k: float) -> float:
    if sigma == 0:
        return 0.0
    raw_value = (abs(x - mu) - k * sigma) / sigma
    delta = min(1, max(0.0, raw_value))
    return delta
# def compute_descriptors_statistics(graphs):
#     descriptor_values = {}
#     for g in graphs:
#         for k, v in g.get("descriptors", {}).items():
#             if isinstance(v, (int, float)):
#                 descriptor_values.setdefault(k, []).append(float(v))

#     stats = {}
#     for key, vals in descriptor_values.items():
#         vals = np.asarray(vals, dtype=np.float32)
#         med  = np.median(vals)                         
#         mad  = np.median(np.abs(vals - med))           
#         stats[key] = (med, mad)
#     return stats


# def compute_deviation(x, med, mad, k):
#     if mad == 0:
#         return 0.0
#     raw = (abs(x - med) - k * mad) / mad
#     return min(1.0, max(0.0, raw))

def compute_deltas_list(
    stats: Dict[str, Tuple[float, float]],
    x_values: List[float],
    k: float
) -> List[float]:
    if len(x_values) != len(stats):
        raise ValueError(
            f"Mismatch in descriptors: {len(x_values)} values vs {len(stats)} stats"
        )
    deltas: List[float] = []
    for (descriptor, (mu, sigma)), x in zip(stats.items(), x_values):
        delta = compute_deviation(x, mu, sigma, k)
        deltas.append(delta)
    return deltas


def probabilistic_s_norm(deltas: List[float]) -> float:
    prod = math.prod([(1 - delta) for delta in deltas])
    s_norm = 1 - prod
    return s_norm


def compute_s_norm_scores(
    train_graphs: List[Dict[str, Any]],
    test_graphs: List[Dict[str, Any]],
    k: float
) -> List[float]:
    stats = compute_descriptors_statistics(train_graphs)
    s_norm_scores: List[float] = []
    pbar = tqdm(total=len(test_graphs), desc="Computing s-norm scores", unit="graph")
    if config.start_time is None:
        config.start_time = time.time()
    for i, graph in enumerate(test_graphs):
        x_values: List[float] = []
        descriptors = graph.get("descriptors", {})
        for key in stats.keys():
            val = descriptors.get(key, 0.0)
            x_values.append(float(val) if isinstance(val, (int, float)) else 0.0)
        deltas = compute_deltas_list(stats, x_values, k)
        s_norm = probabilistic_s_norm(deltas)
        s_norm_scores.append(s_norm)
        pbar.set_postfix({})
        pbar.update(1)
    pbar.close()
    return s_norm_scores


def predict_s_norm_labels(
    train_graphs: List[Dict[str, Any]],
    eval_graphs: List[Dict[str, Any]],
    k: float,
    threshold: float
) -> Tuple[List[str], List[float], List[str]]:
    s_norm_scores = compute_s_norm_scores(train_graphs, eval_graphs, k)
    predicted_labels: List[str] = []
    true_labels: List[str] = []
    for graph, s_score in zip(eval_graphs, s_norm_scores):
        predicted_labels.append("Normal" if s_score < threshold else "Anomaly")
        true_labels.append(graph.get("cat", "Unknown"))
    return predicted_labels, s_norm_scores, true_labels