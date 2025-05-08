#data.py
import logging
import gc
import random
from typing import Tuple, List, Dict, Any
import time
import pandas as pd
import pickle 
from tqdm import tqdm
from utiles import *
import networkx as nx
import numpy as np

def load_graphs_info_from_pickle(filename: str) -> List[Dict[str, Any]]:
    """
    Loads the list of dictionaries 'graphs_info' from a pickle file.
    
    Parameters:
        filename (str): Path to the pickle file.
        
    Returns:
        List[Dict[str, Any]]: The list of dictionaries 'graphs_info'.
    """
    with open(filename, 'rb') as f:
        graphs_info = pickle.load(f)
    print(f"Graphs loaded from {filename}")
    return graphs_info
def partition_graphs(
    pkl_file: str,
    train: int,
    test: int,
    validacion: int,
    min_normals_test: int = 1,
    min_normals_valid: int = 1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partitions graphs into train/test/validation, ensuring:
      - train: exactly `train` normals (so all are normal)
      - test: exactly `test` graphs, with at least `min_normals_test` normals
      - validation: exactly `validacion` graphs, with at least `min_normals_valid` normals

    Raises ValueError if any of these minima cannot be met.
    """
    logging.info("Loading graphs from %s", pkl_file)
    all_graphs = load_graphs_info_from_pickle(pkl_file)
    logging.info("Total loaded: %d", len(all_graphs))

    # 1) Pick training (all normals)
    normals = [g for g in all_graphs if g['cat'].lower() == "normal"]
    if len(normals) < train:
        raise ValueError(f"Need {train} normals for training, have {len(normals)}")
    training_set = random.sample(normals, train)
    logging.info("Training set: %d normals", len(training_set))

    # 2) Build pool of remaining graphs
    used_ids = {id(g) for g in training_set}
    pool = [g for g in all_graphs if id(g) not in used_ids]
    pool_normals = [g for g in pool if g['cat'].lower() == "normal"]

    # 3) Sanity checks
    if test < min_normals_test:
        raise ValueError(f"test={test} < min_normals_test={min_normals_test}")
    if validacion < min_normals_valid:
        raise ValueError(f"validacion={validacion} < min_normals_valid={min_normals_valid}")

    total_needed_normals = min_normals_test + min_normals_valid
    if len(pool_normals) < total_needed_normals:
        raise ValueError(
            f"Not enough normals left for test/validation minima: "
            f"need {total_needed_normals}, have {len(pool_normals)}"
        )

    # 4) Build test set
    test_set: List[Dict[str, Any]] = []
    # 4a) reserve normals for test
    normals_for_test = random.sample(pool_normals, min_normals_test)
    test_set.extend(normals_for_test)
    for g in normals_for_test:
        pool.remove(g)
        pool_normals.remove(g)

    # 4b) fill rest of test
    remainder_test = test - min_normals_test
    if len(pool) < remainder_test:
        raise ValueError(f"Need {remainder_test} more graphs to fill test, have {len(pool)}")
    test_set.extend(random.sample(pool, remainder_test))
    logging.info("Test set: %d graphs (%d normals)", len(test_set), min_normals_test)

    # 5) Build validation set
    validation_set: List[Dict[str, Any]] = []
    normals_for_val = random.sample(pool_normals, min_normals_valid)
    validation_set.extend(normals_for_val)
    for g in normals_for_val:
        pool.remove(g)
        pool_normals.remove(g)

    remainder_val = validacion - min_normals_valid
    if len(pool) < remainder_val:
        raise ValueError(f"Need {remainder_val} more graphs to fill validation, have {len(pool)}")
    validation_set.extend(random.sample(pool, remainder_val))
    logging.info("Validation set: %d graphs (%d normals)", len(validation_set), min_normals_valid)

    # 6) Final size checks
    assert len(training_set) == train
    assert len(test_set)       == test
    assert len(validation_set) == validacion

    # 7) Cleanup
    del all_graphs, normals, pool, pool_normals
    gc.collect()

    logging.info("Partitioning done: train=%d, test=%d, valid=%d",
                 train, test, validacion)
    return training_set, test_set, validation_set