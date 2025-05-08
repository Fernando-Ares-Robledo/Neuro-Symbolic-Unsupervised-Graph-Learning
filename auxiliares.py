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


def load_dataset(dataset):
    print("Loading IoT Network Intrusion Dataset...")
    t0 = time.time()
    df = pd.read_csv(dataset)
    print(df.shape)
    print(df.columns)
    t1 = time.time()
    return df

def create_graphs_from_dataset(df, record_every=500):
    graphs_info = []
    grouped = df.groupby('Timestamp')
    total_timestamps = df['Timestamp'].nunique()
    
    pbar = tqdm(grouped, total=total_timestamps, desc="Creating graphs")
    try:
        for i, (timestamp, group) in enumerate(pbar):
            mem_usage, _ = record_memory_usage("Before graph creation", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            group_data = group.copy()
            cat = group['Cat'].unique()[0]
            G = nx.Graph()
            for _, row in group.iterrows():
                src = row['Src_IP']
                dst = row['Dst_IP']
                
                if not G.has_node(src):
                    G.add_node(src, flows=0, last_timestamp=None)
                if not G.has_node(dst):
                    G.add_node(dst, flows=0, last_timestamp=None)
                
                G.nodes[src]['flows'] += 1
                G.nodes[src]['last_timestamp'] = row['Timestamp']
                G.nodes[dst]['flows'] += 1
                G.nodes[dst]['last_timestamp'] = row['Timestamp']
                
                if G.has_edge(src, dst):
                    G.edges[src, dst]['flow_data'].append(row.to_dict())
                    G.edges[src, dst]['flow_count'] += 1
                else:
                    G.add_edge(src, dst, flow_data=[row.to_dict()], flow_count=1)
            
            graphs_info.append({
                'timestamp': timestamp,
                'cat': cat,
                'graph': G,
                'data': group_data
            })
            del group, group_data, G, cat
            #gc.collect()
            mem_usage, _ = record_memory_usage("After graph creation", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
    except KeyboardInterrupt:
        print("Interrupción detectada durante la creación de grafos. Saliendo del bucle.")
    
    del grouped
    gc.collect()
    return graphs_info

def save_graphs_info_to_pickle(graphs_info, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graphs_info, f)
    print(f"Graphs saved to {filename}")




def graph_descriptors(g):
    """
    Computes various descriptors for the input graph 'g' and returns them as a dictionary.
    """
    descriptors = {}
    descriptors['num_nodes'] = g.number_of_nodes()
    descriptors['num_edges'] = g.number_of_edges()
    descriptors['density'] = nx.density(g)

    degrees = dict(g.degree())
    descriptors['average_degree'] = np.mean(list(degrees.values()))

    degree_distribution = {}
    for deg in degrees.values():
        degree_distribution[deg] = degree_distribution.get(deg, 0) + 1
    descriptors['degree_distribution'] = degree_distribution
    del degrees, degree_distribution

    descriptors['average_clustering'] = nx.average_clustering(g)

    if nx.is_connected(g):
        descriptors['diameter'] = nx.diameter(g)
        descriptors['average_shortest_path_length'] = nx.average_shortest_path_length(g)
    else:
        descriptors['diameter'] = None
        descriptors['average_shortest_path_length'] = None

    centrality = {
        'degree': nx.degree_centrality(g),
        'closeness': nx.closeness_centrality(g),
        'betweenness': nx.betweenness_centrality(g)
    }
    descriptors['centrality_measures'] = centrality
    del centrality

    L = nx.laplacian_matrix(g).todense()
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = np.sort(np.real(eigenvalues))
    spectral_properties = {'eigenvalues': eigenvalues}
    nonzero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    spectral_gap = nonzero_eigenvalues[0] if nonzero_eigenvalues.size > 0 else 0.0
    spectral_properties['spectral_gap'] = spectral_gap
    descriptors['spectral_properties'] = spectral_properties
    del L, eigenvalues, nonzero_eigenvalues, spectral_properties, spectral_gap

    #gc.collect()
    return descriptors


def analyze_graphs(graphs_info, output_filename, record_every=500):
    
    accumulators = {}
    descriptor_names = ['num_nodes', 'num_edges', 'density', 
                        'average_degree', 'average_clustering', 
                        'diameter', 'average_shortest_path_length']
    spectral_name = 'spectral_gap'
    
    pbar = tqdm(graphs_info, total=len(graphs_info), desc="Analyzing graphs")
    try:
        for i, item in enumerate(pbar):
            mem_usage, _ = record_memory_usage("Before analyzing graph", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            cat = item['cat']
            g = item['graph']
            
            desc = graph_descriptors(g)
            item["descriptors"] = desc
            
            scalar_values = {}
            for key in descriptor_names:
                value = desc.get(key, np.nan)
                if value is None:
                    value = np.nan
                scalar_values[key] = value
            
            spec_gap = desc.get('spectral_properties', {}).get('spectral_gap', np.nan)
            if spec_gap is None:
                spec_gap = np.nan
            scalar_values[spectral_name] = spec_gap
            
            data_df = item.get("data")
            if data_df is not None:
                numeric_data = data_df.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    for col in numeric_data.columns:
                        scalar_values[f"data_mean_{col}"] = np.nanmean(numeric_data[col].values)
                else:
                    scalar_values["data_mean"] = np.nan
            else:
                scalar_values["data_mean"] = np.nan
            
            item["descriptors"].update(scalar_values)
            
            if cat not in accumulators:
                accumulators[cat] = {k: [] for k in item["descriptors"].keys()}
            
            for key, value in scalar_values.items():
                accumulators[cat][key].append(value)
            
            #del g, desc, scalar_values, spec_gap, data_df, cat
            #gc.collect()
            mem_usage, _ = record_memory_usage("After analyzing graph", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
    except KeyboardInterrupt:
        print("Interrupción detectada durante el análisis de grafos. Saliendo del bucle.")
    
    aggregated_results = {}
    for cat, desc_dict in accumulators.items():
        aggregated_results[cat] = {}
        for key, values in desc_dict.items():
            values = np.array(values, dtype=np.float64)
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            aggregated_results[cat][key] = (mean_val, std_val)
    del desc_dict
        #gc.collect()
    
    del accumulators
    gc.collect()

    with open(output_filename, "wb") as f:
        pickle.dump(graphs_info, f)
    print(f"Descriptors saved to {output_filename}")
    
    return aggregated_results

def convert_to_line_graph(g):
    """
    Converts the input graph g into its line graph.
    
    In the resulting line graph:
      - Each node represents an edge of the original graph g.
      - Two nodes are connected if the corresponding edges in g share at least one common vertex. 
    
    Parameters:
        g (networkx.Graph): The input graph.
    
    Returns:
        networkx.Graph: The line graph of g.
    """
    line_g = nx.line_graph(g)
    return line_g
def convert_and_save_line_graphs(input_filename: str,
                                 output_filename: str,
                                 record_every: int = 500) -> None:

    def transfer_edge_attributes(g, line_g, normalize: bool = True):
        attr_vals: Dict[str, list] = {}
        for u, v, data in g.edges(data=True):
            for k, val in data.items():
                if isinstance(val, (int, float)):
                    attr_vals.setdefault(k, []).append(val)
        mins, maxs = {}, {}
        if normalize:
            for k, vals in attr_vals.items():
                mins[k], maxs[k] = min(vals), max(vals)
        for node in line_g.nodes():
            data = g.get_edge_data(*node, default={})
            for k, val in data.items():
                line_g.nodes[node][k] = val
            if normalize:
                for k, val in data.items():
                    if isinstance(val, (int, float)):
                        mn, mx = mins[k], maxs[k]
                        norm_val = (val - mn) / (mx - mn) if mx != mn else 0.0
                        line_g.nodes[node][f"{k}_norm"] = norm_val
        return line_g

    def assign_cosine_distance_weights(line_g):
        for u, v in line_g.edges():
            attrs_u = line_g.nodes[u]
            attrs_v = line_g.nodes[v]
            common_keys = sorted(k for k in attrs_u if k.endswith("_norm") and k in attrs_v)
            if not common_keys:
                w = 0.0
            else:
                vec_u = np.array([attrs_u[k] for k in common_keys], float)
                vec_v = np.array([attrs_v[k] for k in common_keys], float)
                nu, nv = np.linalg.norm(vec_u), np.linalg.norm(vec_v)
                if nu > 0 and nv > 0:
                    sim = np.dot(vec_u, vec_v) / (nu * nv)
                    w = 1.0 - sim
                else:
                    w = 0.0
            line_g.edges[u, v]['weight'] = w
        return line_g

    graphs = load_graphs_info_from_pickle(input_filename)
    new_graphs = []

    for i, item in enumerate(tqdm(graphs, desc="Converting to line graphs", unit="graph")):
        mem, _ = record_memory_usage(record_every=record_every, iteration=i)
        if mem is not None:
            tqdm.write(f"Iter {i}: Memory {mem:.2f} GB")

        g = item['graph']
        line_g = convert_to_line_graph(g)
        line_g = transfer_edge_attributes(g, line_g, normalize=True)
        line_g = assign_cosine_distance_weights(line_g)

        new_item = item.copy()
        new_item['graph'] = line_g
        new_graphs.append(new_item)

        del g, line_g, new_item
        #gc.collect()

    with open(output_filename, 'wb') as f:
        pickle.dump(new_graphs, f)
    print(f"Saved converted line graphs to {output_filename}")

    del graphs, new_graphs
    gc.collect()

def compute_and_save_mst_graphs(input_filename: str, output_filename: str, record_every: int = 500) -> None:
    """
    Loads a pickle file containing a list of graph dictionaries, computes mts
    for each graph (preserving node attributes), and saves the new list of dictionaries into a new pickle file.

    The MST is computed using NetworkX's minimum_spanning_tree() function over the edge weights (using the attribute 'weight').

    Parameters:
        input_filename (str): Path to the input pickle file containing the original graphs.
        output_filename (str): Path to the output pickle file where the MST graphs will be saved.
        record_every (int): Frequency (in iterations) to record memory usage.

    Returns:
        None

    """

    graphs_info = load_graphs_info_from_pickle(input_filename)
    new_graphs_info = []
    
    pbar = tqdm(graphs_info, total=len(graphs_info), desc="Computing MST graphs")
    try:
        for i, item in enumerate(pbar):
            mem_usage, _ = record_memory_usage("Before MST", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            
            G = item.get('graph')
            if G is not None:
                mst = nx.minimum_spanning_tree(G, weight='weight')
                new_item = item.copy()
                new_item['graph'] = mst
                new_graphs_info.append(new_item)
                del mst, new_item
            else:
                new_graphs_info.append(item)
            
            del G, item
            #gc.collect()
            
            mem_usage, _ = record_memory_usage("After MST", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            pbar.update(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting MST computation loop.")
    
    pbar.close()
    
    with open(output_filename, 'wb') as f:
        pickle.dump(new_graphs_info, f)
    print(f"MST graphs saved to {output_filename}")
    
    del new_graphs_info, graphs_info
    gc.collect()


def compute_and_save_mst_graphs(input_filename: str, output_filename: str, record_every: int = 500) -> None:

    graphs_info = load_graphs_info_from_pickle(input_filename)
    new_graphs_info = []
    
    pbar = tqdm(graphs_info, total=len(graphs_info), desc="Computing MST graphs")
    try:
        for i, item in enumerate(pbar):
            mem_usage, _ = record_memory_usage("Before MST", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            
            G = item.get('graph')
            if G is not None:
                mst = nx.minimum_spanning_tree(G, weight='weight')
                new_item = item.copy()
                new_item['graph'] = mst
                new_graphs_info.append(new_item)
                del mst, new_item
            else:
                new_graphs_info.append(item)
            
            del G, item
            #gc.collect()
            
            mem_usage, _ = record_memory_usage("After MST", record_every, i)
            if mem_usage is not None:
                pbar.set_postfix(memory=f"{mem_usage:.2f} GB")
            pbar.update(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting MST computation loop.")
    
    pbar.close()
    
    with open(output_filename, 'wb') as f:
        pickle.dump(new_graphs_info, f)
    print(f"MST graphs saved to {output_filename}")
    
    del new_graphs_info, graphs_info
    gc.collect()