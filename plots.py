
from utiles import *
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt

def plot_memory(memory_times,memory_values, proceso=""):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(memory_times, memory_values, marker='o')
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Memory Usage (GB)")
        plt.title(f"Memory Consumption during {proceso}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib is required for plotting memory consumption.")





def plot_graph_by_category(pkl_file, target_cat):
    graphs_info = load_graphs_info_from_pickle(pkl_file)
    
    filtered = [g for g in graphs_info if g['cat'].lower() == target_cat.lower()]
    
    if not filtered:
        raise ValueError(f"No se encontraron grafos con la categoría '{target_cat}'")
    
    selected = filtered[0]
    G = selected['graph']
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(f"Visualization of graph with cat='{target_cat}' (Timestamp: {selected.get('timestamp', 'N/A')})")
    plt.tight_layout()
    plt.show()
    
    #return selected


def plot_random_graph_by_category(pkl_file, target_cat, caracteristicas="False"):
    graphs_info = load_graphs_info_from_pickle(pkl_file)
    
    filtered = [g for g in graphs_info if g['cat'].lower() == target_cat.lower()]
    
    if not filtered:
        raise ValueError(f"No se encontraron grafos con la categoría '{target_cat}'")
    
    selected = random.choice(filtered)
    G = selected['graph']
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(f"Visualization of random graph with cat='{target_cat}' (Timestamp: {selected.get('timestamp', 'N/A')})")
    plt.tight_layout()
    plt.show()

    if caracteristicas==True:
        return selected

