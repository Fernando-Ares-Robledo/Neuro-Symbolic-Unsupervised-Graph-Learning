import os
import torch
import random
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import time
import psutil
import config
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)



  


def record_memory_usage(label: str = "", record_every: int = 500, iteration: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """
    Records the memory usage (in GB) and the elapsed time since the start.
    
    This function records values only when iteration % record_every == 0
    """
    if iteration % record_every == 0:
        current_time = time.time() - config.start_time
        process = psutil.Process()
        mem_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        config.memory_times.append(current_time)
        config.memory_values.append(mem_usage_gb)
        return mem_usage_gb, current_time
    return None, None 
def doc(section: Optional[str] = None) -> None:
    """
    Displays the content from 'docs.md' using IPython Markdown.
    
    Parameters:
        section (str, optional): A substring to search for in the headers 
                                 (lines starting with '##'). If provided, only the content
                                 from that section until the next section header will be displayed.
                                 If None, the whole document is displayed.
    
    Example:
        doc("modulo_symbolico") 
    """
    from IPython.display import Markdown, display

    with open("docs.md", "r", encoding="utf-8") as f:
        content = f.readlines()
    
    if section is None:
        display(Markdown("".join(content)))
        return

    section_found = False
    extracted_lines = []
    
    for line in content:
        if line.startswith("##"):

            if section_found:
                break
            if section.lower() in line.lower():
                section_found = True
                extracted_lines.append(line)
                continue
        
        if section_found:
            extracted_lines.append(line)
    
    if not section_found:
        display(Markdown(f"**Section '{section}' not found in the document.**"))
    else:
        display(Markdown("".join(extracted_lines)))