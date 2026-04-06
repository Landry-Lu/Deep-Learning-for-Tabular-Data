import random
import numpy as np
import torch
import os
import json
from typing import Any, Dict

def set_seed(seed: int = 42):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_results(results: Dict[str, Any], filename: str):
    """保存结果到 JSON 文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        return json.load(f)
