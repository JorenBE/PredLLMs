from transformers.utils import logging
from transformers.file_utils import TRANSFORMERS_CACHE

# Get the default cache folder
cache_folder = TRANSFORMERS_CACHE
print(f"The cache folder is: {cache_folder}")

import torch

torch.cuda.empty_cache()