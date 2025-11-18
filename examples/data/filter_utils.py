import os
import sys
import re
from pathlib import Path
from PIL import Image
import numpy as np


def create_regex_from_template(template):
    fields = template.split('_')
    regex_parts = [r'([^_]+)' for _ in fields]
    full_pattern = '_'.join(regex_parts)
    return re.compile(full_pattern), fields


def extract_metadata_from_dirname(dirname, template):
    normalized_dirname = dirname.replace('-', '_')
    regex, field_names = create_regex_from_template(template)
    match = regex.match(normalized_dirname)
    
    if match:
        return dict(zip(field_names, match.groups()))
    return {}


def process_dataset_names(root_dir, template, file_extension='.tif'):
    root_path = Path(root_dir)
    results = []
    
    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue
        
        metadata = extract_metadata_from_dirname(subdir.name, template)
        
        if not metadata:
            print(f"Warning: Could not parse directory name: {subdir.name}")
            continue
        
        result = {
            'directory_path': str(subdir),
            'directory_name': subdir.name,
            **metadata
        }
        results.append(result)
    
    return results

def filter_dataset_names(samples, filters):
    final_samples = []
    for i, result in enumerate(samples):
        keep = True
        for key, value in filters.items():
            if value is not None:
                keep = keep and (value in result[key])
        if keep:
            final_samples.append(result.copy())
    return final_samples
