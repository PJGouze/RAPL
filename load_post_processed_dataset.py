#load_post_processed_dataset.py, update : 03/08 9:30
import numpy as np
import os
import sys
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
#from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.dataset.utils import *
from src.dataset.PyG_dataset_disk import KGDataset
from src.dataset.PostProcessedDataset import ProcessedDiskDataset
import pickle
import warnings
import argparse
warnings.filterwarnings("ignore")


def _parse_rational_paths(file_dir, file_name):
    
    """
    Given a directory (file_dir) and a file name (file_name) that points to a text file
    (e.g. file_dir='path/to' and file_name='sample_1.txt'), this function reads and parses
    all lines after the phrase "The rational paths are:". Each path line in the file is
    expected to follow this format:

        1. [location.location.partially_contains, geography.river.basin_countries]
        2. [location.location.time_zones]
        3. [other.path.example]

    The function extracts each bracketed sequence, splits by commas, and creates a tuple
    of strings. It then collects these tuples into a list and returns it.
    """
    
    full_path = os.path.join(file_dir, file_name)
    lines = []
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    parsed_paths = []
    in_section = False
    pattern = re.compile(r'^\s*\d+\.\s*\[(.*?)\]\s*$')
    for line in lines:
        if "The rational paths are:" in line:
            in_section = True
            continue
        if in_section:
            match = pattern.match(line.strip())
            if match:
                bracket_content = match.group(1)
                parts = [p.strip() for p in bracket_content.split(',')]
                parsed_paths.append(tuple(parts))
    return parsed_paths

def _collect_annotated_paths(folder_path):
    if not os.path.isdir(folder_path):
        return None

    txt_files = [f for f in os.listdir(folder_path)
                    if f.startswith("sample_") and f.endswith(".txt")]
    def extract_index(fname):
        base = fname.replace(".txt", "")
        parts = base.split("_")
        return int(parts[-1])
    txt_files_sorted = sorted(txt_files, key=extract_index)

    parsed_paths_list = []
    for txt_file in txt_files_sorted:
        parsed = _parse_rational_paths(folder_path, txt_file)
        parsed_paths_list.append(parsed)
    return parsed_paths_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset name.')
    parser.add_argument('--name', type=str, default='toy', help='Name of the dataset')
    args = parser.parse_args()
    name = args.name


    for split in ['train', 'val','test']:
        print (f"post process dataset:{name}, {split} part")
        print ('using LLM labels')
        post_processed_dir = os.path.join(
                                        f"data_files/{name}",
                                        "final"
                                    )
        kgdataset = KGDataset(root=f'data_files/{name}/',split=split)

        

        # raw_dir = f"data_files/{name}/raw/"
        # processed_dir = f"data_files/{name}/processed"

        base_dir = f"data_files/{name}"
        processed_dir = os.path.join(base_dir, "processed")
        annotated_dir = os.path.join(base_dir, "annotated_paths_LLM")
        final_dir = os.path.join(base_dir, "final")

        print ('loading retrieval files')
        with open(os.path.join(processed_dir, f"{split}_retrieval.pkl"), "rb") as f:
            retrieval_list = pickle.load(f)
        print ('loading annotated text path labels')



        text_labels = _collect_annotated_paths(
            os.path.join(annotated_dir, split)
        )

        # Replacing "metadata file by the files that contain the h,r and t_id_list, which is used in the metadata file"
        print ('loading metadata files')
        
        # with open(f"{processed_dir}/{split}/metadata_{split}.pkl", 'rb') as f:
        #     metadata_list = pickle.load(f)
        
        # print (len(text_labels),len(metadata_list),len(retrieval_list)) # type: ignore
        # assert len(text_labels)==len(metadata_list)==len(retrieval_list) # type: ignore

        print (f"start making final {split} datasets")
        labeled_topic_relation_path=f"data_files/{name}/relationtargeting/{split}_results.pkl"
        post_dataset = ProcessedDiskDataset(processed_dir, split, kgdataset, labeled_topic_relation_path,retrieval_list=retrieval_list,metadata_list=None,text_labels=text_labels)
        print (post_dataset[0])
        print(post_dataset[0].keys())
        data = post_dataset[0]
        print("num_nodes", data.num_nodes)
        print(len(data.one_hop_neighbors))
        print("topic_candidates", data.topic_candidates.sum())
        print("topic labels:", data.topic_labels.sum())
        print("LLM labels:", data.llm_topic_labels.sum())
        print("preprocessing achieved")

    
