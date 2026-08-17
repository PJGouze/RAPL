#update: 06/08 10:32
from pathlib import Path

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever_v2 import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from src.model.llms import TransformersLLM

import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import re
from copy import deepcopy as dp
from termcolor import colored
#import openai

import warnings
warnings.filterwarnings("ignore")

import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dict_to_text(data):
    """
    Converts a dictionary containing a question, translated paths, and reasoning paths
    into a formatted text representation.

    Args:
        data (dict): The dictionary containing:
            - "question": The main question.
            - "translated_paths": A list of paths.
            - "reasoning_paths": A list of reasoning steps corresponding to each path.

    Returns:
        str: A formatted string representation of the data.
    """
    text = f"Question: {data['question']}\nThe paths are:\n"

    for i, (path, reasoning) in enumerate(zip(data["translated_paths"], data["reasoning_paths"]), 1):
        text += f"{i}. {path}, the corresponding reasoning path is: {reasoning}\n\n"

    return text.strip()  # Remove the trailing newline

def extract_latest_answer(response_text):
    # Split into different turns based on [INST] markers
    turns = re.split(r"\[INST\]", response_text)
    
    if len(turns) < 2:
        return response_text  # Return raw response if it doesn't follow the expected format
    
    # Get the last assistant's response (after the last user question)
    last_turn = turns[-1]  # Last user question + assistant response
    last_answer = re.sub(r"</?s>", "", last_turn)  # Remove <s> and </s>
    
    # Extract only the assistant's part (text after [/INST])
    last_answer = last_answer.split("[/INST]")[-1].strip()
    
    return last_answer

def convert_messages_to_response_text(messages):
    """
    Converts a list of structured chat messages into a formatted response text 
    with <s> and [INST] markers, mimicking Mistral-7B's output.

    Args:
        messages (list): List of {"role": "user"/"assistant", "content": "..."} messages.

    Returns:
        str: The formatted chat response string.
    """
    response_text = ""
    
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            response_text += f"<s> [INST] {msg['content']} [/INST] "
        elif msg["role"] == "assistant":
            response_text += f"{msg['content']} </s>"

    return response_text.strip()

def process_split(
        samples,
        save_path,
        split_name,
        llm,
        messages,
        batch_size=8):


    os.makedirs(save_path, exist_ok=True)
    print(batch_size)

    processed = 0

    # Create batches
    for start_idx in tqdm(
            range(0, len(samples), batch_size),
            desc=split_name):

        batch_messages = []
        batch_indices = []

        # Prepare current batch
        for idx in range(
                start_idx,
                min(start_idx + batch_size, len(samples))):

            outfile = os.path.join(
                save_path,
                f"sample_{idx}.txt"
            )


            # Skip already processed samples
            if os.path.exists(outfile):
                continue

            prompt = messages + [
                {
                    "role": "user",
                    "content": samples[idx]
                }
            ]

            batch_messages.append(prompt)
            batch_indices.append(idx)


        # Entire batch already processed
        if len(batch_messages) == 0:
            continue


        try:
            # print("=" * 80)
            # print(batch_messages[0][-1]["content"])
            # print("=" * 80)

            # Batched generation
            answers = llm.generate(batch_messages)


            # Save results
            for idx, answer in zip(
                    batch_indices,
                    answers):

                outfile = os.path.join(
                    save_path,
                    f"sample_{idx}.txt"
                )

                with open(outfile, "w") as f:
                    f.write(answer)

                processed += 1


        except Exception as e:

            print(
                f"{split_name} batch starting at "
                f"{start_idx} failed: {e}"
            )

            # Retry one by one if batch fails
            print("Retrying samples individually...")

            for idx, prompt in zip(
                    batch_indices,
                    batch_messages):

                outfile = os.path.join(
                    save_path,
                    f"sample_{idx}.txt"
                )

                try:

                    answer = llm.generate(prompt)

                    with open(outfile, "w") as f:
                        f.write(answer)

                    processed += 1

                except Exception as single_error:

                    print(
                        f"Sample {idx} failed: "
                        f"{single_error}"
                    )

                    time.sleep(60)


    print(
        f"{split_name}: processed {processed}"
    )

def LLM_answering(args):
    
    split = args.split
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data_files"

    train_split_id = args.train_part
    dataset = args.dataset
    llm_name = args.llm_name
    #Loading the datasets containing the retrieved path from the data pickles
    path_tr = f"{DATA_DIR}/{dataset}/processed/train_text_dict_list.pickle"
    path_val = f"{DATA_DIR}/{dataset}/processed/val_text_dict_list.pickle"
    path_test = f"{DATA_DIR}/{dataset}/processed/test_text_dict_list.pickle"

    with open(path_tr, "rb") as f:
        train_data = pickle.load(f)
    with open(path_val, "rb") as f:
        val_data = pickle.load(f)
    with open(path_test, "rb") as f:
        test_data = pickle.load(f)  
    print (f"val data len: {len(val_data)}",
           f"train data len: {len(train_data)}",
           f"test data len: {len(test_data)}")
    

    # Define the system and prompt as per the original function
    # first_prompt = """
    # For the QA task, follow the following template to answer the question and list the rational paths:
    
    # <Solution> The rational paths are:
    # 1. <relation path1>
    # 2. <relation_path2>
    # ....

    #  <Solution> is the special token here.
    # Next, let's start with a example.
    # Question: what character did john noble play in lord of the rings
    # The reasoning paths are:
    # 1. John Noble -> film.actor.film -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    # 2. John Noble -> film.actor.film -> m.0528y98 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    # 3. John Noble -> award.award_winner.awards_won -> m.09k3pgy -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    # """

    # first_response = """
    # The most direct and relevant way to determine John Noble’s “Lord of the Rings” character is via his actor–performance relationship, rather than detouring through award links. Paths #1 and #2 both use the same reasoning relations (“film.actor.film” → “film.performance.character”) and therefore are duplicates from a reasoning standpoint. The longer award-based paths (#3–#6) are not the most straightforward way to answer “What character did John Noble play?” and thus are less rational for this specific question.

    # after deduplication, the rational path is:

    # <Solution> The rational paths are:
    # 1. [film.actor.film, film.performance.character]
    # """


    # job_start = """Now, let's begin! Identify all the rational paths, and return them as following:
    #         <rational paths>
    #         1. [...]
    #         2. [...]
    #     """
    
    first_prompt = """
    You are an assistant specialized in Knowledge Graph reasoning path selection.

    Your task is NOT to solve the question from your own knowledge.
    You are given:
    1. A question.
    2. A list of candidate paths extracted from a Knowledge Graph.

    Your goal is to select the rational paths that directly answer the question.

    IMPORTANT RULES:
    - Use ONLY the candidate paths provided in the input.
    - NEVER create new paths.
    - NEVER modify the paths.
    - NEVER use external knowledge.
    - NEVER explain your reasoning.
    - NEVER summarize the question.
    - Remove duplicate reasoning paths if they represent the same sequence of relations.
    - Return only the relation sequences.

    The answer MUST follow exactly this format:

    <Solution>
    1. [relation_1, relation_2, ...]
    2. [relation_1, relation_2, ...]

    Example:

    Question:
    what character did john noble play in lord of the rings

    The candidate paths are:

    1. John Noble -> film.actor.film -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']

    2. John Noble -> award.award_winner.awards_won -> m.09k3pgy -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    """

    first_response = """
    <Solution>
    1. [John Noble -> film.actor.film -> m.03l6qx7 -> film.performance.character -> Denethor II]
    """

    job_start = """
    Now process the following example.

    Return ONLY the rational relation paths.

    Do not include:
    - explanations
    - comments
    - reasoning
    - additional text

    Only output:

    <Solution>
    1. [...]
    2. [...]
    """

    N_tr = len(train_data)
    N_val = len(val_data)
    N_test = len(test_data)
    train_samples = [dict_to_text(train_data[i][i]) for i in range(N_tr)]
    val_samples = [dict_to_text(val_data[i][i]) for i in range(N_val)]
    test_samples = [dict_to_text(test_data[i][i]) for i in range(N_test)]

    print("=" * 80)
    print(train_samples[0])
    print("=" * 80)
    # Messages to pass for the GPT API or the local LLM
    messages = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": job_start}
    ]

    count = 0
    SLEEP_TIME = 60  # Adjust this value if needed based on your rate limit

    ### Process training set, these commands create the folder storing the processed training set
    save_train_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/{llm_name}/train/"
    save_val_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/{llm_name}/val/"
    save_test_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/{llm_name}/test/"
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_val_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)

    
    end_idx = N_tr
    train_samples_to_process = train_samples
    val_samples_to_process = val_samples
    test_samples_to_process = test_samples
   
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = TransformersLLM(
                model_name=llm_name, #Qwen/Qwen3.5-35B-A3B-FP8, Qwen/Qwen2.5-0.5B-Instruct
                device=device
                )

    if split in ("train", "all"):
        process_split(train_samples, save_train_path, "train", llm, messages, batch_size=16)

    if split in ("val", "all"):
        process_split(val_samples, save_val_path, "val", llm, messages, batch_size=16)

    if split in ("test", "all"):
        process_split(test_samples, save_test_path, "test", llm, messages, batch_size=16)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='OntoOmicsKG_step2',
                        choices=['webqsp', 'cwq','toy', 'OntoOmicsKG_step2'], help='Dataset name')
    parser.add_argument('--train_part', type=int, default=-1,
                        choices=[-1,0,1,2], help='-1 mean make the entire train dataset')
    parser.add_argument('--which_key', type=int, default=0,
                        choices=[0], help='which openai key to use')
    parser.add_argument('--split', type=str, default='all',
                            choices=['train','val','test','all'], help='which split to process')
    parser.add_argument('--llm_name', type=str, default='Qwen/Qwen2.5-14B-Instruct',
                                choices=['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen2.5-14B-Instruct','Qwen/Qwen3.5-35B-A3B-FP8'], help='which LLM does the annotation')
    args = parser.parse_args()
    LLM_answering(args)
    
    
