
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
        messages):

    os.makedirs(save_path, exist_ok=True)

    processed = 0

    for idx, sample in enumerate(tqdm(samples, desc=split_name)):

        outfile = os.path.join(save_path, f"sample_{idx}.txt")

        if os.path.exists(outfile):
            continue

        prompt = messages + [
            {"role": "user", "content": sample}
        ]

        try:

            answer = llm.generate(prompt)

            with open(outfile, "w") as f:
                f.write(answer)

            processed += 1

        except Exception as e:

            print(e)
            time.sleep(60)

    print(f"{split_name}: processed {processed}")

def LLM_answering(args):

    split = args.split
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data_files"

    train_split_id = args.train_part
    dataset = args.dataset
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
    first_prompt = """
    For the QA task, follow the following template to answer the question and list the rational paths:
    
    <Solution> The rational paths are:
    1. <relation path1>
    2. <relation_path2>
    ....

     <Solution> is the special token here.
    Next, let's start with a example.
    Question: what character did john noble play in lord of the rings
    The reasoning paths are:
    1. John Noble -> film.actor.film -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    2. John Noble -> film.actor.film -> m.0528y98 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['film.actor.film', 'film.performance.character']
    3. John Noble -> award.award_winner.awards_won -> m.09k3pgy -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    4. John Noble -> award.award_nominee.award_nominations -> m.09k3q0p -> award.award_nomination.nominated_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_nominee.award_nominations', 'award.award_nomination.nominated_for', 'film.film.starring', 'film.performance.character']
    5. John Noble -> award.award_winner.awards_won -> m.0n7xsws -> award.award_honor.honored_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_winner.awards_won', 'award.award_honor.honored_for', 'film.film.starring', 'film.performance.character']
    6. John Noble -> award.award_nominee.award_nominations -> m.0b4d5rz -> award.award_nomination.nominated_for -> The Lord of the Rings: The Return of the King -> film.film.starring -> m.03l6qx7 -> film.performance.character -> Denethor II, the corresponding reasoning path is: ['award.award_nominee.award_nominations', 'award.award_nomination.nominated_for', 'film.film.starring', 'film.performance.character']
    """

    first_response = """
    The most direct and relevant way to determine John Noble’s “Lord of the Rings” character is via his actor–performance relationship, rather than detouring through award links. Paths #1 and #2 both use the same reasoning relations (“film.actor.film” → “film.performance.character”) and therefore are duplicates from a reasoning standpoint. The longer award-based paths (#3–#6) are not the most straightforward way to answer “What character did John Noble play?” and thus are less rational for this specific question.

    after deduplication, the rational path is:

    <Solution> The rational paths are:
    1. [film.actor.film, film.performance.character]
    """


    job_start = "Now, let's begin! Identify all the rational paths, and list below with explanations. "

    # Load training and validation data
    N_tr = len(train_data)
    N_val = len(val_data)
    N_test = len(test_data)
    train_samples = [dict_to_text(train_data[i][i]) for i in range(N_tr)]
    val_samples = [dict_to_text(val_data[i][i]) for i in range(N_val)]
    test_samples = [dict_to_text(test_data[i][i]) for i in range(N_test)]

    # Messages to pass for the GPT API or the local LLM
    messages = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": job_start}
    ]

    count = 0
    SLEEP_TIME = 60  # Adjust this value if needed based on your rate limit

    ### Process training set, these commands create the folder storing the processed training set
    save_train_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/train/"
    save_val_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/val/"
    save_test_path = f"{DATA_DIR}/{dataset}/annotated_paths_LLM/test/"
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_val_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)


    end_idx = N_tr
    train_samples_to_process = train_samples
    val_samples_to_process = val_samples
    test_samples_to_process = test_samples


    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = TransformersLLM(
                model_name="Qwen/Qwen2.5-0.5B-Instruct", #Qwen/Qwen3.5-35B-A3B-FP8
                device=device
                )

    if split=='train' or split=='all':
        #########################################
        #########For the train split ############
        #########################################
        start_idx = 0 #reset
        for idx, sample in tqdm(enumerate(train_samples_to_process)):
            if os.path.exists(save_train_path+ f'sample_{start_idx + idx}.txt'):
                print(colored(f"sample_{start_idx + idx}.txt already exists, skipping", 'red'))
                continue
            print(colored(f"Processing sample {start_idx + idx}/{N_tr} \n", 'yellow'))

            # Prepare LLM messages
            copy_messages = dp(messages)
            copy_messages.append({"role": "user", "content": sample})

            # Request GPT-4o response
            try:

                answer = llm.generate(copy_messages)

                with open(save_train_path + f'sample_{start_idx + idx}.txt', 'w') as f:
                    f.write(answer)
                count += 1


            except Exception as e:
                # Handle other exceptions: Wait 180 seconds before retrying
                print(colored(f"Error encountered: {e}. Retrying after 60 seconds...", 'cyan'))
                time.sleep(60)
            print ('processed sample count:',count)

    if split=='val' or split=='all':
        #########################################
        #########For the val split ##############
        #########################################
        start_idx = 0 #reset
        for idx, sample in tqdm(enumerate(val_samples_to_process)):
            if os.path.exists(save_val_path+ f'sample_{start_idx + idx}.txt'):
                print(colored(f"sample_{start_idx + idx}.txt already exists, skipping", 'red'))
                continue
            print(colored(f"Processing sample {start_idx + idx}/{N_val} \n", 'yellow'))
            
            # Prepare LLM messages
            copy_messages = dp(messages)
            copy_messages.append({"role": "user", "content": sample})

            # Request LLM response
            try:

                answer = llm.generate(copy_messages)

                with open(save_val_path + f'sample_{start_idx + idx}.txt', 'w') as f:
                    f.write(answer)
                count += 1

            except Exception as e:
                # Handle other exceptions: Wait 180 seconds before retrying
                print(colored(f"Error encountered: {e}. Retrying after 60 seconds...", 'cyan'))
                time.sleep(60)
            print ('processed sample count:',count)

    if split=='test' or split=='all':
        #########################################
        #########For the test split #############
        #########################################
        start_idx = 0 #reset
        for idx, sample in tqdm(enumerate(test_samples_to_process)):
            if os.path.exists(save_test_path+ f'sample_{start_idx + idx}.txt'):
                print(colored(f"sample_{start_idx + idx}.txt already exists, skipping", 'red'))
                continue
            print(colored(f"Processing sample {start_idx + idx}/{N_test} \n", 'yellow'))

            # Prepare LLM messages
            copy_messages = dp(messages)
            copy_messages.append({"role": "user", "content": sample})

            try:

                answer = llm.generate(copy_messages)


                with open(save_test_path + f'sample_{start_idx + idx}.txt', 'w') as f:
                    f.write(answer)
                count += 1

            except Exception as e:
                # Handle other exceptions: Wait 180 seconds before retrying
                print(colored(f"Error encountered: {e}. Retrying after 60 seconds...", 'cyan'))
                time.sleep(60)
            print ('processed sample count:',count)



if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='toy',
                        choices=['webqsp', 'cwq','toy', 'OntoOmicsKG_step2'], help='Dataset name')
    parser.add_argument('--train_part', type=int, default=-1,
                        choices=[-1,0,1,2], help='-1 mean make the entire train dataset')
    parser.add_argument('--which_key', type=int, default=0,
                        choices=[0], help='which openai key to use')
    parser.add_argument('--split', type=str, default='all',
                            choices=['train','val','test','all'], help='which split to process')
    args = parser.parse_args()
    LLM_answering(args)
    
    