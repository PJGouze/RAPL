import os
import sys
import pickle
import json
import time
#import openai
from argparse import ArgumentParser
from xml.parsers.expat import model
from tqdm import tqdm
from termcolor import colored
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat(args):
    """
    This function reads three pickle files (train, val, test). Each pickle file is
    a list of dictionaries with the structure:
        {
            'question': str,
            'cand_relations': set(...),
            'gt_relations': set(...)
        }

    For each dictionary in the training data, we can optionally process only a subset
    of the dataset (1/3 at a time) if --train_part is set to 0, 1, or 2. If
    --train_part = -1, we process the entire training dataset.

    For each dictionary in these lists, it asks GPT to identify which of the candidate
    relations are relevant to the question, in the format of a Python list of strings.

    Inputs:
    ---------
    args : Parsed command line arguments:
        - which_key: An integer specifying which API key to use.
        - input_path, val_path, test_path: Paths to train, val, and test pickle files. Default: data_files/toy/processed
        - out_dir: The directory to save the output (JSON) files.
        - train_part: -1 for the entire training set; 0, 1, or 2 for splitting the
          training data into three parts.
        - split: commanding which split to process. Default is 'train', options are: ['train', 'val', 'test', 'all']
    
    Outputs:
    --------
    The script saves three JSON files into out_dir:
        - train_results.json
        - val_results.json
        - test_results.json

    Each JSON file is a list of response dictionaries, for example:
        [
            {
                "question": <question_str>,
                "cand_relations": <candidate_relations_list>,
                "response_text": <raw GPT response>
            },
            ...
        ]
    """

    # Set your OpenAI API key
    # key_id = args.which_key
    # if key_id == 0:
    #     print ('type your openai key here')
    #     openai.api_key = "xxx"  # Replace with your actual API key

    #loading the model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    model.to(device)
    model.eval()
    split=args.split

    # Create output directory if needed
    os.makedirs(args.out_dir, exist_ok=True)
    INPUT_PATH = args.input_path
    train_input_path=f'{INPUT_PATH}/train_text_dict_list.pickle'
    val_input_path=f'{INPUT_PATH}/val_text_dict_list.pickle'
    test_input_path=f'{INPUT_PATH}/test_text_dict_list.pickle'

    # Load data from pickle files
    with open(train_input_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_input_path, "rb") as f:
        val_data = pickle.load(f)
    with open(test_input_path, "rb") as f:
        test_data = pickle.load(f)


    # If we only want a fraction of the training data, handle it here
    N_tr = len(train_data)
    if args.train_part in [0, 1, 2]:
        split_size = N_tr // 3
        start_idx = split_size * args.train_part
        # If it's the last part, take everything until the end
        end_idx = split_size * (args.train_part + 1) if args.train_part < 2 else N_tr
        train_data = train_data[start_idx:end_idx]
    elif args.train_part == -1:
        # Use the entire training set
        pass
    else:
        raise ValueError("train_part must be one of [-1, 0, 1, 2]")

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant for identifying relevant relations "
            "given a question and a set of candidate relations."
        )
    }

    def build_prompt(question, cand_relations):
        """
        Builds the prompt for GPT by taking in a question and a set of candidate relations.

        Inputs:
        question: str
            The input question for which we want relevant relations.
        cand_relations: iterable
            The set (or list) of candidate relations.

        Output:
        A string that we feed into the 'content' of a 'user' role message.
        """
        prompt_str = (   # try without examples
            f"We have the question:\n\n{question}\n\n"
            f"And a set of relations for this question:\n\n{cand_relations}\n\n"
            "List the relevant relations for this question. Please respond using the following format:\n\n"
            "<Solution:> [r1,r2,....] \n\n"
        )
        return prompt_str

    """
    def gpt_call(prompt):

        Sends a chat completion request to OpenAI's API using the messages
        in 'prompt'. This function continually retries on error.

        Input:
        ------
        prompt: List[dict]
            Example structure:
            [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."}
            ]

        Output:
        -------
        str
            The content of the assistant's response.

        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=prompt,
                    temperature=0.0,
                )
                return response["choices"][0]["message"]["content"]

            except openai.error.RateLimitError:
                print(colored("Rate limit reached. Waiting 60 seconds...", 'red'))
                time.sleep(60)
                # Retry with the same prompt in the next loop iteration
    """

    def llm_call(prompt):

        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        )

        device = model.get_input_embeddings().weight.device

        inputs = {k: v.to(device)
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.8
            )

        generated = outputs[:, inputs["input_ids"].shape[1]:]

        answer = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        return answer.strip()

    @torch.inference_mode()
    def llm_call_batch(prompts):

        texts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]


        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left"
        ).to(device)


        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )


        answers = []

        for output, input_ids in zip(
            outputs,
            inputs["input_ids"]
        ):

            generated = output[input_ids.shape[-1]:]

            answer = tokenizer.decode(
                generated,
                skip_special_tokens=True
            )

            answers.append(answer.strip())


        return answers

    def process_data(data_list):
        """
        Processes a list of data items. Each data item is a dictionary:
            {
                'question': str,
                'cand_relations': set(...),
                'gt_relations': set(...)
            }

        We build a prompt for GPT and then store GPT's response in a list.

        Inputs:
        data_list: list of dictionaries

        Output:
        A list of dictionaries, each containing the question, candidate relations,
        and the LLM response text.
        """
        results = []
        for i in tqdm(range(len(data_list)), desc="Processing data"):

            item = data_list[i][i]

            q = item["question"]
            cand_rels = list(item["cand_relations"])
            gt_relations = list(item["gt_relations"])

            user_prompt = build_prompt(q, cand_rels)
            messages = [
                system_message,
                {"role": "user", "content": user_prompt}
            ]
            #old version with OpenAI API call
            #response_text = gpt_call(messages)
            response_text = llm_call(messages)

            results.append({
                "question": q,
                "cand_relations": cand_rels,
                "response_text": response_text,
                "gt_relations": gt_relations
            })

        return results

    if split=='train' or split=='all':
        #########################################
        #########For the train split #############
        #########################################
        start_idx = 0 #reset
        print(colored(f"Processing training set (size={len(train_data)}).", 'green'))
        train_results = process_data(train_data)

        # Save all results to out_dir as pickle files
        with open(os.path.join(args.out_dir, "train_results.pkl"), "wb") as f:
            pickle.dump(train_results, f)

    if split=='val' or split=='all':
        #########################################
        #########For the val split #############
        #########################################
        start_idx = 0 #reset
        print(colored(f"Processing val set (size={len(val_data)}).", 'green'))
        val_results = process_data(val_data)

        # Save all results to out_dir as pickle files
        with open(os.path.join(args.out_dir, "val_results.pkl"), "wb") as f:
            pickle.dump(val_results, f)

    if split=='test' or split=='all':
        #########################################
        #########For the test split #############
        #########################################
        start_idx = 0 #reset
        print(colored(f"Processing test set (size={len(test_data)}).", 'green'))
        test_results = process_data(test_data)

        # Save all results to out_dir as pickle files
        with open(os.path.join(args.out_dir, "test_results.pkl"), "wb") as f:
            pickle.dump(test_results, f)

    print(colored("All results saved to output directory.", 'blue'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data_files/toy/processed",
    )
    parser.add_argument('--out_dir', type=str, default="data_files/toy/relationtargeting/", help='Directory to save the results')
    #parser.add_argument('--which_key', type=int, default=0, choices=[0,1,2], help='Which OpenAI key to use')
    parser.add_argument('--train_part', type=int, default=-1,
                        choices=[-1,0,1,2],
                        help='-1 means use the entire training set; 0,1,2 for splitting the dataset into thirds')
    parser.add_argument('--split', type=str, default='all',
                                choices=['train','val','test','all'], help='which split to process')

    args = parser.parse_args()
    chat(args)

