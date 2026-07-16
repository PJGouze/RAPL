import os
import torch
import sys
import pickle

from datasets import load_dataset,load_from_disk
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset
from termcolor import colored

def get_emb(subset, text_encoder, save_file):
    """
    Generate and save text embeddings for a dataset subset.

    This function iterates over all samples in a dataset subset, computes
    embeddings for the question, entities, and relations using the provided
    text encoder, and stores the resulting embeddings in a dictionary indexed
    by sample identifier. The complete embedding dictionary is then serialized
    to disk using ``torch.save``.

    Parameters
    ----------
    subset : Sequence
        Dataset subset containing samples. Each sample is expected to be a
        tuple of the form::

            (id, q_text, text_entity_list, relation_list)

        where:

        - ``id`` is a unique sample identifier.
        - ``q_text`` is the question text.
        - ``text_entity_list`` is a list of textual entity names.
        - ``relation_list`` is a list of textual relation names.

    text_encoder : callable
        A callable object that encodes textual inputs into vector
        representations. It must accept three arguments:

        ``(q_text, text_entity_list, relation_list)``

        and return a tuple::

            (q_emb, entity_embs, relation_embs)

        where each output is a tensor or array containing the corresponding
        embeddings.

    save_file : str or pathlib.Path
        Path where the generated embedding dictionary will be saved.

    Returns
    -------
    None
        The function does not return any value. The generated embeddings are
        saved to ``save_file``.

    Notes
    -----
    The saved dictionary has the following structure::

        {
            sample_id: {
                "q_emb": question_embedding,
                "entity_embs": entity_embeddings,
                "relation_embs": relation_embeddings
            },
            ...
        }

    The resulting file can later be loaded with ``torch.load`` for inference
    or downstream processing.
    """
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]

        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list)
        emb_dict_i = {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs
        }
        emb_dict[id] = emb_dict_i

    torch.save(emb_dict, save_file)

def main(args):
    """
    Main pipeline for pre-computing text embeddings for a knowledge graph
    retrieval dataset.

    This function loads the experiment configuration, prepares the selected
    dataset, initializes the text encoder model, computes embeddings for
    questions, entities, and relations, and saves the resulting embeddings
    to disk.

    The pipeline consists of the following steps:

    1. Load the YAML configuration file corresponding to the selected dataset
        and text encoder.
    2. Load the dataset splits (train, validation, test) from Hugging Face
        datasets.
    3. Load the list of valid entity identifiers used to filter and process
        graph entities.
    4. Convert raw dataset samples into ``EmbInferDataset`` objects compatible
        with the embedding generation pipeline.
    5. Initialize the text encoder model on the available CUDA device.
    6. Generate question, entity, and relation embeddings for each dataset
        split.
    7. Save the computed embeddings as PyTorch files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

        Attributes
        ----------
        dataset : str
            Name of the dataset to process. Supported datasets are:
            ``"webqsp"``, ``"cwq"`` or "toy" and potentially custom datasets if added
            to the loading logic.

    Returns
    -------
    None
        This function does not return a value. It creates processed dataset
        files and embedding files on disk.

    Raises
    ------
    FileNotFoundError
        If the configuration file, entity identifier file, or any required
        dataset resource cannot be found.

    NotImplementedError
        If the requested text encoder is not implemented.

    RuntimeError
        If CUDA is requested but unavailable, or if an error occurs during
        embedding computation.

    Notes
    -----
    The generated files are stored under::

        data_files/{dataset}/processed/
            train.pkl
            val.pkl
            test.pkl

        data_files/{dataset}/emb/{text_encoder_name}/
            train.pth
            val.pth
            test.pth

    The embedding files contain dictionaries mapping sample identifiers to
    their corresponding embeddings:

    .. code-block:: python

        {
            sample_id: {
                "q_emb": question_embedding,
                "entity_embs": entity_embeddings,
                "relation_embs": relation_embeddings
            }
        }

    Examples
    --------
    Run embedding generation for the WebQSP dataset from the command line:

    >>> python script.py --dataset webqsp

    """
    # existing implementation
    # Modify the config file for advanced settings and extensions.
    
    text_encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
    config_file = f'configs/emb/mini_encoder/{args.dataset}.yaml'
    config = load_yaml(config_file)
    print (colored(config,'red'))
    
    torch.set_num_threads(config['env']['num_threads'])

    
    #! load from huggingface
    if args.dataset == 'cwq' or args.dataset == 'webqsp':
        if args.dataset == 'cwq':
            input_file = os.path.join('rmanluo', 'RoG-cwq')
        elif args.dataset == 'webqsp':
            input_file = os.path.join('ml1996', 'webqsp')
        
        train_set = load_dataset(input_file, split='train')
        val_set = load_dataset(input_file, split='validation')
        test_set = load_dataset(input_file, split='test')

    elif args.dataset == "toy":

        with open("/home/pjgouze/Documents/RAPL-env/RAPL/data_files/toy/raw/train.pkl", "rb") as f:
            train_set = pickle.load(f)
        with open("/home/pjgouze/Documents/RAPL-env/RAPL/data_files/toy/raw/val.pkl", "rb") as f:
            val_set = pickle.load(f)
        with open("/home/pjgouze/Documents/RAPL-env/RAPL/data_files/toy/raw/test.pkl", "rb") as f:
            test_set = pickle.load(f)

    elif args.dataset == "prot_sepsis":
        raise NotImplementedError
    
    else:

        print('specify the dataset')

    
    entity_identifiers = []
    #======== Joining the entity identifier file =======#
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    #==================================================#

    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)
    
    #device = torch.device('cuda:0')
    device = torch.device("cpu")
    
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from RAPL.src.model.text_encoders.original_encoder import GTELargeEN
        text_encoder = GTELargeEN(device)
    elif text_encoder_name == 'sentence-transformers/all-MiniLM-L6-v2':
        from RAPL.src.model.text_encoders.mini_encoder import MiniLML6Encoder
        text_encoder = MiniLML6Encoder(text_encoder_name, device)

    
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)
    
    print ('process val emb')
    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, 'train.pth'))
    get_emb(val_set, text_encoder, os.path.join(emb_save_dir, 'val.pth'))
    get_emb(test_set, text_encoder, os.path.join(emb_save_dir, 'test.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'toy'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
