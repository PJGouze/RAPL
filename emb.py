#update: 03/08 16:52
import os
import torch
import sys
import pickle
from pathlib import Path

from datasets import load_dataset,load_from_disk
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset
from termcolor import colored

def get_emb(subset, text_encoder, save_file, batch_size=128):
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
    # emb_dict = dict()
    # for i in tqdm(range(len(subset))):
    #     id, q_text, text_entity_list, relation_list = subset[i]

    #     q_emb, entity_embs, relation_embs = text_encoder(
    #         q_text, text_entity_list, relation_list)
    #     emb_dict_i = {
    #         'q_emb': q_emb,
    #         'entity_embs': entity_embs,
    #         'relation_embs': relation_embs
    #     }
    #     emb_dict[id] = emb_dict_i

    # torch.save(emb_dict, save_file)
    

    print("Collecting unique entities and relations...")

    unique_entities = set()
    unique_relations = set()
    questions = []
    sample_ids = []

    for sample in subset:
        sample_id, q_text, entity_list, relation_list = sample

        sample_ids.append(sample_id)
        questions.append(q_text)

        unique_entities.update(entity_list)
        unique_relations.update(relation_list)

    unique_entities = sorted(unique_entities)
    unique_relations = sorted(unique_relations)

    print(f"{len(unique_entities)} unique entities")
    print(f"{len(unique_relations)} unique relations")

    ####################################################
    # Encode all entities once
    ####################################################

    entity_cache = {}

    for i in tqdm(
            range(0, len(unique_entities), batch_size),
            desc="Encoding entities"):

        batch = unique_entities[i:i + batch_size]
        emb = text_encoder.embed(batch)

        for entity, vec in zip(batch, emb):
            entity_cache[entity] = vec

    ####################################################
    # Encode all relations once
    ####################################################

    relation_cache = {}

    for i in tqdm(
            range(0, len(unique_relations), batch_size),
            desc="Encoding relations"):

        batch = unique_relations[i:i + batch_size]
        emb = text_encoder.embed(batch)

        for relation, vec in zip(batch, emb):
            relation_cache[relation] = vec

    ####################################################
    # Encode all questions in batches
    ####################################################

    print("Encoding questions...")

    question_embs = text_encoder.encode_questions(
        questions,
        batch_size=batch_size
    )

    ####################################################
    # Build final dictionary
    ####################################################

    emb_dict = {}

    for sample, q_emb in tqdm(
            zip(subset, question_embs),
            total=len(subset),
            desc="Building embedding dictionary"):

        sample_id, _, entity_list, relation_list = sample

        entity_embs = (
            torch.stack([entity_cache[e] for e in entity_list])
            if entity_list
            else torch.empty(0, q_emb.shape[-1])
        )

        relation_embs = (
            torch.stack([relation_cache[r] for r in relation_list])
            if relation_list
            else torch.empty(0, q_emb.shape[-1])
        )

        emb_dict[sample_id] = {
            "q_emb": q_emb,
            "entity_embs": entity_embs,
            "relation_embs": relation_embs,
        }

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
            ``"webqsp"``, ``"cwq"``, "toy" or "OntoOmimcsKG_stepX" and potentially custom datasets if added
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

        {DATA_DIR}/{dataset}/processed/
            train.pkl
            val.pkl
            test.pkl

        {DATA_DIR}/{dataset}/emb/{text_encoder_name}/
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
    
    encoder_config_dirs = {
        "mini": "mini_encoder",
        "biobert": "biobert_encoder",
    }

    if args.encoder not in encoder_config_dirs:
        raise ValueError(
            f"Unsupported encoder {args.encoder!r}. "
            f"Expected one of {sorted(encoder_config_dirs)}."
        )

    encoder_config_dir = encoder_config_dirs[args.encoder]

    config_file = (
        f"configs/emb/{encoder_config_dir}/{args.dataset}.yaml"
    )

    if not Path(config_file).is_file():
        raise FileNotFoundError(
            f"Embedding configuration not found: {config_file}"
        )

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
        PROJECT_ROOT = Path(__file__).resolve().parent
        DATA_DIR = PROJECT_ROOT / "data_files"

        with open(DATA_DIR / "toy" / "raw" / "train.pkl", "rb") as f:
            train_set = pickle.load(f)

        with open(DATA_DIR / "toy" / "raw" / "val.pkl", "rb") as f:
            val_set = pickle.load(f)

        with open(DATA_DIR / "toy" / "raw" / "test.pkl", "rb") as f:
            test_set = pickle.load(f)

    elif args.dataset == "OntoOmicsKG_step2":
            PROJECT_ROOT = Path(__file__).resolve().parent
            DATA_DIR = PROJECT_ROOT / "data_files"
    
            with open(DATA_DIR / "OntoOmicsKG_step2" / "raw" / "train.pkl", "rb") as f:
                train_set = pickle.load(f)
    
            with open(DATA_DIR / "OntoOmicsKG_step2" / "raw" / "val.pkl", "rb") as f:
                val_set = pickle.load(f)
    
            with open(DATA_DIR / "OntoOmicsKG_step2" / "raw" / "test.pkl", "rb") as f:
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

    text_encoder_name = config["text_encoder"]["name"]

    safe_encoder_name = text_encoder_name.replace("/", "__")

    if args.embedding_input_dir is None:
        save_dir = os.path.join(
            DATA_DIR,
            args.dataset,
            "embedding_inputs",
            safe_encoder_name,
        )
    else:
        save_dir = args.embedding_input_dir

    os.makedirs(save_dir, exist_ok=True)

    print(f"Embedding input directory : {save_dir}")

    raw_split_datasets = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    split_datasets = {}

    for split in args.splits:
        raw_subset = raw_split_datasets[split]

        if args.max_samples is not None:
            raw_subset = raw_subset[:args.max_samples]

        dataset_kwargs = {}

        if split == "test":
            dataset_kwargs.update(
                skip_no_topic=False,
                skip_no_ans=False,
            )

        split_datasets[split] = EmbInferDataset(
            raw_subset,
            entity_identifiers,
            os.path.join(
                save_dir,
                f"{split}.pkl",
            ),
            **dataset_kwargs,
        )
    
    if args.device < 0:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available."
            )

        device = torch.device(
            f"cuda:{args.device}"
        )

    print(f"Embedding device: {device}")
    
    if text_encoder_name == "gte-large-en-v1.5":
        from src.model.text_encoders.original_encoder import GTELargeEN

        text_encoder = GTELargeEN(device)

    elif text_encoder_name == "sentence-transformers/all-MiniLM-L6-v2":
        from src.model.text_encoders.mini_encoder import MiniLML6Encoder

        text_encoder = MiniLML6Encoder(
            text_encoder_name,
            device,
        )

    elif text_encoder_name == "dmis-lab/biobert-base-cased-v1.2":
        from src.model.text_encoders.biobert_encoder import BioBERTEncoder

        text_encoder = BioBERTEncoder(
            text_encoder_name,
            device,
        )

    else:
        raise NotImplementedError(text_encoder_name)
    
    if args.embedding_output_dir is None:
        emb_save_dir = os.path.join(
            DATA_DIR,
            args.dataset,
            "emb",
            text_encoder_name,
        )
    else:
        emb_save_dir = args.embedding_output_dir

    os.makedirs(emb_save_dir, exist_ok=True)

    print(f"Embedding model           : {text_encoder_name}")
    print(f"Embedding output directory: {emb_save_dir}")

    print(f"Selected splits: {args.splits}")
    print(f"Maximum samples per split: {args.max_samples}")

    for split in args.splits:
        subset = split_datasets[split]

        print(
            f"Embedding split={split!r}, "
            f"samples={len(subset)}"
        )

        get_emb(
            subset,
            text_encoder,
            os.path.join(
                emb_save_dir,
                f"{split}.pth",
            ),
        )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'toy', 'OntoOmicsKG_step2'], help='Dataset name')
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["mini", "biobert"],
        default="mini",
        help=(
            "Text encoder configuration to use. "
            "'mini' preserves the MiniLM baseline; "
            "'biobert' uses the isolated BioBERT configuration."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Dataset splits to embed.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples to embed per selected split. "
            "Use this for small pilot runs."
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help=(
            "CUDA device index. "
            "Use -1 for CPU."
        ),
    )

    parser.add_argument(
        "--embedding-output-dir",
        type=str,
        default=None,
        help=(
            "Explicit directory for generated embedding .pth files. "
            "If omitted, use the legacy path "
            "data_files/<dataset>/emb/<text_encoder_name>/."
        ),
    )

    parser.add_argument(
        "--embedding-input-dir",
        type=str,
        default=None,
        help=(
            "Explicit directory for EmbInferDataset intermediate pickle files. "
            "If omitted, use the legacy encoder-derived directory."
        ),
    )

    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be strictly positive.")
    
    main(args)

