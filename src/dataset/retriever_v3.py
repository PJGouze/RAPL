#update: 09/08 12:12
import networkx as nx
import numpy as np
import os
import sys
import pickle
import torch
import torch.nn.functional as F

from tqdm import tqdm

class RetrieverDataset:

    """
        Initialize the retrieval dataset.

        This constructor loads a preprocessed retrieval dataset from a cached
        pickle file when available. Otherwise, it preprocesses the raw dataset by
        extracting graph structures, computing weak supervision signals (triple
        scores), extracting translated paths and relations, assembling all
        information into a unified representation, and caching the result for
        future use.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing the dataset settings, including
            the dataset name and other preprocessing parameters.

        split : str
            Dataset split to load (e.g., ``"train"``, ``"val"``, or ``"test"``).

        skip_no_path : bool, optional
            Whether to discard samples for which no valid path can be found
            between the topic entity and the answer entity during preprocessing.
            Default is True.

        Attributes
        ----------
        nx_graphs : list
            List of NetworkX graph objects associated with the processed samples.

        first_processed_samples : list
            List of processed samples containing all information required for
            retrieval, including graph data, supervision signals, and translated
            paths/relations.

        emb_dict : dict or None
            Dictionary of precomputed entity embeddings. Set to ``None`` when
            embeddings are not used.

        Notes
        -----
        The processed dataset is cached as a pickle file under::

            data_files/<dataset_name>/processed/<split>_retrieval.pkl

        to avoid repeating the preprocessing pipeline on subsequent runs.
        """
        

    def __init__(
        self,
        config: str,
        split: str,
        skip_no_path=True
    ):

        self.nx_graphs = []

        dataset_name = config['dataset']['name']

        RetrieverDatasetPath = (
            f'data_files/{dataset_name}/processed/'
            f'{split}_retrieval.pkl'
        )

        TripleScorePath = (
            f'data_files/{dataset_name}/triple_scores/'
            f'{split}.pth'
        )

        # ============================================================
        # 1. Load existing retrieval dataset if available
        # ============================================================

        if os.path.exists(RetrieverDatasetPath):

            print(
                f'Loading first_processed_samples from '
                f'{RetrieverDatasetPath}'
            )

            with open(RetrieverDatasetPath, 'rb') as f:
                self.first_processed_samples = pickle.load(f)

            print('Loaded first_processed_samples from pkl')

        else:

            print('Creating the first processed samples')

            non_processed_samples = self._load_non_processed(
                dataset_name,
                split
            )

            triple_score_dict = self._get_triple_scores(
                dataset_name,
                split,
                non_processed_samples
            )

            path_relation_dict = (
                self._get_translated_paths_and_relations(
                    dataset_name,
                    split,
                    non_processed_samples
                )
            )

            emb_dict = None
            self.emb_dict = emb_dict

            self.first_processed_samples = (
                self._creating_first_processed_samples(
                    non_processed_samples,
                    triple_score_dict,
                    emb_dict,
                    skip_no_path,
                    path_relation_dict
                )
            )

            try:
                os.makedirs(
                    os.path.dirname(RetrieverDatasetPath),
                    exist_ok=True
                )

                with open(
                    RetrieverDatasetPath,
                    'wb'
                ) as f:

                    pickle.dump(
                        self.first_processed_samples,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

                print(
                    f'Saved first_processed_samples to '
                    f'{RetrieverDatasetPath}'
                )

            except Exception as e:
                print(e)

        # ============================================================
        # 2. Independently generate triple scores if missing
        # ============================================================

        if not os.path.exists(TripleScorePath):

            print(
                f'Triple score file not found:\n'
                f'    {TripleScorePath}'
            )

            print(
                'Generating triple scores independently...'
            )

            non_processed_samples = self._load_non_processed(
                dataset_name,
                split
            )

            self._get_triple_scores(
                dataset_name,
                split,
                non_processed_samples
            )

            print(
                f'Triple scores generated for {split}.'
            )

        else:

            print(
                f'Triple score file already exists:\n'
                f'    {TripleScorePath}'
            )
    def _load_non_processed(
        self,
        dataset_name: str,
        split: str
    ):
     
        """
        Load the non processed dataset from a pickle file containing query, answer and the graph associcated.

        This method reads the serialized dataset corresponding to the specified
        dataset name and split, and returns it as a Python object.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        split : str
            Dataset split to load (e.g., ``"train"``, ``"dev"``, or ``"test"``).

        Returns
        -------
        list
            List of  samples loaded from the pickle file. Each element is
            a dictionary containing the preprocessed information associated with a
            question and its corresponding knowledge graph.

        Notes
        -----
        The expected file location is::

            data_files/<dataset_name>/processed/<split>.pkl
        """
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)
        
    def _load_processed_retrieval(
        self,
        dataset_name,
        split
    ):
       
        """
        Load the preprocessed retrieval dataset from a pickle file.

        This method reads the serialized retrieval dataset corresponding to the
        specified dataset name and split, and returns it as a Python object.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        split : str
            Dataset split to load (e.g., ``"train"``, ``"dev"``, or ``"test"``).

        Returns
        -------
        list of dict
            List of processed retrieval samples. Each dictionary contains the
            information required by the retrieval model, such as the question,
            graph representation, supervision signals, translated paths and
            relations, and any additional preprocessing outputs.

        Notes
        -----
        The expected file location is::

            data_files/<dataset_name>/processed/<split>_retrieval.pkl
        """
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}_retrieval.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name: str,
        split: str,
        non_processed_samples: list
    ):
      
        """
        Compute or load weak supervision signals (triple scores) for graph-based retrieval.

        This method either loads precomputed triple scores from disk or computes them
        from scratch by extracting paths between topic entities and answer entities
        for each sample in the dataset. The results are cached to speed up future runs.

        For each sample, it computes:
        - triple-level scores used as weak supervision for graph traversal or ranking
        - the maximum path length between topic and answer entities

        Parameters
        ----------
        dataset_name : str
            Name of the dataset used to locate the cache directory.

        split : str
            Dataset split (e.g., ``"train"``, ``"dev"``, ``"test"``).

        non_processed_samples : list of dict
            List of non yet processed samples. Each sample must contain at least an
            ``"id"`` field and graph-related information required to compute paths.

        Returns
        -------
        dict
            Dictionary indexed by sample id. Each entry is a dictionary with:
            
            - ``triple_scores``: scoring information for triples in the graph
            - ``max_path_length``: maximum candidate path length between query and answer entities

        Notes
        -----
        If a cached file exists at::

            data_files/<dataset_name>/triple_scores/<split>.pth

        it is loaded directly using ``torch.load``. Otherwise, the scores are computed
        using `_extract_paths_and_score` and saved for future reuse.
        """
        
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')
        
        #Testing if the file exists, loading it if so
        if os.path.exists(save_file):
            print (f"Loading triple_score_dict from {save_file}")
            return torch.load(save_file)

        triple_score_dict = dict()

        for i in tqdm(range(len(non_processed_samples))):
            sample_i = non_processed_samples[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i
            )
            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }
        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    def _extract_paths_and_score(
        self,
        sample: dict
    ):

        """
        Extract shortest paths between question and answer entities and compute
        triple-level supervision scores.

        This method builds a NetworkX graph from the sample, computes shortest
        paths between all question entities and answer entities, converts these
        entity-level paths into triple-level paths, and finally computes
        supervision scores for each triple based on its occurrence in paths.

        Parameters
        ----------
        sample : dict
            A single dataset sample containing at least:
            - ``h_id_list``: list of head entity IDs (graph nodes)
            - ``r_id_list``: list of relation IDs (graph edges)
            - ``t_id_list``: list of tail entity IDs (graph nodes)
            - ``q_entity_id_list``: list of question entity IDs
            - ``a_entity_id_list``: list of answer entity IDs

        Returns
        -------
        triple_scores : array-like
            Scores assigned to triples, used as weak supervision signals for
            ranking or retrieval over the knowledge graph.

        max_path_length : int or None
            Maximum number of triples observed in any shortest path between
            question and answer entities. Returns None if no path exists.

        Notes
        -----
        The method proceeds in three steps:

        1. Graph construction:
        A NetworkX graph is built from the triples in the sample using
        `_get_nx_g`.

        2. Path extraction:
        All shortest paths between question entities and answer entities are
        computed using `_shortest_path`.

        3. Path transformation and scoring:
        Entity-level paths are converted into triple-level paths using edge
        identifiers, and `_score_triples` is used to compute supervision
        signals.
        """
        #building the graph based on the triplets stored along with the query
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        # Finding all the shortest path to create 
        # a reference for the candidate paths
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
    
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Convert raw entity-based path to triple-based path.
        path_list = []
        for path in path_list_:
            current_path_length = len(path) - 1
            max_path_length = max(max_path_length, current_path_length)
            triples_path = []
            # Adding the triples to concatenate into a path
            for i in range(current_path_length):
                h_id_i = path[i]
                t_id_i = path[i+1]
                #weird formulation here but works because we don't use a MultiDigraph
                triple_id_i_list = [nx_g[h_id_i][t_id_i]['triple_id']]
                triples_path.append(triple_id_i_list)
            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(path_list, num_triples)
        
        return triple_scores, max_path_length

    def _get_nx_g(
        self,
        h_id_list: list,
        r_id_list: list,
        t_id_list: list
    ):
    
        """
        Build a directed NetworkX graph from triples and store it in the dataset.

        This method constructs a directed graph where nodes correspond to entities
        and edges correspond to knowledge graph triples. Each edge is annotated
        with both a triple identifier and a relation identifier.

        The resulting graph is also stored in `self.nx_graphs` for later reuse.

        Parameters
        ----------
        h_id_list : list
            List of head entity IDs for each triple.

        r_id_list : list
            List of relation IDs corresponding to each triple.

        t_id_list : list
            List of tail entity IDs for each triple.

        Returns
        -------
        nx.DiGraph
            A directed graph where:
            - nodes are entity IDs
            - edges represent triples
            - each edge contains:
                - ``triple_id``: index of the triple in the input lists
                - ``relation_id``: identifier of the relation type

        Notes
        -----
        The graph is built using NetworkX (`nx.DiGraph`) and appended to
        `self.nx_graphs` for potential debugging or analysis of intermediate
        graph constructions.
        """

        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
        self.nx_graphs.append(nx_g)
        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        """
        Compute all shortest paths between two entities in a directed graph,
        considering both forward and backward directions.

        This method uses NetworkX shortest path algorithms to retrieve all
        shortest paths between a question entity and an answer entity. It
        searches in both directions (q → a and a → q), merges the results,
        and retains only the paths with minimal length.

        Parameters
        ----------
        nx_g : nx.DiGraph
            Directed knowledge graph built from triples.

        q_entity_id : int or str
            Entity ID corresponding to the question (source node).

        a_entity_id : int or str
            Entity ID corresponding to the answer (target node).

        Returns
        -------
        list of list
            A list of shortest paths, where each path is represented as a list
            of entity IDs. Only paths with minimal length across both directions
            are returned.

        Notes
        -----
        - The method first attempts to compute shortest paths in the forward
        direction (q_entity_id → a_entity_id).
        - It then attempts the reverse direction (a_entity_id → q_entity_id).
        - If both directions return valid paths, only those with minimal length
        among all candidates are kept.
        - If no path exists in one or both directions, the method gracefully
        falls back to the available paths without raising an exception.
        """
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        #if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            #return full_paths
        
        # Only keep minimal ones if both directions exist
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        return refined_paths

    def _score_triples(
        self,
        path_list: list,
        triple_count: int
    ):
        """
        Assign binary supervision scores to triples based on their occurrence in paths.

        This method builds a binary score vector over all triples in the graph.
        A triple is assigned a score of 1.0 if it appears in at least one of the
        provided paths, and 0.0 otherwise.

        Parameters
        ----------
        path_list : list
            List of paths, where each path is represented as a sequence of triples.
            Each element typically contains lists of triple IDs.

        triple_count : int
            Total number of triples in the graph. Defines the size of the output
            score vector.

        Returns
        -------
        torch.Tensor
            A 1D tensor of shape (triple_count) containing binary scores:
            - 1.0 if the triple appears in any path
            - 0.0 otherwise

        Notes
        -----
        This scoring mechanism provides weak supervision for graph-based
        retrieval models by highlighting triples that lie on at least one
        shortest path between question and answer entities.
        """

        triple_scores = torch.zeros(triple_count)
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.0
        return triple_scores

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        
        """
        Load precomputed entity embeddings from disk.

        This method loads a dictionary of precomputed embeddings associated with
        entities in the dataset. The embeddings are expected to have been generated
        using a specific text encoder and stored in a hierarchical directory
        structure.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset used to locate the embedding files.

        text_encoder_name : str
            Name of the text encoder used to generate the embeddings
            (e.g., BERT, SentenceTransformer, etc.).

        split : str
            Dataset split for which embeddings are loaded
            (e.g., ``"train"``, ``"dev"``, ``"test"``).

        Returns
        -------
        dict
            Dictionary mapping entity identifiers to their corresponding embedding
            vectors (typically stored as tensors).

        Notes
        -----
        The embeddings are loaded from:

            data_files/<dataset_name>/emb/<text_encoder_name>/<split>.pth

        using ``torch.load`` without additional processing.
        """
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)
        return dict_file

    # -------------------------------------------------------------------------
    # NEW CODE STARTS HERE
    # -------------------------------------------------------------------------

    def cut_off_paths(self, nx_g, q_entity_id, a_entity_id, tolerance=2):
        """
        This function computes the shortest path length d from q_entity_id
        to a_entity_id (one direction only, no reverse). Then uses
        nx.all_simple_paths to collect all simple paths whose length in edges
        is between d and d + tolerance.

        Parameters
        ----------
        nx_g : nx.DiGraph
            The directed graph.
        q_entity_id : int
            Source entity ID.
        a_entity_id : int
            Target entity ID.
        tolerance : int
            Allowed extra path length beyond the shortest path.

        Returns
        -------
        list_of_paths : list
            A list of paths (each path is a list of node-IDs) whose edge-length
            is in [d, d + tolerance].
        """
        try:
            # Get a single shortest path (one-direction only, no backward).
            shortest_paths = nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id)
            shortest_paths = list(shortest_paths)
            if len(shortest_paths)>50 or len(shortest_paths[0])>5:
                return shortest_paths
        except:
            return []  # no path from q_entity_id to a_entity_id

        # Among all shortest paths from q_entity_id -> a_entity_id, pick the length
        # of the first one (any would do, they are all minimal).
        if len(shortest_paths) == 0:
            return []
        min_len = len(shortest_paths[0]) - 1  # number of edges

        # Now gather all simple paths with edge-length up to min_len + tolerance
        cutoff = min_len + tolerance
        all_paths_within_cutoff = nx.all_simple_paths(
            nx_g,
            source=q_entity_id,
            target=a_entity_id,
            cutoff=cutoff
        )
        # Filter out those that have edges < min_len or edges > min_len + tolerance
        list_of_paths = []
        for path in all_paths_within_cutoff:
            length_in_edges = len(path) - 1
            if min_len <= length_in_edges <= (min_len + tolerance):
                list_of_paths.append(path)

        return list_of_paths

    def translate_paths(self, nx_g, paths, sample):
        """
        This function translates a list of paths (each path is a list of node-IDs)
        into textual format and collects related relation-IDs.

        Parameters
        ----------
        nx_g : nx.DiGraph
            The directed graph.
        paths : list of lists
            A list of paths, each path is a list of entity-IDs (e.g. [1, 5, 2]).
        sample : dict
            A processed sample containing 'id2entities' and 'id2relations',
            among other fields.

        Returns
        -------
        translated_paths : list of str
            A list of paths in textual form, e.g. for path [1,5,2],
            "entity1 -> relationX -> entity5 -> relationY -> entity2"
        reasoning_paths : list of list
            A list of only the relations in textual form per path, e.g.
            for [1, 5, 2] => ["relationX", "relationY"].
        distances : list of int
            A list of path lengths in edges. E.g. if paths=[[1,5,2],[1,7,4,9]],
            then distances=[2,3].
        relation_ids_tensor : list of torch.Tensor
            A list of variable-length Tensors containing relation IDs
            for each path. You could pad them if you need a single 2D tensor.
        """
        
        id2entity = sample['id2entities']
        id2relation = sample['id2relations']

        translated_paths = []
        reasoning_paths = []
        distances = []
        relation_ids = []  # will be a list of lists

        for path in paths:
            # Number of edges = number of "h->t" transitions
            path_len = len(path) - 1
            distances.append(path_len)

            # Collect textual expansions and relation IDs
            path_text = []
            rels_text = []
            current_rel_ids = []

            # Initialize the textual path with the first entity
            if path:  # ensure path is not empty
                path_text.append(id2entity[path[0]])

            for i in range(path_len):
                h_id = path[i]
                t_id = path[i+1]
                rel_id = nx_g[h_id][t_id]['relation_id']

                # Append the relation text
                relation_text = id2relation[rel_id]
                path_text.append(relation_text)  # "-> relation ->"
                rels_text.append(relation_text)

                current_rel_ids.append(rel_id)

                # Append the tail entity text so we see each intermediate node
                path_text.append(id2entity[t_id])  # "-> entity_t ->"

            # Convert path_text into a single string
            # e.g. "Justin Bieber -> film.producer.film -> someEntity -> film.actor.film -> ..."
            translated_paths.append(" -> ".join(path_text))

            # For reasoning_paths, we only keep the relations
            # If you want more detail (including entities), you can store them here instead.
            reasoning_paths.append(rels_text)

            # Append the current path's relation IDs to the big list
            relation_ids.append(current_rel_ids)

        # Convert each list of relation IDs to a Tensor.
        # If you need a single 2D tensor, you can pad them first.
        relation_ids_tensor = [torch.tensor(rids) for rids in relation_ids]
        # print (translated_paths)
        # print (reasoning_paths)
        # print (distances)
        return translated_paths, reasoning_paths, distances, relation_ids_tensor
      
    def _extract_translated_path_and_relations(
            self, 
            sample: dict, 
            tolerance=2):
        """
        Extract and translate constrained shortest paths between question and answer entities.

        This method constructs a directed knowledge graph from the input sample,
        retrieves all forward paths between question and answer entities within a
        given distance tolerance, and then translates these paths into structured
        representations (e.g., relations, reasoning chains, distances).

        Parameters
        ----------
        sample : dict
            A dataset sample containing at least:
            - ``h_id_list``: head entity IDs
            - ``r_id_list``: relation IDs
            - ``t_id_list``: tail entity IDs
            - ``q_entity_id_list``: question entity IDs
            - ``a_entity_id_list``: answer entity IDs

        tolerance : int, optional
            Maximum allowed deviation from the shortest path length when selecting
            candidate paths. Default is 2.

        Returns
        -------
        translated_paths : list
            Human-readable or symbolic representation of paths (e.g., relation
            sequences or translated triples).

        reasoning_paths : list
            Structured representations of reasoning chains derived from paths.

        distances : list
            Distances (number of hops) associated with each extracted path.

        relation_ids_tensor : list or tensor
            Encoded relation identifiers associated with each path.

        Notes
        -----
        - Only forward paths (q → a) are considered.
        - Paths are first filtered using a bounded shortest-path approximation
        via `cut_off_paths`.
        - If no valid path is found, the function returns empty lists for all outputs.
        - Final path representations are produced using `translate_paths`,
        which maps entity-level paths into relation-level reasoning structures.
        """
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # We'll collect all paths from q_entity -> a_entity with tolerance cutoff
        all_paths = []
        for q_id in sample['q_entity_id_list']:
            for a_id in sample['a_entity_id_list']:
                # Only forward direction, as per your requirement.
                # This yields paths in [d, d + tolerance].
                forward_paths = self.cut_off_paths(nx_g, q_id, a_id, tolerance)
                if len(forward_paths) > 0:
                    all_paths.extend(forward_paths)
        if len(all_paths) == 0:
            # No valid paths found, return default empty.
            return [], [], [], []

        # Translate them
        translated_paths, reasoning_paths, distances, relation_ids_tensor = \
            self.translate_paths(nx_g, all_paths, sample)

        return translated_paths, reasoning_paths, distances, relation_ids_tensor

    def _get_translated_paths_and_relations(
        self,
        dataset_name: str,
        split: str,
        processed_dict_list: list
    ):
       
        """
        Compute translated paths and relation-level representations for all samples.

        This method iterates over the dataset and applies
        `_extract_translated_path_and_relations` to each sample in order to
        extract structured path information, including translated paths,
        reasoning chains, distances, and relation identifiers.

        Unlike `_get_triple_scores`, this method does not use caching by default
        and recomputes all outputs at each call.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset (kept for interface consistency; not used directly).

        split : str
            Dataset split (e.g., ``"train"``, ``"dev"``, ``"test"``).

        processed_dict_list : list of dict
            List of samples, each containing at least:
            - ``id``: unique sample identifier
            - graph-related fields required for path extraction

        Returns
        -------
        dict
            Dictionary indexed by sample ID. Each entry contains:

            - ``translated_paths``: translated or symbolic representations of paths
            - ``reasoning_paths``: structured reasoning chains derived from paths
            - ``distances``: hop distances of extracted paths
            - ``relation_ids``: encoded relation identifiers for each path

        Notes
        -----
        - This method relies on `_extract_translated_path_and_relations`
        with a fixed tolerance of 2.
        - No disk caching is performed (all computations are recomputed at runtime).
        - The output is used to enrich retrieval training with path-level signals.
        """

        # You can optionally save to disk if you want, here we simply compute them.
        path_relation_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            tpaths, rpaths, distances, rel_ids = self._extract_translated_path_and_relations(
                sample_i, tolerance=2
            )
            path_relation_dict[sample_i_id] = {
                'translated_paths': tpaths,
                'reasoning_paths': rpaths,
                'distances': distances,
                'relation_ids': rel_ids
            }

        return path_relation_dict

    # -------------------------------------------------------------------------
    # END OF NEW CODE
    # -------------------------------------------------------------------------

    def _creating_first_processed_samples(
        self,
        first_processed_samples,
        triple_score_dict,
        emb_dict,
        skip_no_path,
        path_relation_dict  # NEW input
    ):
        
       
        """
        Assemble the final retrieval dataset by merging graph structures,
        supervision signals, and path-level annotations.

        This method builds the final `first_processed_samples` used for training or
        inference by combining:
        - raw processed samples
        - triple-level supervision signals
        - graph structures (NetworkX graphs)
        - optional path and relation-level information
        - optional entity embeddings (currently unused for memory efficiency)

        It also applies filtering rules (e.g., skipping samples without valid paths)
        and constructs auxiliary features such as topic entity masks.

        Parameters
        ----------
        first_processed_samples : list of dict
            Raw first processed dataset samples.

        triple_score_dict : dict
            Dictionary mapping sample IDs to:
            - ``triple_scores``: weak supervision over graph triples
            - ``max_path_length``: maximum shortest path length

        emb_dict : dict or None
            Optional dictionary of entity embeddings (not used in current version
            to reduce memory consumption).

        skip_no_path : bool
            If True, samples with no valid path (or max_path_length == 0/None)
            are discarded.

        path_relation_dict : dict
            Dictionary mapping sample IDs to path-level annotations, including:
            - translated paths
            - reasoning paths
            - path distances
            - relation ID tensors

        Returns
        -------
        None
            The method updates `self.first_processed_samples` in place.

        Side Effects
        ------------
        - Populates `self.first_processed_samples` with fully assembled samples.
        - Stores associated NetworkX graphs in each sample under `nx_graph`.
        - Adds:
            - `target_triple_probs`
            - `max_path_length`
            - `topic_entity_one_hot`
            - `translated_paths`
            - `reasoning_paths`
            - `path_distances`
            - `relation_id_tensor`
        - Prints dataset statistics:
            - number of skipped samples
            - distribution of relevant triples (median / mean / max)

        Notes
        -----
        - Duplicate answer entities are removed via set conversion.
        - Topic entities are encoded as a one-hot mask over all entities in the sample.
        - Embeddings (`emb_dict`) are intentionally not used to reduce memory usage.
        - This function is the final step of dataset construction before caching.
        """
        self.first_processed_samples = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(first_processed_samples))):
            sample_i = first_processed_samples[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['nx_graph'] = self.nx_graphs[i]

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue
            
            # Embeddings
            # sample_i.update(emb_dict[sample_i_id]) # no need to save emb, to save memory 

            # Clean up answer entity duplicates
            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            # print("q_entity_id_list:", sample_i['q_entity_id_list'])
            # print("topic_entity_mask:", topic_entity_mask)
            # print("unique:", torch.unique(topic_entity_mask))
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()
            # print(
            #     "one hot unique:",
            #     torch.unique(topic_entity_one_hot, dim=0)
            # )

            # print(
            #     "topic entity row:",
            #     topic_entity_one_hot[sample_i['q_entity_id_list']]
            # )

            # print(
            #     "first rows:",
            #     topic_entity_one_hot[:5]
            # )

            # -------------------------------------------------------------
            # NEW: attach the path/relations info from path_relation_dict
            # -------------------------------------------------------------
            if sample_i_id in path_relation_dict:
                sample_i['translated_paths'] = path_relation_dict[sample_i_id]['translated_paths']
                sample_i['reasoning_paths'] = path_relation_dict[sample_i_id]['reasoning_paths']
                sample_i['path_distances'] = path_relation_dict[sample_i_id]['distances']
                sample_i['relation_id_tensor'] = path_relation_dict[sample_i_id]['relation_ids']
            else:
                # If not found in dict, default to empty
                sample_i['translated_paths'] = []
                sample_i['reasoning_paths'] = []
                sample_i['path_distances'] = []
                sample_i['relation_id_tensor'] = []
            # -------------------------------------------------------------

            self.first_processed_samples.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')
        return first_processed_samples
    
    def __len__(self):
        return len(self.first_processed_samples)
    
    def __getitem__(self, i):
        return self.first_processed_samples[i]

def collate_retriever(data):
    """
    Collate function for the retriever DataLoader.

    This function prepares a single sample (batch size = 1) for input into the
    retrieval model. It converts graph structure components into tensors and
    aggregates all necessary embeddings, path information, and metadata into
    a structured output.

    Parameters
    ----------
    data : list of dict
        Batch of dataset samples. This implementation assumes batch size is 1
        and therefore uses only the first element.

    Returns
    -------
    h_id_tensor : torch.Tensor
        Tensor of head entity IDs in the knowledge graph.

    r_id_tensor : torch.Tensor
        Tensor of relation IDs corresponding to edges in the graph.

    t_id_tensor : torch.Tensor
        Tensor of tail entity IDs in the knowledge graph.

    q_emb : torch.Tensor or array-like
        Embedding of the question.

    entity_embs : torch.Tensor or array-like
        Embeddings of all entities in the sample graph.

    num_non_text_entities : int
        Number of entities that are not associated with textual descriptions.

    relation_embs : torch.Tensor or array-like
        Embeddings of relations in the knowledge graph.

    translated_paths : list
        Translated representations of extracted paths.

    reasoning_paths : list
        Structured reasoning paths derived from graph traversal.

    path_distances : list
        Distances (in hops) associated with extracted paths.

    relation_id_tensor : tensor or list
        Encoded relation identifiers for each path.

    id2entities : dict
        Mapping from entity IDs to entity representations or names.

    id2relations : dict
        Mapping from relation IDs to relation representations or names.

    non_text_entity_list : list
        List of entities without textual descriptions.

    text_entity_list : list
        List of entities with textual descriptions.

    nx_graph : networkx.DiGraph
        Graph representation of the sample knowledge graph.

    Notes
    -----
    - This collate function is designed for batch size = 1.
    - It performs no padding or batching across multiple samples.
    - It directly returns structured components required by the retriever model.
    - Graph structure and embeddings are preserved in their original form.
    """
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['translated_paths'], sample['reasoning_paths'], sample['path_distances'], sample['relation_id_tensor'],\
        sample['id2entities'], sample['id2relations'],sample['non_text_entity_list'],sample['text_entity_list'],sample['nx_graph']



