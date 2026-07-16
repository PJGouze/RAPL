import os
import pickle

from tqdm import tqdm

class EmbInferDataset:
    def __init__(
        self,
        raw_set,
        entity_identifiers,
        save_path,
        skip_no_topic=True,
        skip_no_ans=True
    ):
        """
        Dataset for preparing knowledge graph samples for embedding inference.

        This class preprocesses a raw question-answering dataset by converting
        entities and relations into integer identifiers, separating entities with
        meaningful textual descriptions from identifier-only entities, and caching
        the processed dataset to disk.

        Each processed sample contains all information required to generate entity
        and relation embeddings and to build the corresponding graph structure.

        Parameters
        ----------
        raw_set : list
            List of raw samples. Each sample must contain at least the following
            fields:

            - ``id`` : sample identifier.
            - ``question`` : natural language question.
            - ``graph`` : list of triples ``(head, relation, tail)``.
            - ``q_entity`` : list of question/topic entities.
            - ``a_entity`` : list of answer entities.
            - ``answer`` : list of answer entities (must be identical to
            ``a_entity``).

        entity_identifiers : set
            Set of entity identifiers that do not have meaningful textual
            descriptions (for example Freebase identifiers such as ``m.06w2sn5``).
            These entities are separated from textual entities so that they can be
            embedded differently.

        save_path : str
            Path to the pickle file used to cache the processed dataset. If the
            file already exists, the processed data are loaded directly instead of
            being recomputed.

        skip_no_topic : bool, default=True
            Whether to discard samples that do not contain any question/topic
            entity after preprocessing.

        skip_no_ans : bool, default=True
            Whether to discard samples that do not contain any answer entity after
            preprocessing.

        Attributes
        ----------
        processed_dict_list : list
            List of processed samples.

        no_topic_cnt : int
            Number of discarded samples without topic entities.

        no_ans_cnt : int
            Number of discarded samples without answer entities.
        """
        self.processed_dict_list = self._process(
            raw_set,
            entity_identifiers,
            save_path)
        
        self.skip_no_topic = skip_no_topic
        self.skip_no_ans = skip_no_ans
        
        processed_dict_list = []
        self.no_topic_cnt = 0
        self.no_ans_cnt = 0
        for processed_dict_i in self.processed_dict_list:
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                self.no_topic_cnt += 1
                continue
            
            if (len(processed_dict_i['a_entity_id_list']) == 0) and skip_no_ans:
                self.no_ans_cnt += 1
                continue
            
            processed_dict_list.append(processed_dict_i)
        self.processed_dict_list = processed_dict_list
        
        print(f'# raw samples: {len(raw_set)} | # processed samples: {len(self.processed_dict_list)},no_topic_samples;{self.no_topic_cnt},no_ans_samples;{self.no_ans_cnt}')

    def _process(
        self,
        raw_set,
        entity_identifiers,
        save_path
    ):
        """
        Process all samples in the dataset and optionally cache the result.

        If a cached version of the processed dataset exists, it is loaded
        directly from disk. Otherwise, every sample is processed individually
        using :meth:`_process_sample` and the resulting list is serialized with
        pickle.

        Parameters
        ----------
        raw_set : list
            Raw dataset.

        entity_identifiers : set
            Set of entities considered identifier-only (without meaningful
            textual descriptions).

        save_path : str
            Path where the processed dataset should be saved.

        Returns
        -------
        list
            List of processed sample dictionaries.
        """
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        
        processed_dict_list = []
        for i in tqdm(range(len(raw_set))):
            sample_i = raw_set[i]
            processed_dict_i = self._process_sample(
                sample_i, 
                entity_identifiers)
            # if processed_dict_i is not None:
            processed_dict_list.append(processed_dict_i)

        with open(save_path, 'wb') as f:
            pickle.dump(processed_dict_list, f)
        
        return processed_dict_list

    def _process_sample(
        self,
        sample,
        entity_identifiers
    ):
        """
        Convert a raw sample into its processed representation.

        The preprocessing performs the following operations:

        - extracts all entities and relations from the graph,
        - separates textual entities from identifier-only entities,
        - assigns deterministic integer identifiers,
        - converts graph triples into integer ID format,
        - maps question and answer entities to their corresponding IDs.

        Parameters
        ----------
        sample : dict
            Raw sample containing a question, graph, topic entities and answer
            entities.

        entity_identifiers : set
            Set of entities that should be considered identifier-only.

        Returns
        -------
        dict
            Dictionary containing the processed sample with the following keys:

            - ``id`` : sample identifier.
            - ``question`` : question text.
            - ``q_entity`` : original topic entities.
            - ``q_entity_id_list`` : topic entity IDs.
            - ``text_entity_list`` : entities with textual descriptions.
            - ``non_text_entity_list`` : identifier-only entities.
            - ``relation_list`` : sorted relation list.
            - ``h_id_list`` : head node IDs.
            - ``r_id_list`` : relation IDs.
            - ``t_id_list`` : tail node IDs.
            - ``a_entity`` : original answer entities.
            - ``a_entity_id_list`` : answer entity IDs.
            - ``id2entities`` : mapping from entity IDs to entity names.
            - ``id2relations`` : mapping from relation IDs to relation names.
        """
        # Model input (0) question
        question = sample['question']
        
        triples = sample['graph']

        all_entities = set()
        all_relations = set()
        for (h, r, t) in triples:
            all_entities.add(h)
            all_relations.add(r)
            all_entities.add(t)
        
        # Sort for deterministic entity IDs.
        entity_list = sorted(all_entities)
        # Parition the entities based on if the associated text is meaningful.
        # Model input (1) text of entities
        #             (2) number of entities without text
        """
        Considering the toy KG or the adapted KG, there is no use in registering the non textual entities

        Previous code : 
        text_entity_list = []
        non_text_entity_list = []
        for entity in entity_list:
            if entity in entity_identifiers:
                non_text_entity_list.append(entity)
            else:
                text_entity_list.append(entity)
        """
        text_entity_list = entity_list
        non_text_entity_list = []
        
        # Create entity IDs.
        entity2id = dict()
        entity_id = 0
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        id2entity = {v: k for k, v in entity2id.items()} # reverse the mapping
        

        # Model input (3) text of relations
        relation_list = sorted(all_relations)
        # Create relation IDs.
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1
        id2relation = {v: k for k, v in rel2id.items()} # reverse the mapping


        # Convert triples to entity and relation IDs.
        # Model input (4) triples in the ID space for
        # graph construction and embedding indexing
        h_id_list = []
        r_id_list = []
        t_id_list = []
        for (h, r, t) in triples:
            h_id_list.append(entity2id[h])
            r_id_list.append(rel2id[r])
            t_id_list.append(entity2id[t])

        # Model input (5) list of question entity IDs
        q_entity_id_list = []
        for entity in sample['q_entity']:
            if entity in entity2id:
                q_entity_id_list.append(entity2id[entity])

        # Prepare output labels.
        assert sample['a_entity'] == sample['answer']
        a_entity_id_list = []
        for entity in sample['a_entity']:
            entity_id = entity2id.get(entity, None)
            if entity_id is not None:
                a_entity_id_list.append(entity_id)

        processed_dict = {
            'id': sample['id'],
            'question': question,
            'q_entity': sample['q_entity'],
            'q_entity_id_list': q_entity_id_list,
            'text_entity_list': text_entity_list,
            'non_text_entity_list': non_text_entity_list,
            'relation_list': relation_list,
            'h_id_list': h_id_list,
            'r_id_list': r_id_list,
            't_id_list': t_id_list,
            'a_entity': sample['a_entity'],
            'a_entity_id_list': a_entity_id_list,
            'id2entities': id2entity,
            'id2relations': id2relation
        }

        return processed_dict

    def __len__(self):
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        """
        Retrieve the information required for embedding inference.

        Parameters
        ----------
        i : int
            Index of the sample.

        Returns
        -------
        tuple
            A tuple containing:

            - **id** (*str or int*) : sample identifier.
            - **q_text** (*str*) : question text.
            - **text_entity_list** (*list*) : entities with textual
              descriptions.
            - **relation_list** (*list*) : relations appearing in the graph.
        """
        sample = self.processed_dict_list[i]
        
        id = sample['id']
        q_text = sample['question']
        text_entity_list = sample['text_entity_list']
        relation_list = sample['relation_list']
        
        return id, q_text, text_entity_list, relation_list

