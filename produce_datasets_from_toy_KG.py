import networkx as nx
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, OWL
import random
import os
import pickle
import argparse


# 1. Creating a direct graph
G = nx.DiGraph()

#  Patients
G.add_node("Patient_1", color="lightblue", type="person")
G.add_node("Patient_2", color="lightblue", type="person")
G.add_node("Patient_3", color="lightblue", type="person")
G.add_node("Patient_4", color="lightblue", type="person")
G.add_node("Patient_5", color="lightblue", type="person")
G.add_node("Patient_6", color="lightblue", type="person")
G.add_node("Patient_7", color="lightblue", type="person")
G.add_node("Patient_8", color="lightblue", type="person")
G.add_node("Patient_9", color="lightblue", type="person")
G.add_node("Patient_10", color="lightblue", type="person")
G.add_node("Patient_11", color="lightblue", type="person")

# Symptoms
G.add_node("headache", color="lightgreen", type="symptom")
G.add_node("confusion", color="lightgreen", type="symptom")
G.add_node("fever", color="lightgreen", type="symptom")
G.add_node("skin_redness", color="lightgreen", type="symptom")
G.add_node("vomiting", color="lightgreen", type="symptom")
G.add_node("hypertension", color="lightgreen", type="symptom")
G.add_node("cough", color="lightgreen", type="symptom")
#Diseases
G.add_node("sepsis", color="red", type="disease")
G.add_node("pneumonia", color="red", type="disease")
G.add_node("indigestion", color="red", type="disease")
G.add_node("no_disease", color="red", type="disease")
G.add_node("infection", color="red", type="typeofdisease")

#drug
G.add_node("stomach_cleansing", color="yellow", type="treatment")
G.add_node("respiratory_assistance", color="yellow", type="treatment")
G.add_node("antibiotics", color="yellow", type="treatment")
G.add_node("cortisone", color="yellow", type="treatment")
G.add_node("nutritional_supplements", color="yellow", type="treatment")
# description
G.add_node("male", color="pink", type="description")
G.add_node("female", color="pink", type="description")
G.add_node("0-10_years_old", color="pink", type="description")
G.add_node("11-20_years_old", color="pink", type="description")
G.add_node("21-30_years_old", color="pink", type="description")
G.add_node("infection_history", color="pink", type="description")
G.add_node("normal_weight", color="pink", type="description")
G.add_node("underweight", color="pink", type="description")
G.add_node("overweight", color="pink", type="description")
G.add_node("Paris",color="pink", type="desciption")
G.add_node("London",color="pink", type="desciption")
G.add_node("New_York",color="pink", type="desciption")
G.add_node("Smoker",color="pink", type="desciption")

# 3. Adding relations
G.add_edge("Patient_1", "sepsis", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_2", "pneumonia", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_3", "indigestion", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_4", "sepsis", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_5", "pneumonia", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_6", "indigestion", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_7", "sepsis", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_8", "pneumonia", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_9", "indigestion", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_10", "indigestion", label="diagnosis", relation="has_diagnosis")
G.add_edge("Patient_11", "no_disease", label="diagnosis", relation="has_diagnosis")

G.add_edge("Patient_1", "male", label="sex", relation="is_a")
G.add_edge("Patient_2", "female", label="sex", relation="is_a")
G.add_edge("Patient_3", "male", label="sex", relation="is_a")
G.add_edge("Patient_4", "female", label="sex", relation="is_a")
G.add_edge("Patient_5", "male", label="sex", relation="is_a")
G.add_edge("Patient_6", "female", label="sex", relation="is_a")
G.add_edge("Patient_7", "male", label="sex", relation="is_a")
G.add_edge("Patient_8", "female", label="sex", relation="is_a")
G.add_edge("Patient_9", "male", label="sex", relation="is_a")
G.add_edge("Patient_10", "female", label="sex", relation="is_a")
G.add_edge("Patient_11", "male", label="sex", relation="is_a")


G.add_edge("Patient_1", "overweight", label="weight", relation="is")
G.add_edge("Patient_2", "underweight", label="weight", relation="is")
G.add_edge("Patient_3", "normal_weight", label="weight", relation="is")
G.add_edge("Patient_4", "overweight", label="weight", relation="is")
G.add_edge("Patient_5", "underweight", label="weight", relation="is")
G.add_edge("Patient_6", "normal_weight", label="weight", relation="is")
G.add_edge("Patient_7", "overweight", label="weight", relation="is")
G.add_edge("Patient_8", "underweight", label="weight", relation="is")
G.add_edge("Patient_9", "normal_weight", label="weight", relation="is")
G.add_edge("Patient_10", "overweight", label="weight", relation="is")
G.add_edge("Patient_11", "normal_weight", label="weight", relation="is")

G.add_edge("Patient_1", "0-10_years_old", label="age", relation="is")
G.add_edge("Patient_2", "11-20_years_old", label="age", relation="is")
G.add_edge("Patient_3", "21-30_years_old", label="age", relation="is")
G.add_edge("Patient_4", "0-10_years_old", label="age", relation="is")
G.add_edge("Patient_5", "11-20_years_old", label="age", relation="is")
G.add_edge("Patient_6", "21-30_years_old", label="age", relation="is")
G.add_edge("Patient_7", "0-10_years_old", label="age", relation="is")
G.add_edge("Patient_8", "11-20_years_old", label="age", relation="is")
G.add_edge("Patient_9", "21-30_years_old", label="age", relation="is")
G.add_edge("Patient_10", "0-10_years_old", label="age", relation="is")
G.add_edge("Patient_11", "11-20_years_old", label="age", relation="is")

G.add_edge("Patient_1", "Paris", label="origin", relation="is_from")
G.add_edge("Patient_2", "London", label="origin", relation="is_from")
G.add_edge("Patient_3", "New_York", label="origin", relation="is_from")
G.add_edge("Patient_4", "Paris", label="origin", relation="is_from")
G.add_edge("Patient_5", "London", label="origin", relation="is_from")
G.add_edge("Patient_6", "New_York", label="origin", relation="is_from")
G.add_edge("Patient_7", "Paris", label="origin", relation="is_from")
G.add_edge("Patient_8", "London", label="origin", relation="is_from")
G.add_edge("Patient_9", "New_York", label="origin", relation="is_from")
G.add_edge("Patient_10", "Paris", label="origin", relation="is_from")
G.add_edge("Patient_11", "London", label="origin", relation="is_from")

#Smoker
G.add_edge("Patient_1", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_2", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_3", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_4", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_5", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_6", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Patient_6", "Smoker", label="lifestyle", relation="is_a")
G.add_edge("Smoker", "cough", label="symptom", relation="can_produce")


#indigestion
G.add_edge("Patient_3", "vomiting", label="symptom", relation="present_symptom")
G.add_edge("Patient_6", "vomiting", label="symptom", relation="present_symptom")
G.add_edge("Patient_9", "vomiting", label="symptom", relation="present_symptom")
G.add_edge("Patient_10", "vomiting", label="symptom", relation="present_symptom")
G.add_edge("Patient_9", "fever", label="symptom", relation="present_symptom")
G.add_edge("Patient_10", "fever", label="symptom", relation="present_symptom")


#pneumonia
G.add_edge("Patient_2", "cough", label="symptom", relation="present_symptom")
G.add_edge("Patient_5", "cough", label="symptom", relation="present_symptom")
G.add_edge("Patient_8", "cough", label="symptom", relation="present_symptom")
G.add_edge("Patient_2", "hypertension", label="symptom", relation="present_symptom")
G.add_edge("Patient_5", "hypertension", label="symptom", relation="present_symptom")
G.add_edge("Patient_8", "hypertension", label="symptom", relation="present_symptom")

#sepsis
G.add_edge("Patient_1", "confusion", label="symptom", relation="present_symptom")
G.add_edge("Patient_4", "confusion", label="symptom", relation="present_symptom")
G.add_edge("Patient_1", "fever", label="symptom", relation="present_symptom")
G.add_edge("Patient_4", "fever", label="symptom", relation="present_symptom")
G.add_edge("Patient_7", "fever", label="symptom", relation="present_symptom")
G.add_edge("Patient_1", "headache", label="symptom", relation="present_symptom")
G.add_edge("Patient_4", "headache", label="symptom", relation="present_symptom")
G.add_edge("Patient_7", "headache", label="symptom", relation="present_symptom")
G.add_edge("Patient_1", "skin_redness", label="symptom", relation="present_symptom")
G.add_edge("Patient_4", "skin_redness", label="symptom", relation="present_symptom")
G.add_edge("Patient_7", "skin_redness", label="symptom", relation="present_symptom")
G.add_edge("Patient_1", "hypertension", label="symptom", relation="present_symptom")
G.add_edge("Patient_4", "hypertension", label="symptom", relation="present_symptom")
G.add_edge("Patient_7", "hypertension", label="symptom", relation="present_symptom")

#no disease
G.add_edge("Patient_11", "headache", label="symptom", relation="present_symptom")

#precisons for diseases
G.add_edge("infection", "antibiotics", label="treatment", relation="treated_by")


G.add_edge("sepsis", "infection", label="description", relation="is_an")

G.add_edge("sepsis", "respiratory_assistance", label="treatment", relation="treated_by")
G.add_edge("sepsis", "antibiotics", label="treatment", relation="treated_by")
G.add_edge("sepsis", "cortisone", label="treatment", relation="treated_by")
G.add_edge("sepsis", "nutritional_supplements", label="treatment", relation="treated_by")

G.add_edge("sepsis", "confusion", label="symptom", relation="has_symtom")
G.add_edge("sepsis", "fever", label="symptom", relation="has_symtom")
G.add_edge("sepsis", "headache", label="symptom", relation="has_symtom")
G.add_edge("sepsis", "skin_redness", label="symptom", relation="has_symtom")
G.add_edge("sepsis", "hypertension", label="symptom", relation="has_symtom")

G.add_edge("pneumonia", "infection", label="description", relation="is_an")

G.add_edge("pneumonia", "respiratory_assistance", label="treatment", relation="treated_by")
G.add_edge("pneumonia", "antibiotics", label="treatment", relation="treated_by")
G.add_edge("pneumonia", "cortisone", label="treatment", relation="treated_by")

G.add_edge("pneumonia", "hypertension", label="symptom", relation="has_symtom")
G.add_edge("pneumonia", "cough", label="symptom", relation="has_symtom")


G.add_edge("indigestion", "stomach_cleansing", label="treatment", relation="treated_by")

G.add_edge("indigestion", "vomiting", label="symptom", relation="has_symptom")
G.add_edge("indigestion", "fever", label="symptom", relation="has_symptom")


#precisons for symptom
G.add_edge("skin_redness", "cortisone", label="treatment", relation="tackled_by")
G.add_edge("hypertension", "cortisone", label="treatment", relation="tackled_by")
G.add_edge("vomiting", "nutritional_supplements", label="treatment", relation="tackled_by")

# medical history
G.add_edge("Patient_1", "infection_history", label="description", relation="has")
G.add_edge("Patient_2", "infection_history", label="description", relation="has")
G.add_edge("Patient_3", "infection_history", label="description", relation="has")
G.add_edge("Patient_4", "infection_history", label="description", relation="has")
G.add_edge("Patient_5", "infection_history", label="description", relation="has")
G.add_edge("Patient_6", "infection_history", label="description", relation="has_no")
G.add_edge("Patient_7", "infection_history", label="description", relation="has_no")
G.add_edge("Patient_8", "infection_history", label="description", relation="has_no")
G.add_edge("Patient_9", "infection_history", label="description", relation="has_no")
G.add_edge("Patient_10", "infection_history", label="description", relation="has_no")
G.add_edge("Patient_11", "infection_history", label="description", relation="has_no")

def networkx_to_owl(nx_graph, output_file="medical_kg.owl"):

    g = Graph()

    EX = Namespace("http://example.org/medical#")

    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # -------------------------
    # Creating classes
    # -------------------------

    classes = set()

    for node, attrs in nx_graph.nodes(data=True):

        node_type = attrs.get("type", "Entity")
        classes.add(node_type)

    for cls in classes:
        g.add((EX[cls], RDF.type, OWL.Class))

    # -------------------------
    # Creating instances
    # -------------------------

    for node, attrs in nx_graph.nodes(data=True):

        node_type = attrs.get("type", "Entity")

        g.add((EX[node], RDF.type, EX[node_type]))
        g.add((EX[node], RDF.type, OWL.NamedIndividual))

    # -------------------------
    # Creating properties
    # -------------------------

    properties = set()

    for source, target, attrs in nx_graph.edges(data=True):

        relation = attrs.get("relation")

        if relation:
            properties.add(relation)

    for prop in properties:

        g.add((EX[prop], RDF.type, OWL.ObjectProperty))

    # -------------------------
    # Creating triplets
    # -------------------------

    for source, target, attrs in nx_graph.edges(data=True):

        relation = attrs.get("relation")

        if relation:

            g.add(
                (
                    EX[source],
                    EX[relation],
                    EX[target]
                )
            )

    # -------------------------
    # Saving
    # -------------------------

    g.serialize(
        destination=output_file,
        format="xml"
    )

    print(f"Ontology saved : {output_file}")

    return g

def build_triples(G):
    """
    Extract all triples from a NetworkX graph.

    Parameters
    ----------
    G : nx.DiGraph
        Knowledge graph where edges contain a 'relation' attribute.

    Returns
    -------
    list of lists
        List of triples in the form [head, relation, tail].
    """
    triples = []
    for h, t, data in G.edges(data=True):
        r = data["relation"]
        triples.append([h, r, t])
    return triples

def neighbors_by_relation(G, node, relation=None):
    """
    Retrieve outgoing neighbors of a node, optionally filtered by relation type.

    Parameters
    ----------
    G : nx.DiGraph
        Knowledge graph.

    node : str
        Source node.

    relation : str or None, optional
        If provided, only edges with this relation type are returned.

    Returns
    -------
    list of tuple
        List of (head, relation, tail) outgoing edges.
    """
    out = []
    for h, t, data in G.out_edges(node, data=True):
        if relation is None or data["relation"] == relation:
            out.append((h, data["relation"], t))
    return out

def generate_dataset(G, n_samples=200):
    """
    Generate a synthetic dataset following the WebQSP format.

    Each sample contains:
    - id
    - question
    - answer
    - q_entity
    - a_entity
    - graph

    Format:
    {
        "id": int,
        "question": str,
        "answer": list[str],
        "q_entity": list[str],
        "a_entity": list[str],
        "graph": list[[head, relation, tail]]
    }
    """

    dataset = []

    nodes = list(G.nodes)

    patients = [
        n for n in nodes 
        if n.startswith("Patient")
    ]

    # Complete KG shared by all questions
    global_graph = [
        [h, r, t]
        for h, r, t in build_triples(G)
    ]


    for i in range(n_samples):

        patient = random.choice(patients)

        sample = {
            "id": i,
            "question": None,
            "answer": [],
            "q_entity": [patient],
            "a_entity": [],
            "graph": global_graph
        }


        q_type = random.choice(
            [
                "diagnosis",
                "symptom",
                "treatment"
            ]
        )


        # ------------------------------------------------
        # Question type 1:
        # What disease does Patient_X have?
        # ------------------------------------------------

        if q_type == "diagnosis":

            edges = neighbors_by_relation(
                G,
                patient,
                "has_diagnosis"
            )

            if not edges:
                continue

            _, _, disease = random.choice(edges)

            sample["question"] = (
                f"What disease does {patient} have?"
            )

            sample["answer"] = [
                disease
            ]

            sample["a_entity"] = [
                disease
            ]


        # ------------------------------------------------
        # Question type 2:
        # Which symptom is associated with disease?
        # ------------------------------------------------

        elif q_type == "symptom":

            disease_edges = neighbors_by_relation(
                G,
                patient,
                "has_diagnosis"
            )

            if not disease_edges:
                continue


            _, _, disease = random.choice(
                disease_edges
            )


            symptom_edges = neighbors_by_relation(
                G,
                disease,
                "has_symtom"
            )

            if not symptom_edges:
                continue


            _, _, symptom = random.choice(
                symptom_edges
            )


            sample["question"] = (
                f"What symptom is associated "
                f"with {disease}?"
            )


            sample["answer"] = [
                symptom
            ]

            sample["a_entity"] = [
                symptom
            ]


            # Important:
            # The entity in the question is now the disease
            sample["q_entity"] = [
                disease
            ]


        # ------------------------------------------------
        # Question type 3:
        # Which treatment is used?
        # ------------------------------------------------

        elif q_type == "treatment":

            disease_edges = neighbors_by_relation(
                G,
                patient,
                "has_diagnosis"
            )

            if not disease_edges:
                continue


            _, _, disease = random.choice(
                disease_edges
            )


            treatment_edges = neighbors_by_relation(
                G,
                disease,
                "treated_by"
            )

            if not treatment_edges:
                continue


            _, _, treatment = random.choice(
                treatment_edges
            )


            sample["question"] = (
                f"What treatment is used "
                f"for {disease}?"
            )


            sample["answer"] = [
                treatment
            ]

            sample["a_entity"] = [
                treatment
            ]


            sample["q_entity"] = [
                disease
            ]


        if sample["question"] is not None:
            dataset.append(sample)


    return dataset
"""
How to use the function: 
train_set = generate_dataset(G, n_samples=200)
"""

def save_dataset(dataset, save_dir, filename):
    """
    Save a dataset as a pickle file.

    Parameters
    ----------
    dataset : list of dict
        Dataset to serialize.

    save_dir : str
        Directory where the dataset will be stored.

    filename : str
        Name of the pickle file (e.g. "train.pkl").

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, filename)

    with open(file_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saving {len(dataset)} samples")
    print(os.path.getsize(file_path))

    print(f"Dataset saved to {file_path}")

def load_pkl_dataset(save_dir, filename):
    """
    Load a dataset from a pickle file.

    Parameters
    ----------
    save_dir : str
        Directory containing the dataset.

    filename : str
        Name of the pickle file.

    Returns
    -------
    list of dict
        Loaded dataset.
    """
    file_path = os.path.join(save_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with open(file_path, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} samples from {file_path}")

    return dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    random.shuffle(dataset)

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_set = dataset[:n_train]
    val_set = dataset[n_train:n_train + n_val]
    test_set = dataset[n_train + n_val:]

    return train_set, val_set, test_set

def main(args):
    # -------------------------
    # 1. Generate dataset
    # -------------------------
    print("Generating dataset...")
    dataset = generate_dataset(G, n_samples=args.n_samples)
    print(f"Total samples generated: {len(dataset)}")

    # -------------------------
    # 2. Split dataset
    # -------------------------
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # -------------------------
    # 3. Save datasets
    # -------------------------
    save_dataset(train_set, args.save_dir, "train.pkl")
    save_dataset(val_set, args.save_dir, "val.pkl")
    save_dataset(test_set, args.save_dir, "test.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="data_files/toy/raw")

    args = parser.parse_args()

    main(args)
