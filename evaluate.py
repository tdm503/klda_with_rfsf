import os
import pandas as pd
import warnings
import torch
import argparse
from classifier import Classifier

warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset paths
DATASET_PATHS = {
    "CLINC": {'train': 'data/clinc_train.csv', 'test': 'data/clinc_test.csv'},
    "BANKING": {'train': 'data/banking_train.csv', 'test': 'data/banking_test.csv'},
    "DBPEDIA": {'train': 'data/dbpedia_train.csv', 'test': 'data/dbpedia_test.csv'},
    "HWU": {'train': 'data/hwu_train.csv', 'test': 'data/hwu_test.csv'},
}

def load_data_from_csv(train_filename, test_filename):
    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)
    return train_df, test_df

def compute_accuracy(model, test_df):
    correct_predictions = 0
    total_predictions = test_df.shape[0]

    for _, row in test_df.iterrows():
        actual_label = row['label']
        text = str(row['text'])
        predicted_label = model.predict(text)

        if actual_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def save_results(dataset_name, model_name, seed, accuracy):
    results_dir = os.path.join("results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    file_name = model_name.replace("/", "-")
    results_file = os.path.join(results_dir, f"{file_name}.txt")

    with open(results_file, "a") as file:
        file.write(f"Seed: {seed}, Accuracy: {accuracy:.4f}\n")

def main(args):
    dataset_name = args.dataset_name.upper()
    
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available datasets are: {', '.join(DATASET_PATHS.keys())}")

    train_filename = DATASET_PATHS[dataset_name]['train']
    test_filename = DATASET_PATHS[dataset_name]['test']
    train_df, test_df = load_data_from_csv(train_filename, test_filename)

    model = Classifier(D=args.D, sigma=args.sigma, num_ensembles=args.num_ensembles, seed=args.seed, model_name=args.model_name)
    model.fit(train_df)

    accuracy = compute_accuracy(model, test_df)
    save_results(dataset_name, args.model_name, args.seed, accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help="Name of the dataset to use (CLINC, Banking, DBpedia, HWU).")
    parser.add_argument('--D', type=int, default=5000, help="Dimension of the transformed features using RFF.")
    parser.add_argument('--sigma', type=float, default=1e-4, help="Bandwidth parameter for the RBF kernel.")
    parser.add_argument('--num_ensembles', type=int, default=5, help="Number of ensemble models to use.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument('--model_name', type=str, default='facebook/bart-base', help="Name of the pre-trained model to use as the backbone.")

    args = parser.parse_args()
    main(args)
