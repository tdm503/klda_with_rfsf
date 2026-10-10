import os
import argparse
import pandas as pd
from classifier import Classifier

DATASETS = ("CLINC", "BANKING", "DBPEDIA", "HWU")

def main(args):
    dataset = args.dataset_name.upper()
    train = pd.read_csv(f"data/{dataset.lower()}_train.csv")
    test = pd.read_csv(f"data/{dataset.lower()}_test.csv")
    model = Classifier(D=args.D, sigma=args.sigma, num_ensembles=args.num_ensembles,
                       seed=args.seed, model_name=args.model_name,
                       mapping=args.mapping, ridge=args.ridge)
    model.fit(train)
    accuracy = sum(model.predict(str(row.text)) == row.label for row in test.itertuples(index=False)) / len(test)
    print(f"{dataset} mapping={args.mapping} D={args.D} seed={args.seed} accuracy={accuracy:.4f}")
    directory = f"results/{dataset}"
    os.makedirs(directory, exist_ok=True)
    name = args.model_name.replace("/", "-")
    with open(f"{directory}/{name}-{args.mapping}.txt", "a", encoding="utf-8") as stream:
        stream.write(f"Seed: {args.seed}, Accuracy: {accuracy:.4f}, D: {args.D}, sigma: {args.sigma}, ridge: {args.ridge}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_name", type=str.upper, choices=DATASETS)
    p.add_argument("--mapping", choices=["rff", "orf"], default="rff")
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--sigma", type=float, default=1e-2)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--num_ensembles", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    main(p.parse_args())
