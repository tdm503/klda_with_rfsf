#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for dataset in CLINC BANKING DBPEDIA HWU; do
  for seed in 0 1 2; do
    for mapping in rff orf; do
      python experiments/orf/evaluate.py "$dataset" --mapping "$mapping" --D 512 --sigma 1e-2 --ridge 1e-3 --num_ensembles 5 --seed "$seed" --model_name sentence-transformers/all-MiniLM-L6-v2
    done
  done
done
