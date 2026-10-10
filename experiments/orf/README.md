# RFF versus ORF in KLDA

This is an **isolated experiment**; the original KLDA and RFSF files are left unchanged.

Run from repository root:

```bash
pip install torch -r requirements.txt
python experiments/orf/evaluate.py CLINC --mapping rff --D 512 --sigma 1e-2 --ridge 1e-3 --num_ensembles 5 --seed 0
python experiments/orf/evaluate.py CLINC --mapping orf --D 512 --sigma 1e-2 --ridge 1e-3 --num_ensembles 5 --seed 0
bash experiments/orf/run_orf.sh
```

Both use identical Gaussian RBF spectral variance (2 * sigma), backbone, seeds, and data splits. ORF samples independent QR-orthogonal blocks and chi-distributed column radii; random projections remain fixed during training and inference.

Fixes relative to the original baseline: correct ensemble softmax dimension, exact incremental within-class scatter merge, and a regularized linear solve instead of a pseudoinverse. These fixes apply to both mappings in this experiment. Results are written to separate `*-rff.txt` and `*-orf.txt` files.

**Validation:** Synthetic embedding smoke tests only; benchmark accuracies on CLINC/BANKING/DBPEDIA/HWU have NOT been rerun. D-dimensional dense scatter requires O(D^2) memory and dense solve O(D^3) time.
