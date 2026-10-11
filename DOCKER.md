# Docker: KLDA RFF/ORF benchmarks

The Docker setup runs the **working `experiments/orf` entry point** only. The legacy RFSF path (`KLDAver2.py`) is not selected, and is unchanged.

## Requirements
- Docker Engine / Docker Desktop with Compose v2
- CPU: Docker on Windows, Linux or macOS (amd64 recommended)
- GPU: Linux + NVIDIA GPU + NVIDIA driver + NVIDIA Container Toolkit, or an NVIDIA-ready cloud image. CUDA 12.1 compatible driver required for the supplied GPU build.
- Network on first run for PyTorch wheels and Hugging Face model weights.

## Run on CPU

From the repository root:

```sh
docker compose build cpu
docker compose run --rm cpu CLINC --mapping orf --D 128 --num_ensembles 1 --sigma 1e-2 --ridge 1e-3 --seed 0
```

Compare RFF:

```sh
docker compose run --rm cpu CLINC --mapping rff --D 128 --num_ensembles 1 --sigma 1e-2 --ridge 1e-3 --seed 0
```

Or run the default CLINC ORF trial:
```sh
docker compose up --build --abort-on-container-exit cpu
```

## Run on an NVIDIA GPU cloud instance

```sh
nvidia-smi
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm gpu CLINC --mapping orf --D 512 --num_ensembles 5 --sigma 1e-2 --ridge 1e-3 --seed 0
```

GPU service uses CUDA 12.1 PyTorch wheels; drivers must be compatible. `gpus: all` requires working NVIDIA container runtime integration. The CPU and GPU services use separate image tags, so they will not overwrite each other.

## Benchmark all four datasets with three seeds

```sh
for ds in CLINC BANKING DBPEDIA HWU; do
  for seed in 0 1 2; do
    for map in rff orf; do
      docker compose --profile gpu run --rm gpu "$ds" --mapping "$map" --D 512 --num_ensembles 5 --sigma 1e-2 --ridge 1e-3 --seed "$seed"
    done
  done
done
```

On a CPU machine, replace `--profile gpu run --rm gpu` with `run --rm cpu`.

### Persisted files
- The repository's `./results` directory is mounted read-write; benchmark outputs survive container removal.
- Named Docker volume `hf-cache` holds downloaded models so repeated container runs avoid model re-downloads.
- Dataset CSVs in `./data` are copied into the image at build time. Rebuild after changing CSV files.
- The repository's original results remain untouched unless the same output filenames are used: experiment writes to `results/<DATASET>/<model>-<mapping>.txt`.

### Notes
- This packaging does **not** cache generated embeddings between experiment runs; that optimization needs a separate code change.
- The entry point accepts arguments defined by `experiments/orf/evaluate.py`.
- GPU cloud pricing is charged for the provisioned instance even if the container is idle. Stop/destroy the cloud instance when finished.
- This container configuration has not yet been verified by running `docker build` on an actual Docker daemon.
