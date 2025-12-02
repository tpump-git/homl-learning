Running the project in a GPU-enabled TensorFlow Jupyter container
===============================================================

This file explains how to start a GPU-ready Jupyter container that mounts this repository and exposes Jupyter on port 8888.

Prerequisites (host)
- Docker installed and working on the host.
- NVIDIA drivers installed on the host (you can verify with `nvidia-smi`).
- `nvidia-container-toolkit` (or the `nvidia` runtime) configured for Docker. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Start the container (run from the project root on the host)

```bash
# from the project root (the directory that contains this README and docker-compose.yml)
# Start the container in the foreground (use -d to run detached):
docker compose up --build

# or (legacy) docker-compose
# docker-compose up --build
```

This will pull `tensorflow/tensorflow:2.17.0-gpu-jupyter` if not present, start a container named `homl_tf_gpu`, mount the repository into `/tf/work`, and run Jupyter Lab.

Open Jupyter
- The container logs will show a URL with a token. Open that URL in your browser (typically `http://127.0.0.1:8888/?token=...`).

Quick verification commands (run on the host)

```bash
# show running containers
docker ps

# show GPU on the host
nvidia-smi

# check GPU visibility inside the container (non-interactive)
docker exec -i homl_tf_gpu python - <<'PY'
import tensorflow as tf
print('TF version:', tf.__version__)
print('Physical devices:', tf.config.list_physical_devices())
print('GPU count:', len(tf.config.list_physical_devices('GPU')))
PY
```

Interactive shell inside container (for debugging)

```bash
docker exec -it homl_tf_gpu bash
# once inside, you can run:
# nvidia-smi
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

Warm-up and first-run JIT note
- For some GPUs (e.g., compute capability 12.0), the official TF wheel may not include precompiled CUDA kernels. TensorFlow will JIT-compile kernels from PTX the first time they run. This adds one-time overhead (possibly several minutes for large kernels). After the first run the compiled kernels are cached and subsequent runs are fast.
- If you want, run a short warm-up matrix multiply (see the notebook cell `GPU verification and warm-up`) to trigger compilation before a long training run.

Stop and remove container

```bash
# stop and remove containers started with compose
docker compose down

# or stop the single container
docker stop homl_tf_gpu
```

If compose fails to start with GPU access
- Make sure Docker has `nvidia-container-toolkit` installed and Docker was restarted on the host.
- Verify `docker info` shows `Runtimes: nvidia` or that the `--gpus` flag is available. If not, install the NVIDIA container toolkit.

If you want I can also:
- Add a `devcontainer.json` for VS Code Remote - Containers that uses this compose file.
- Insert the GPU test cell directly into `12_custom_models_and_training_with_tensorflow.ipynb` for you (I can patch the notebook file). 
