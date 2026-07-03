# vLLM GPU Manager

A lightweight, single-binary CLI and web console for running [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible servers as Docker containers on NVIDIA GPU Linux hosts.

This project is intentionally small. It is not a cluster scheduler, model platform, or desktop AI studio. It is an internal operations tool for a GPU server where you want a clear UI over Docker, `nvidia-smi`, Hugging Face cache management, and vLLM launch parameters.

## Features

- **NVIDIA GPU monitoring:** Reads GPU memory and utilization from `nvidia-smi`.
- **CLI operations:** Manage GPUs, models, cache, and settings directly over SSH without starting the dashboard.
- **vLLM container lifecycle:** Start, stop, and restart managed vLLM Docker containers.
- **GPU placement controls:** Select GPU count, pin exact GPU IDs, and tune `--gpu-memory-utilization`.
- **VRAM launch checks:** Blocks launches when model weights plus estimated KV cache will not fit inside the requested GPU memory utilization budget.
- **KV cache estimation:** Reads downloaded model config files to estimate KV cache bytes per token and required KV memory for `--max-model-len`.
- **Hugging Face cache management:** Configure the host Hugging Face home directory, list cached models, download models, and delete cached model directories.
- **Download progress:** Starts model downloads in background Docker containers and estimates progress from cached bytes on disk.
- **Model recommendations:** Ranks an embedded open-source model catalog against detected GPUs, use case, VRAM fit, and estimated speed.
- **HF token handling:** Stores an optional Hugging Face token in `settings.json` with file mode `0600`.
- **vLLM API key handling:** Stores one optional global vLLM API key and applies it to newly created serving containers.
- **Optional dashboard authentication:** Protect the web UI and JSON APIs with a simple password-backed session cookie.
- **Single binary deployment:** The React frontend is embedded into the Go binary using `go:embed`.

## Scope

The manager currently targets:

- Linux GPU hosts
- NVIDIA GPUs
- Docker
- NVIDIA Container Toolkit
- Configurable vLLM Docker image (defaults to `vllm/vllm-openai:latest`)
- Hugging Face Hub model cache layout
- internal/private network usage

It does not provide multi-user authorization, Kubernetes orchestration, or cluster scheduling. It **does** provide automatic local HTTPS termination (via self-signed certificates) and an optional dynamic OpenAI proxy gateway that routes `/v1/` requests to active model containers.

## How It Works

The binary runs as either a local CLI or an optional Go `net/http` dashboard server. CLI commands call the same local Docker/GPU/cache code directly, so the dashboard does not need to be running for command-line use.

When dashboard mode is enabled, the backend serves the embedded React app and exposes internal JSON APIs under `/api`.

Runtime operations are performed through host integrations:

- `nvidia-smi` for GPU status
- Docker Engine API for container listing, inspection, creation, start, stop, restart, removal, image inspection, and image pulls
- Detached Docker Engine API containers using the configured vLLM Docker image and entrypoint `hf` for model downloads
- A short-lived Docker Engine API `alpine` container mounted at `<hf-home>/hub:/target` for cached model deletion

Managed Docker containers are labelled so the app can distinguish serving containers from download containers:

- `vllm-manager.managed=true`
- `vllm-manager.kind=server|download`
- `vllm-manager.model=<huggingface-model-id>`

## Requirements

Runtime host:

- Linux
- NVIDIA GPU drivers
- `nvidia-smi` available in `PATH`
- Docker installed and running
- NVIDIA Container Toolkit configured so Docker supports `--runtime nvidia`
- Configured vLLM Docker image (defaults to `vllm/vllm-openai:latest`) available locally or pullable by Docker
- `alpine` image available locally for cached model deletion

Build machine:

- Go version compatible with `go.mod`
- Node.js and npm

## Build

```bash
cd frontend
npm install
cd ..

./build.sh
```

The build script runs the frontend build and compiles a standalone backend binary named `vllm-manager`.

## Release

Pushing a version tag that starts with `v` creates a GitHub Release and uploads Linux AMD64 and ARM64 binaries plus SHA256 checksums.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow builds the React frontend first, embeds it into the Go binary, runs frontend linting and backend tests, then publishes:

```text
vllm-manager-<tag>-linux-amd64
vllm-manager-<tag>-linux-amd64.sha256
vllm-manager-<tag>-linux-arm64
vllm-manager-<tag>-linux-arm64.sha256
```

## CLI

```bash
./vllm-manager
```

Running with no arguments prints CLI help. Add `--json` to commands for machine-readable output.

Examples:

```bash
./vllm-manager gpu list
./vllm-manager model active
./vllm-manager model cached
./vllm-manager model download Qwen/Qwen2.5-Coder-7B-Instruct
./vllm-manager model progress Qwen/Qwen2.5-Coder-7B-Instruct
./vllm-manager model recommend                     # Show recommended models for detected GPUs
./vllm-manager model start Qwen/Qwen2.5-Coder-7B-Instruct --port 8000 --gpus 1 --gpu-ids 0 --mem-util 0.85 --kv-cache-dtype auto
./vllm-manager model stop --port 8000
./vllm-manager cache delete Qwen/Qwen2.5-Coder-7B-Instruct --yes
./vllm-manager settings get --json
```

Destructive CLI commands prompt by default. Use `--yes` for non-interactive automation.

## Dashboard

```bash
./vllm-manager --dashboard --port 9090
```

Set `VLLM_MANAGER_PASSWORD` to require browser login before the dashboard can call `/api` endpoints:

```bash
VLLM_MANAGER_PASSWORD='change-me' ./vllm-manager --dashboard --port 9090
```

Enable the built-in `/v1/` OpenAI-compatible proxy gateway with `--proxy`:

```bash
./vllm-manager --dashboard --port 9090 --proxy
```

When the proxy is not enabled, no `/v1/` gateway is registered and the Traffic Routing page is hidden from the dashboard.

Then open:

```text
http://<host>:9090
```

You can also use the helper script:

```bash
./manager.sh start
./manager.sh start 8080
./manager.sh start 8080 change-me --proxy
./manager.sh status
./manager.sh stop
./manager.sh restart
```

Logs are written to `vllm-manager.log` when started through `manager.sh`.

Pass `--proxy` to `manager.sh start` or set `VLLM_MANAGER_PROXY=1` to enable the built-in `/v1/` proxy gateway from the helper script.
The helper writes `.vllm-manager.state` with the last port, proxy mode, and dashboard password so `./manager.sh restart` reuses them. The state file is written with `0600` permissions but stores the password as plaintext.

Dashboard mode handles `SIGINT` and `SIGTERM` with a graceful HTTP shutdown timeout, so in-flight API requests get a short window to finish during restarts.

## Settings

Settings are stored in `settings.json` beside the binary.

```json
{
  "hf_cache": "/mnt/data/models",
  "hf_token": "hf_...",
  "vllm_api_key": "token-...",
  "inference_api_key": "optional-key-for-proxy",
  "enable_cuda_compatibility": false,
  "hf_hub_offline": false,
  "vllm_image": "vllm/vllm-openai:latest"
}
```

`inference_api_key` is an optional key required to access the `/v1/` proxy gateway. When set, all proxied OpenAI-compatible requests must include `Authorization: Bearer <inference_api_key>`.

`hf_cache` is the Hugging Face home directory, not the `hub` directory itself. If a user enters a path ending in `/hub`, the backend normalizes it to the parent directory.

`vllm_api_key` is optional. When set, the manager passes it to newly created serving containers as `--api-key <value>`, so OpenAI-compatible clients must send `Authorization: Bearer <value>`. Existing containers keep the command they were created with; recreate a container to apply a changed key.

`enable_cuda_compatibility` sets `VLLM_ENABLE_CUDA_COMPATIBILITY=1` on newly created serving containers. `hf_hub_offline` sets `HF_HUB_OFFLINE=1` on newly created serving containers and should only be enabled after required models are cached locally. `vllm_image` is optional and specifies the Docker image name and tag used to run serving containers and model downloads (defaults to `vllm/vllm-openai:latest`). When set online, it verifies the tag exists on Docker Hub. Existing containers keep their original environment/image; recreate a container to apply changed settings.

Example:

```text
/mnt/data/models/hub -> /mnt/data/models
```

The app mounts the configured Hugging Face home into vLLM containers as:

```text
<hf_cache>:/hf_cache
HF_HOME=/hf_cache
```

The Hugging Face cache path can only be changed when no managed vLLM containers exist. This avoids leaving existing containers attached to an old model cache while the UI points to a new one.

## Model Downloads

Downloads run in detached Docker containers using the vLLM image. The backend creates them through the Docker Engine API with settings equivalent to:

```bash
docker run -d --rm \
  --network host \
  -v <hf_cache>:/hf_cache \
  -e HF_HOME=/hf_cache \
  --entrypoint hf \
  <vllm_image> \
  download <model>
```

The UI shows progress by polling cached bytes on disk through `/api/models/download/progress`. The progress is an estimate based on expected Hugging Face model size and the current cache directory size.

Only one download operation per model/cache path is allowed at a time.

## Recommended Models

The Recommended Models page and `model recommend` CLI command rank an embedded catalog from `api/data/recommend_models.json` against the GPUs detected on the host. The catalog is intentionally stored as JSON in the repository so open-source contributors can review and extend it without adding runtime services or package dependencies.

Catalog fields include:

- `parameter_b`: total parameter count in billions, used for weight memory estimates.
- `active_parameter_b`: optional active parameter count for MoE models, used for KV/cache and speed estimates.
- `quantization`: weight precision hint, such as `BF16`, `FP8`, or `AWQ-4BIT`.
- `context_length`: advertised context window used for recommendation sizing, capped internally for launch safety.
- `use_cases`: one or more of `general`, `coding`, `reasoning`, `chat`, or `vision`.
- `quality`: curated 0-100 model quality hint. This is not a live benchmark; the final recommendation `score` also includes VRAM fit, estimated speed, and use-case match.

## Starting A Model

Serving containers are created through the Docker Engine API with host networking and the NVIDIA runtime, equivalent to:

```bash
docker run -d \
  --ipc=host \
  --runtime nvidia \
  --gpus <selection> \
  --network host \
  -v <hf_cache>:/hf_cache \
  -e HF_HOME=/hf_cache \
  <vllm_image> \
  --model <model> \
  --tensor-parallel-size <gpus> \
  --port <port> \
  --gpu-memory-utilization <mem_util>
```

Optional launch parameters currently exposed by the UI/API:

- `--max-model-len`
- `--max-num-seqs`
- `--kv-cache-dtype`
- `--tool-call-parser`
- `--enable-auto-tool-choice`
- `--api-key` when a global vLLM API key is configured

Managed container restart works for both running and stopped containers. Running containers are restarted through Docker; stopped containers are started with their original image, command, environment, GPU devices, and volume mounts.

Before launching, the backend validates:

- requested port is available
- requested GPU count and GPU IDs are valid
- no active download container is currently downloading the same model
- combined `gpu_memory_utilization` on any selected GPU does not exceed 100%
- downloaded model size can fit in the requested GPU budget
- estimated KV cache can fit for the requested or default max context length
- KV heads are compatible with the requested tensor parallel size when config data is available

## Deleting Cached Models

Cached model deletion removes the Hugging Face hub model directory through an Alpine container mounted at the configured cache hub path.

Deletion is blocked when:

- the model is currently being downloaded
- a managed serving container references that model
- the resolved model path would escape the configured Hugging Face hub directory

## API

All APIs are under `/api`.

| Method | Path                                   | Description                                                               |
|--------|----------------------------------------|---------------------------------------------------------------------------|
| `GET`  | `/auth/status`                         | Return whether dashboard authentication is required and satisfied.        |
| `POST` | `/auth/login`                          | Create a browser session when `VLLM_MANAGER_PASSWORD` is configured.      |
| `POST` | `/auth/logout`                         | Clear the browser session cookie.                                         |
| `GET`  | `/settings`                            | Read HF cache path, secret status, cache disk usage, and system RAM usage. |
| `POST` | `/settings`                            | Save HF cache path and optionally update or clear stored secrets.          |
| `GET`  | `/routing`                             | Read the traffic routing table mapping models to local container ports.   |
| `POST` | `/routing`                             | Save the traffic routing table.                                           |
| `GET`  | `/gpus`                                | Return parsed NVIDIA GPU status from `nvidia-smi`.                        |
| `GET`  | `/models/active`                       | List managed vLLM serving containers.                                     |
| `GET`  | `/models/downloaded`                   | List cached Hugging Face models and local size/KV metadata.               |
| `GET`  | `/models/info?model=<id>`              | Fetch model metadata from the Hugging Face API.                           |
| `POST` | `/models/download`                     | Start a background model download container.                              |
| `GET`  | `/models/download/progress?model=<id>` | Return current cached byte count for a model.                             |
| `POST` | `/models/start`                        | Start a vLLM serving container.                                           |
| `POST` | `/models/stop`                         | Stop a serving container by port.                                         |
| `POST` | `/models/restart`                      | Restart a running managed container or start a stopped one by container ID. |
| `POST` | `/models/remove`                       | Remove a managed stopped container by container ID.                       |
| `POST` | `/models/downloaded/delete`            | Delete a cached model directory.                                          |
| `GET`  | `/recommendations`                     | Return ranked models based on detected GPU capabilities.                  |

### Proxy Gateway

When dashboard mode is started with `--proxy`, the manager provides a built-in reverse proxy gateway on `/v1/`. You can point your OpenAI-compatible clients directly to the manager. It will inspect the incoming JSON payload (e.g. `{"model": "..."}`), look up the configured routing table, and forward the request to the correct underlying vLLM container port. Optional API Key authorization is applied if an inference API key is configured.

The frontend currently hides container removal from the active model UI, but the backend endpoint remains available for operational cleanup.

## Security

This is an internal utility. It provides:

- Optional single-password dashboard authentication for the web UI and JSON APIs
- Self-signed HTTPS termination on the dashboard port (certificate auto-generated on first run)
- Request-level `Authorization: Bearer <inference_api_key>` for `/v1/` proxy endpoints

Anyone who can access the web interface can inspect GPU state, start and stop model containers, trigger model downloads, delete cached models, and use configured Hugging Face or vLLM API keys indirectly.
