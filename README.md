# vLLM Management Console

A lightweight, standalone web console designed for managing local [vLLM](https://github.com/vllm-project/vllm) OpenAI-compatible inference servers on GPU-accelerated Linux hosts.

This project wraps Docker, `nvidia-smi`, and Hugging Face model caching into a simple and intuitive UI. It allows you to quickly download models, allocate GPU resources, and spin up OpenAI-compatible API endpoints without having to memorize complex Docker run commands.

## Key Features

*   **GPU Monitoring:** Real-time visibility into NVIDIA GPU memory usage and utilization directly from `nvidia-smi`.
*   **Model Management:** Download models directly from the Hugging Face Hub. View cached models and manage your local disk space.
*   **vLLM Container Orchestration:**
    *   Start new vLLM servers as Docker containers.
    *   Configure exact port bindings, GPU count allocations, and specific GPU ID pinning.
    *   Tune advanced vLLM parameters like memory utilization (`--gpu-memory-utilization`), max context length (`--max-model-len`), and tool call parsers (`--tool-call-parser`).
*   **Lifecycle Management:** List running and stopped vLLM instances, and effortlessly stop, remove, or restart managed containers.
*   **Single Binary Deployment:** The React (Vite) frontend is compiled and statically embedded into the Go backend binary using `go:embed`. Just drop the single executable on your server and run it.

## Architecture

*   **Backend:** Go (Golang) standard library HTTP server (`net/http`). Interacts with the host OS via `os/exec` to run `docker` and `nvidia-smi` commands.
*   **Frontend:** React 19, built with Vite and styled with CSS.
*   **Storage:** Minimal local state is stored in a `settings.json` file in the same directory as the executable, which holds the Hugging Face cache directory path and API token.

## Requirements

To run the binary and manage vLLM instances, the host server must have:

*   Linux operating system.
*   NVIDIA GPU(s) with drivers installed.
*   `nvidia-smi` available in the system PATH.
*   [Docker](https://docs.docker.com/get-docker/) installed and running.
*   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (NVIDIA runtime support for Docker) properly configured so that `--runtime nvidia` works.

To build the project from source, you will also need:

*   [Node.js](https://nodejs.org/) and `npm` (for the frontend).
*   [Go](https://go.dev/) 1.21+ (for the backend).

## Building from Source

A helper script is provided to build the frontend, package it, and compile the Go backend into a single executable.

```bash
# Clone the repository
git clone <repository-url>
cd gpu-vllm-management

# Install frontend dependencies first!
cd frontend
npm install
cd ..

# Run the build script
./build.sh
```

This script will:
1. Navigate into the `frontend` directory.
2. Run `npm run build` to generate the static SPA files in `frontend/dist`.
3. Compile the Go backend, embedding the `frontend/dist` files into a single `vllm-manager` executable.

## Running the Server

Once built, you can run the binary directly:

```bash
./vllm-manager -port 8000
```

### Using the Manager Script

A handy `manager.sh` script is included to help run the server as a background process (using `nohup`).

```bash
# Start the server in the background (defaults to port 8000)
./manager.sh start

# Start the server on a specific port
./manager.sh start 8080

# Check if the server is running
./manager.sh status

# Stop the background server
./manager.sh stop

# Restart the server
./manager.sh restart
```

Logs from the background process are written to `vllm-manager.log` in the current working directory.

## API Endpoints

The Go backend exposes several internal API endpoints under the `/api` prefix, handling cross-origin resource sharing (CORS) natively:

*   `GET /api/settings` / `POST /api/settings`: Manage Hugging Face cache directory and token.
*   `GET /api/gpus`: Retrieve parsed output from `nvidia-smi` about installed GPUs.
*   `GET /api/models/active`: List currently running and stopped vLLM Docker containers.
*   `GET /api/models/downloaded`: List models cached locally in the Hugging Face hub directory.
*   `GET /api/models/info`: Fetch metadata (size, likes, downloads) directly from the Hugging Face API.
*   `POST /api/models/download`: Trigger a containerized download of a Hugging Face model.
*   `POST /api/models/start`: Launch a new vLLM Docker container with specified parameters.
*   `POST /api/models/stop`: Gracefully stop a running vLLM container.
*   `POST /api/models/restart`: Restart an existing vLLM container.
*   `POST /api/models/remove`: Delete a stopped vLLM container.

## Security Note

This tool is designed as an **internal utility** for secure environments. **It does not include user authentication, authorization, or HTTPS termination.** Anyone with access to the web interface can deploy models, utilize GPU resources, and view your Hugging Face token (if configured).

Do not expose this interface directly to the public internet. Ensure it is only accessible via localhost, a secure VPN/VPC, or behind a reverse proxy that implements proper authentication (like Nginx with Basic Auth, or OAuth2 Proxy).
