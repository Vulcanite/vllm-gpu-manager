package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/auth/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"required": false, "authenticated": true})
	})

	mux.HandleFunc("/api/gpus", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		gpus := []map[string]any{
			{
				"id":                 0,
				"uuid":               "GPU-8a12f94c-34e8-4a11-872e-d018c1e40001",
				"name":               "NVIDIA H100 80GB HBM3",
				"memory_total":       85899345920,
				"memory_used":        36700160000,
				"memory_free":        49199185920,
				"memory_percent":     42.7,
				"gpu_utilization":    68,
				"compute_capability": "9.0",
				"processes": []map[string]any{
					{
						"pid":                24819,
						"user":               "vllm-manager",
						"name":               "python3 (vLLM::Engine)",
						"used_memory":        36700160000,
						"used_memory_human":  "34.18 GiB",
					},
				},
			},
			{
				"id":                 1,
				"uuid":               "GPU-8a12f94c-34e8-4a11-872e-d018c1e40002",
				"name":               "NVIDIA H100 80GB HBM3",
				"memory_total":       85899345920,
				"memory_used":        36700160000,
				"memory_free":        49199185920,
				"memory_percent":     42.7,
				"gpu_utilization":    68,
				"compute_capability": "9.0",
				"processes": []map[string]any{
					{
						"pid":                24820,
						"user":               "vllm-manager",
						"name":               "python3 (vLLM::Worker)",
						"used_memory":        36700160000,
						"used_memory_human":  "34.18 GiB",
					},
				},
			},
		}
		json.NewEncoder(w).Encode(gpus)
	})

	mux.HandleFunc("/api/models/active", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		active := []map[string]any{
			{
				"container_id": "c1f90a2b8e34",
				"name":         "vllm-Qwen2.5-Coder-32B-Instruct-8080",
				"model":        "Qwen/Qwen2.5-Coder-32B-Instruct",
				"served_model_name": "qwen-coder",
				"port":         8080,
				"gpus":         2,
				"gpu_ids":      "0,1",
				"state":        "running",
				"status":       "Up 4 hours (healthy)",
				"health":       "ready",
				"mem_util":     "0.85",
				"max_len":      "32768",
				"dtype":        "bfloat16",
				"kv_cache_dtype": "fp8",
				"parallelism_mode": "tensor",
				"created_at":   1723940000,
			},
		}
		json.NewEncoder(w).Encode(active)
	})

	mux.HandleFunc("/api/models/downloaded", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		models := []map[string]any{
			{
				"model":      "Qwen/Qwen2.5-Coder-32B-Instruct",
				"size_bytes": 67843000000,
				"size_human": "63.18 GiB",
				"active":     true,
			},
			{
				"model":      "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
				"size_bytes": 29843000000,
				"size_human": "27.79 GiB",
				"active":     false,
			},
			{
				"model":      "meta-llama/Llama-3.3-70B-Instruct",
				"size_bytes": 142843000000,
				"size_human": "133.03 GiB",
				"active":     false,
			},
		}
		json.NewEncoder(w).Encode(models)
	})

	mux.HandleFunc("/api/settings", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		settings := map[string]any{
			"hf_cache":                  "/var/lib/vllm-manager/.cache/huggingface",
			"hf_token_set":              true,
			"vllm_api_key_set":          true,
			"vllm_image":                "vllm/vllm-openai:v0.23.0",
			"server_bind_ip":            "0.0.0.0",
			"enable_cuda_compatibility": false,
			"hf_hub_offline":            false,
			"cache_disk_total_bytes":    2147483648000,
			"cache_disk_free_bytes":     1647483648000,
			"host_ram_total_bytes":      268435456000,
			"host_ram_available_bytes":  198435456000,
			"routing_table": map[string]int{
				"Qwen/Qwen2.5-Coder-32B-Instruct": 8080,
				"qwen-coder":                      8080,
			},
			"proxy_users": []map[string]any{
				{"username": "cursor-ide", "api_key": "vllm_proxy_a91b2c3d4e5f...", "request_count": 1420},
				{"username": "developer-alice", "api_key": "vllm_proxy_7c8d9e0f1a2b...", "request_count": 384},
				{"username": "analytics-batch", "api_key": "vllm_proxy_3e4f5a6b7c8d...", "request_count": 89},
			},
		}
		json.NewEncoder(w).Encode(settings)
	})

	mux.HandleFunc("/api/proxy-users", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		users := []map[string]any{
			{"username": "cursor-ide", "api_key": "vllm_proxy_a91b2c3d4e5f...", "request_count": 1420},
			{"username": "developer-alice", "api_key": "vllm_proxy_7c8d9e0f1a2b...", "request_count": 384},
			{"username": "analytics-batch", "api_key": "vllm_proxy_3e4f5a6b7c8d...", "request_count": 89},
		}
		json.NewEncoder(w).Encode(users)
	})

	mux.HandleFunc("/api/routing", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		routing := map[string]int{
			"Qwen/Qwen2.5-Coder-32B-Instruct": 8080,
			"qwen-coder":                      8080,
		}
		json.NewEncoder(w).Encode(routing)
	})

	// Serve frontend static files
	distDir := filepath.Join("frontend", "dist")
	fileServer := http.FileServer(http.Dir(distDir))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(distDir, r.URL.Path)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			http.ServeFile(w, r, filepath.Join(distDir, "index.html"))
			return
		}
		fileServer.ServeHTTP(w, r)
	})

	fmt.Println("Mock dashboard listening on http://127.0.0.1:9095")
	http.ListenAndServe("127.0.0.1:9095", mux)
}
