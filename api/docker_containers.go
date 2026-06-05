package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

const modelHealthCheckTimeout = 5 * time.Second

func validateGPUSelection(gpus int, gpuIds string) (string, []GPUInfo, error) {
	gpuList, err := GetGPUInfo()
	if err == nil && len(gpuList) > 0 && gpus > len(gpuList) {
		return "", nil, validationError("requested %d GPUs, but only %d are available on this server", gpus, len(gpuList))
	}

	gpuIds = strings.TrimSpace(gpuIds)
	if gpuIds == "" || gpuIds == "all" {
		if err != nil || len(gpuList) == 0 {
			// No nvidia-smi data — fall back to "all" and skip VRAM preflight.
			return "all", nil, nil
		}

		sort.Slice(gpuList, func(i, j int) bool {
			return gpuList[i].ID < gpuList[j].ID
		})
		selected := gpuList[:gpus]

		// Fix GPT-P2a: never pass --gpus all to Docker — it gives the container
		// visibility into every GPU on the host, not just the TP-assigned ones.
		// Convert to an explicit device list so the container is scoped correctly
		// and our VRAM/overlap preflight matches what the container actually sees.
		ids := make([]string, len(selected))
		for i, g := range selected {
			ids[i] = strconv.Itoa(g.ID)
		}
		return fmt.Sprintf("device=%s", strings.Join(ids, ",")), selected, nil
	}

	if err != nil {
		return "", nil, fmt.Errorf("cannot validate gpu_ids because nvidia-smi failed: %v", err)
	}

	available := make(map[int]GPUInfo, len(gpuList))
	for _, gpu := range gpuList {
		available[gpu.ID] = gpu
	}

	parts := strings.Split(gpuIds, ",")
	seen := make(map[int]bool, len(parts))
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return "", nil, validationError("gpu_ids must be 'all' or comma-separated GPU IDs")
		}

		id, err := strconv.Atoi(part)
		if err != nil || id < 0 {
			return "", nil, validationError("invalid GPU ID: %s", part)
		}
		if _, ok := available[id]; !ok {
			return "", nil, validationError("GPU ID %d is not available on this server", id)
		}
		if seen[id] {
			return "", nil, validationError("duplicate GPU ID: %d", id)
		}
		seen[id] = true
		ids = append(ids, id)
	}

	if len(ids) != gpus {
		return "", nil, validationError("gpu_ids contains %d GPUs, but tensor parallel size is %d", len(ids), gpus)
	}

	sort.Ints(ids)
	values := make([]string, len(ids))
	selected := make([]GPUInfo, len(ids))
	for i, id := range ids {
		values[i] = strconv.Itoa(id)
		selected[i] = available[id]
	}
	return fmt.Sprintf("device=%s", strings.Join(values, ",")), selected, nil
}

func ensureManagedContainer(containerID string) error {
	containerID = strings.TrimSpace(containerID)
	if containerID == "" {
		return validationError("container id is required")
	}

	return withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		metadata, err := inspectContainer(ctx, cli, containerID)
		if err != nil {
			return fmt.Errorf("failed to inspect container %s: %v", containerID, err)
		}

		name := strings.TrimPrefix(metadata.Name, "/")
		labels := map[string]string(nil)
		if metadata.Config != nil {
			labels = metadata.Config.Labels
		}
		if labels["app"] == "vllm-manager" || labels[managedContainerLabel] == "true" || strings.HasPrefix(name, "vllm-") {
			return nil
		}

		return validationError("container %s is not managed by vllm-manager", containerID)
	})
}

func GetActiveModels() ([]ActiveModel, error) {
	var models []ActiveModel
	err := withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		containers, err := listContainers(ctx, cli)
		if err != nil {
			return fmt.Errorf("docker container list failed: %v", err)
		}

		for _, item := range containers {
			name := dockerContainerName(item)
			if name == "" || !strings.HasPrefix(name, "vllm-") {
				continue
			}
			// Download containers (vllm-download-*) hold no GPU allocation and no
			// vLLM server port — exclude them from the active model list so they
			// don't poison GPU overlap accounting or the UI model table.
			if strings.HasPrefix(name, "vllm-download-") {
				continue
			}

			// Inspect the container to get args + GPU device assignments.
			metadata, err := inspectContainer(ctx, cli, item.ID)
			if err != nil {
				continue
			}

			args := metadata.Args
			labels := map[string]string(nil)
			if metadata.Config != nil {
				labels = metadata.Config.Labels
				if len(args) == 0 {
					args = metadata.Config.Cmd
				}
			}
			var deviceRequests []container.DeviceRequest
			if metadata.HostConfig != nil {
				deviceRequests = metadata.HostConfig.DeviceRequests
			}

			model := ActiveModel{
				ContainerID: item.ID,
				Model:       name, // fallback
				Port:        "unknown",
				GPUs:        "unknown",
				GPUIDs:      "all",
				MemUtil:     "unknown",
				MaxLen:      "default",
				ToolParser:  "disabled",
				APIUrl:      "unknown",
				IsReady:     false,
				State:       string(item.State),
			}

			if labels[containerModelLabel] != "" {
				model.Model = labels[containerModelLabel]
			}

			for i := 0; i < len(args); i++ {
				if args[i] == "--model" && i+1 < len(args) && labels[containerModelLabel] == "" {
					model.Model = args[i+1]
				}
				if args[i] == "--port" && i+1 < len(args) {
					model.Port = args[i+1]
				}
				if args[i] == "--tensor-parallel-size" && i+1 < len(args) {
					model.GPUs = args[i+1]
				}
				if args[i] == "--gpu-memory-utilization" && i+1 < len(args) {
					model.MemUtil = args[i+1]
				}
				if args[i] == "--max-model-len" && i+1 < len(args) {
					model.MaxLen = args[i+1]
				}
				if args[i] == "--tool-call-parser" && i+1 < len(args) {
					model.ToolParser = args[i+1]
				}
			}

			// Parse actual GPU IDs from DeviceRequests (--gpus device=0,1 → DeviceIDs: ["0","1"]).
			// If DeviceIDs is empty, the container was launched with --gpus all.
			for _, dr := range deviceRequests {
				if len(dr.DeviceIDs) > 0 {
					model.GPUIDs = strings.Join(dr.DeviceIDs, ",")
				}
				// else: GPUIDs stays "all"
			}

			if model.Port != "unknown" {
				model.APIUrl = fmt.Sprintf("http://localhost:%s/v1", model.Port)

				// Vision Heuristics.
				lower := strings.ToLower(model.Model)
				if strings.Contains(lower, "vl") || strings.Contains(lower, "vision") || strings.Contains(lower, "llava") || strings.Contains(lower, "pixtral") {
					model.Arch = "Vision + Text"
				} else {
					model.Arch = "Text-Only"
				}
			}

			models = append(models, model)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Health checks: run in parallel to avoid sequential timeouts stacking up.
	var wg sync.WaitGroup
	for i := range models {
		if models[i].Port == "unknown" || models[i].State != "running" {
			continue
		}
		wg.Add(1)
		go func(m *ActiveModel) {
			defer wg.Done()
			client := http.Client{Timeout: modelHealthCheckTimeout}
			resp, err := client.Get(fmt.Sprintf("http://localhost:%s/v1/models", m.Port))
			if err != nil {
				return
			}
			defer resp.Body.Close()
			m.IsReady = (resp.StatusCode == http.StatusOK)
			if m.IsReady {
				var modelResp struct {
					Data []struct {
						MaxModelLen int `json:"max_model_len"`
					} `json:"data"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&modelResp); err == nil && len(modelResp.Data) > 0 {
					if m.MaxLen == "default" {
						m.MaxLen = strconv.Itoa(modelResp.Data[0].MaxModelLen)
					}
				}
			}
		}(&models[i])
	}
	wg.Wait()

	return models, nil
}

func GetVLLMContainerNames() ([]string, error) {
	var names []string
	err := withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		containers, err := listContainers(ctx, cli)
		if err != nil {
			return fmt.Errorf("docker container list failed: %v", err)
		}
		for _, item := range containers {
			name := dockerContainerName(item)
			if name == "" || !strings.HasPrefix(name, "vllm-") {
				continue
			}
			names = append(names, name)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	return names, nil
}

func StartModel(model string, port int, gpus int, gpuIds string, memUtil string, maxLen string, toolParser string, hfCache string, hfToken string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return validationError("model is required")
	}
	if port < 1 || port > 65535 {
		return validationError("port must be between 1 and 65535")
	}
	if gpus < 1 {
		return validationError("must specify at least 1 GPU")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	if containerName, err := findDownloadContainerForModel(model); err != nil {
		return fmt.Errorf("failed to verify download containers before starting model: %v", err)
	} else if containerName != "" {
		return validationError("cannot start %s because it is currently being downloaded by %s", model, containerName)
	}

	utilization, err := parseGPUMemoryUtilization(memUtil)
	if err != nil {
		return err
	}
	memUtil = strconv.FormatFloat(utilization, 'f', -1, 64)

	if _, err := parseMaxModelLen(maxLen); err != nil {
		return err
	}

	if toolParser != "" {
		matched, _ := regexp.MatchString(`^[a-zA-Z0-9\-]+$`, toolParser)
		if !matched {
			return validationError("invalid tool parser format")
		}
	}

	// Check if port is already in use
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return validationError("port %d is already in use", port)
	}
	l.Close()

	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	shortName = sanitizeDockerNamePart(shortName)
	containerName := fmt.Sprintf("vllm-%s-%d", shortName, port)

	gpuFlag, selectedGPUs, err := validateGPUSelection(gpus, gpuIds)
	if err != nil {
		return err
	}

	// Fix #2 / GPT-P1a: reject if combined mem_util on any requested GPU would exceed 1.0
	totalGPUCount := len(selectedGPUs)
	if totalGPUCount == 0 {
		if gpuList, err := GetGPUInfo(); err == nil {
			totalGPUCount = len(gpuList)
		}
	}
	var requestedGPUIDs []int
	if gpuIds != "" && gpuIds != "all" {
		for _, part := range strings.Split(gpuIds, ",") {
			if id, err := strconv.Atoi(strings.TrimSpace(part)); err == nil {
				requestedGPUIDs = append(requestedGPUIDs, id)
			}
		}
	}
	// selectedGPUs is now always an explicit list (GPT-P2a fix), so derive IDs from it
	// when gpuIds was "all" — requestedGPUIDs will be empty in that case.
	if len(requestedGPUIDs) == 0 && len(selectedGPUs) > 0 {
		for _, g := range selectedGPUs {
			requestedGPUIDs = append(requestedGPUIDs, g.ID)
		}
	}
	if err := checkGPUOverlap(requestedGPUIDs, totalGPUCount, utilization); err != nil {
		return err
	}

	if err := CheckModelFitsInVRAM(model, hfCache, selectedGPUs, gpus, memUtil, maxLen); err != nil {
		return err
	}

	// Deploy should create a fresh container only. Existing containers keep their
	// original GPU devices, env, image, and volume mounts, so restarting them here
	// can silently run a different configuration than the user requested.
	if names, err := GetVLLMContainerNames(); err == nil {
		for _, name := range names {
			if name == containerName {
				return validationError("container %q already exists; remove it before deploying a new model with this name and port", containerName)
			}
		}
	}

	labels := map[string]string{
		"app":                 "vllm-manager",
		managedContainerLabel: "true",
		containerKindLabel:    containerKindServer,
		containerModelLabel:   model,
	}
	env := []string{"HF_HOME=/hf_cache"}

	if hfToken != "" {
		env = append(env, fmt.Sprintf("HF_TOKEN=%s", hfToken))
	}

	// Warn if the configured image is not present locally.
	checkVLLMImageExists()

	cmd := []string{
		"--model", model,
		"--tensor-parallel-size", strconv.Itoa(gpus),
		"--port", strconv.Itoa(port),
		"--gpu-memory-utilization", memUtil,
	}

	if maxLen != "" {
		cmd = append(cmd, "--max-model-len", maxLen)
	}
	if toolParser != "" {
		cmd = append(cmd, "--tool-call-parser", toolParser, "--enable-auto-tool-choice")
	}

	return withDockerClient(0, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, vllmImage); err != nil {
			return fmt.Errorf("failed to ensure vLLM image: %v", err)
		}

		logDockerAPI(
			"container create",
			containerName,
			"image="+vllmImage,
			"ipc=host",
			"runtime=nvidia",
			"gpus="+gpuFlag,
			"network=host",
			"bind="+fmt.Sprintf("%s:/hf_cache", hfCache),
			"cmd="+strings.Join(cmd, " "),
		)
		created, err := cli.ContainerCreate(ctx, dockerclient.ContainerCreateOptions{
			Name: containerName,
			Config: &container.Config{
				Image:  vllmImage,
				Cmd:    cmd,
				Env:    env,
				Labels: labels,
			},
			HostConfig: &container.HostConfig{
				IpcMode:     container.IPCModeHost,
				Runtime:     "nvidia",
				NetworkMode: container.NetworkMode("host"),
				Binds:       []string{fmt.Sprintf("%s:/hf_cache", hfCache)},
				Resources: container.Resources{
					DeviceRequests: []container.DeviceRequest{dockerDeviceRequest(gpuFlag)},
				},
			},
		})
		if err != nil {
			return err
		}

		logDockerAPI("container start", created.ID)
		if _, err := cli.ContainerStart(ctx, created.ID, dockerclient.ContainerStartOptions{}); err != nil {
			return err
		}
		return nil
	})
}

func StopModel(port int) error {
	suffix := fmt.Sprintf("-%d", port)
	stopped := false
	err := withDockerClient(execTimeoutLong, func(ctx context.Context, cli *dockerclient.Client) error {
		containers, err := listContainers(ctx, cli)
		if err != nil {
			return err
		}
		for _, item := range containers {
			name := dockerContainerName(item)
			if strings.HasPrefix(name, "vllm-") && strings.HasSuffix(name, suffix) {
				logDockerAPI("container stop", name)
				if _, err := cli.ContainerStop(ctx, name, dockerclient.ContainerStopOptions{}); err != nil {
					return err
				}
				stopped = true
				return nil
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if stopped {
		return nil
	}
	return validationError("no vLLM server found on port %d", port)
}

func RemoveModel(containerID string) error {
	if err := ensureManagedContainer(containerID); err != nil {
		return err
	}
	return withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		logDockerAPI("container remove", containerID, "force=true")
		if _, err := cli.ContainerRemove(ctx, containerID, dockerclient.ContainerRemoveOptions{Force: true}); err != nil {
			return fmt.Errorf("failed to remove container %s: %v", containerID, err)
		}
		return nil
	})
}

func RestartModel(containerID string) error {
	if err := ensureManagedContainer(containerID); err != nil {
		return err
	}
	return withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		logDockerAPI("container start", containerID)
		if _, err := cli.ContainerStart(ctx, containerID, dockerclient.ContainerStartOptions{}); err != nil {
			return fmt.Errorf("failed to restart container %s: %v", containerID, err)
		}
		return nil
	})
}
