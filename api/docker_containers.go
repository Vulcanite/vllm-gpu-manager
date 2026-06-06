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
	"time"
	"unicode"

	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

const (
	modelHealthCheckTimeout        = 5 * time.Second
	modelHealthCheckOverallTimeout = 5 * time.Second
)

func validateGPUSelection(gpus int, gpuIds string) (string, []GPUInfo, error) {
	gpuList, err := RequireDetectedGPUs()
	if err != nil {
		return "", nil, err
	}
	if gpus > len(gpuList) {
		return "", nil, validationError("requested %d GPUs, but only %d are available on this server", gpus, len(gpuList))
	}

	gpuIds = strings.TrimSpace(gpuIds)
	if gpuIds == "" || gpuIds == "all" {
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

func containerLabels(metadata container.InspectResponse) map[string]string {
	if metadata.Config == nil {
		return nil
	}
	return metadata.Config.Labels
}

func isManagedContainer(labels map[string]string) bool {
	return labels[managedContainerLabel] == "true" || labels["app"] == "vllm-manager"
}

func isManagedServerContainer(labels map[string]string) bool {
	if !isManagedContainer(labels) {
		return false
	}
	kind := labels[containerKindLabel]
	return kind == "" || kind == containerKindServer
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

		if isManagedContainer(containerLabels(metadata)) {
			return nil
		}

		return validationError("container %s is not managed by vllm-manager", containerID)
	})
}

type managedVLLMContainer struct {
	id       string
	name     string
	state    string
	metadata container.InspectResponse
	labels   map[string]string
}

// listManagedVLLMContainers enumerates all vllm-manager-owned containers with a
// single docker list followed by one inspect per container. Callers derive
// active models, managed container names, and download lookups from this one
// snapshot instead of repeatedly re-listing and re-inspecting the daemon.
func listManagedVLLMContainers(ctx context.Context, cli *dockerclient.Client) ([]managedVLLMContainer, error) {
	containers, err := listContainers(ctx, cli)
	if err != nil {
		return nil, fmt.Errorf("docker container list failed: %v", err)
	}

	var result []managedVLLMContainer
	for _, item := range containers {
		name := dockerContainerName(item)
		if name == "" || !strings.HasPrefix(name, "vllm-") {
			continue
		}
		metadata, err := inspectContainer(ctx, cli, item.ID)
		if err != nil {
			continue
		}
		labels := containerLabels(metadata)
		if !isManagedContainer(labels) {
			continue
		}
		result = append(result, managedVLLMContainer{
			id:       item.ID,
			name:     name,
			state:    string(item.State),
			metadata: metadata,
			labels:   labels,
		})
	}
	return result, nil
}

// snapshotManagedVLLMContainers takes a single docker snapshot of all managed
// vLLM containers.
func snapshotManagedVLLMContainers() ([]managedVLLMContainer, error) {
	var containers []managedVLLMContainer
	err := withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		result, err := listManagedVLLMContainers(ctx, cli)
		containers = result
		return err
	})
	return containers, err
}

func managedContainerNames(containers []managedVLLMContainer) []string {
	names := make([]string, 0, len(containers))
	for _, vc := range containers {
		names = append(names, vc.name)
	}
	sort.Strings(names)
	return names
}

// buildActiveModel converts an inspected container into an ActiveModel. It
// returns false for anything that is not a managed vLLM server container
// (download containers hold no GPU allocation or server port and must not
// poison GPU overlap accounting or the UI model table).
func buildActiveModel(vc managedVLLMContainer) (ActiveModel, bool) {
	if !isManagedServerContainer(vc.labels) || strings.HasPrefix(vc.name, "vllm-download-") {
		return ActiveModel{}, false
	}

	metadata := vc.metadata
	args := metadata.Args
	if metadata.Config != nil && len(args) == 0 {
		args = metadata.Config.Cmd
	}
	var deviceRequests []container.DeviceRequest
	if metadata.HostConfig != nil {
		deviceRequests = metadata.HostConfig.DeviceRequests
	}

	model := ActiveModel{
		ContainerID:   vc.id,
		ContainerName: vc.name,
		Model:         vc.name, // fallback
		Port:          "unknown",
		GPUs:          "unknown",
		GPUIDs:        "all",
		MemUtil:       "unknown",
		MaxLen:        "default",
		KVCacheDType:  "auto",
		ToolParser:    "disabled",
		APIUrl:        "unknown",
		IsReady:       false,
		State:         vc.state,
	}

	if vc.labels[containerModelLabel] != "" {
		model.Model = vc.labels[containerModelLabel]
	}

	for i := 0; i < len(args); i++ {
		if args[i] == "--model" && i+1 < len(args) && vc.labels[containerModelLabel] == "" {
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
		if args[i] == "--kv-cache-dtype" && i+1 < len(args) {
			model.KVCacheDType = args[i+1]
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

	return model, true
}

func activeModelsFromContainers(containers []managedVLLMContainer) []ActiveModel {
	var models []ActiveModel
	for _, vc := range containers {
		if model, ok := buildActiveModel(vc); ok {
			models = append(models, model)
		}
	}
	return models
}

// activeModelsNoHealth returns active server models without the per-container
// HTTP readiness probes. Internal callers that only need GPU assignments (GPU
// overlap and cache-deletion checks) use this to avoid the health-check
// round-trips that GetActiveModels performs for the UI.
func activeModelsNoHealth() ([]ActiveModel, error) {
	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return nil, err
	}
	return activeModelsFromContainers(containers), nil
}

func GetActiveModels() ([]ActiveModel, error) {
	models, err := activeModelsNoHealth()
	if err != nil {
		return nil, err
	}
	healthCheckActiveModels(models)
	return models, nil
}

type activeModelHealthResult struct {
	index  int
	ready  bool
	maxLen string
}

func healthCheckActiveModels(models []ActiveModel) {
	ctx, cancel := context.WithTimeout(context.Background(), modelHealthCheckOverallTimeout)
	defer cancel()

	targets := 0
	results := make(chan activeModelHealthResult, len(models))
	for i := range models {
		if models[i].Port == "unknown" || models[i].State != "running" {
			continue
		}
		targets++
		go func(index int, port string, currentMaxLen string) {
			client := http.Client{Timeout: modelHealthCheckTimeout}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://localhost:%s/v1/models", port), nil)
			if err != nil {
				return
			}
			resp, err := client.Do(req)
			if err != nil {
				return
			}
			defer resp.Body.Close()
			result := activeModelHealthResult{
				index: index,
				ready: resp.StatusCode == http.StatusOK,
			}
			if result.ready && currentMaxLen == "default" {
				var modelResp struct {
					Data []struct {
						MaxModelLen int `json:"max_model_len"`
					} `json:"data"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&modelResp); err == nil && len(modelResp.Data) > 0 {
					result.maxLen = strconv.Itoa(modelResp.Data[0].MaxModelLen)
				}
			}
			select {
			case results <- result:
			case <-ctx.Done():
			}
		}(i, models[i].Port, models[i].MaxLen)
	}

	for i := 0; i < targets; i++ {
		select {
		case result := <-results:
			models[result.index].IsReady = result.ready
			if result.maxLen != "" {
				models[result.index].MaxLen = result.maxLen
			}
		case <-ctx.Done():
			return
		}
	}
}

func GetVLLMContainerNames() ([]string, error) {
	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return nil, err
	}
	return managedContainerNames(containers), nil
}

func StartModel(model string, port int, gpus int, gpuIds string, memUtil string, maxLen string, kvCacheDType string, toolParser string, hfCache string, hfToken string, vllmAPIKey string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}
	vllmAPIKey = strings.TrimSpace(vllmAPIKey)
	if strings.IndexFunc(vllmAPIKey, unicode.IsSpace) >= 0 {
		return validationError("vLLM API key cannot contain whitespace")
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

	// Single docker snapshot reused for the download-in-progress, GPU overlap,
	// and name-conflict checks below instead of three separate enumerations.
	snapshot, err := snapshotManagedVLLMContainers()
	if err != nil {
		return fmt.Errorf("failed to inspect existing containers before starting model: %v", err)
	}

	if downloadName := downloadContainerForModel(snapshot, model); downloadName != "" {
		return validationError("cannot start %s because it is currently being downloaded by %s", model, downloadName)
	}

	utilization, err := parseGPUMemoryUtilization(memUtil)
	if err != nil {
		return err
	}
	memUtil = strconv.FormatFloat(utilization, 'f', -1, 64)

	if _, err := parseMaxModelLen(maxLen); err != nil {
		return err
	}

	kvCacheDType, err = normalizeKVCacheDType(kvCacheDType)
	if err != nil {
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
	if err := checkGPUOverlap(activeModelsFromContainers(snapshot), requestedGPUIDs, totalGPUCount, utilization); err != nil {
		return err
	}

	if err := CheckModelFitsInVRAM(model, hfCache, selectedGPUs, gpus, memUtil, maxLen, kvCacheDType); err != nil {
		return err
	}

	// Deploy should create a fresh container only. Existing containers keep their
	// original GPU devices, env, image, and volume mounts, so restarting them here
	// can silently run a different configuration than the user requested.
	for _, vc := range snapshot {
		if vc.name == containerName {
			return validationError("container %q already exists; remove it before deploying a new model with this name and port", containerName)
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
	if kvCacheDType != "auto" {
		cmd = append(cmd, "--kv-cache-dtype", kvCacheDType)
	}
	if toolParser != "" {
		cmd = append(cmd, "--tool-call-parser", toolParser, "--enable-auto-tool-choice")
	}
	if vllmAPIKey != "" {
		cmd = append(cmd, "--api-key", vllmAPIKey)
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
				metadata, err := inspectContainer(ctx, cli, item.ID)
				if err != nil {
					return fmt.Errorf("failed to inspect container %s: %v", item.ID, err)
				}
				if !isManagedServerContainer(containerLabels(metadata)) {
					continue
				}
				logDockerAPI("container stop", name)
				if _, err := cli.ContainerStop(ctx, item.ID, dockerclient.ContainerStopOptions{}); err != nil {
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
	return withDockerClient(execTimeoutLong, func(ctx context.Context, cli *dockerclient.Client) error {
		metadata, err := inspectContainer(ctx, cli, containerID)
		if err != nil {
			return fmt.Errorf("failed to inspect container %s: %v", containerID, err)
		}
		if metadata.State != nil && metadata.State.Running {
			timeout := int(execTimeoutLong.Seconds())
			logDockerAPI("container restart", containerID)
			if _, err := cli.ContainerRestart(ctx, containerID, dockerclient.ContainerRestartOptions{Timeout: &timeout}); err != nil {
				return fmt.Errorf("failed to restart container %s: %v", containerID, err)
			}
			return nil
		}

		logDockerAPI("container start", containerID)
		if _, err := cli.ContainerStart(ctx, containerID, dockerclient.ContainerStartOptions{}); err != nil {
			return fmt.Errorf("failed to start container %s: %v", containerID, err)
		}
		return nil
	})
}
