package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/netip"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/moby/moby/api/pkg/stdcopy"
	"github.com/moby/moby/api/types/container"
	"github.com/moby/moby/api/types/network"
	dockerclient "github.com/moby/moby/client"
)

const (
	modelHealthCheckTimeout = 5 * time.Second
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
		MaxNumSeqs:    "default",
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
		if args[i] == "--max-num-seqs" && i+1 < len(args) {
			model.MaxNumSeqs = args[i+1]
		}
		if args[i] == "--kv-cache-dtype" && i+1 < len(args) {
			model.KVCacheDType = args[i+1]
		}
		if args[i] == "--tool-call-parser" && i+1 < len(args) {
			model.ToolParser = args[i+1]
		}
		if args[i] == "--api-key" && i+1 < len(args) {
			model.APIKey = args[i+1]
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

// GetManagedModelByContainerID returns metadata for a managed serving
// container without running readiness probes. containerID may be a full ID,
// short ID, or container name.
func GetManagedModelByContainerID(containerID string) (ActiveModel, bool, error) {
	containerID = strings.TrimSpace(strings.TrimPrefix(containerID, "/"))
	if containerID == "" {
		return ActiveModel{}, false, validationError("container id is required")
	}
	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return ActiveModel{}, false, err
	}
	for _, vc := range containers {
		if vc.id != containerID && !strings.HasPrefix(vc.id, containerID) && vc.name != containerID {
			continue
		}
		model, ok := buildActiveModel(vc)
		return model, ok, nil
	}
	return ActiveModel{}, false, nil
}

func GetActiveModels(vllmAPIKey string) ([]ActiveModel, error) {
	models, err := activeModelsNoHealth()
	if err != nil {
		return nil, err
	}
	healthCheckActiveModels(models, vllmAPIKey)
	return models, nil
}

// GetServerContainersUsingAPIKey returns managed server containers, running or
// stopped, that were launched with the given vLLM --api-key.
func GetServerContainersUsingAPIKey(apiKey string) ([]ActiveModel, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, nil
	}
	models, err := activeModelsNoHealth()
	if err != nil {
		return nil, err
	}
	var matches []ActiveModel
	for _, model := range models {
		if model.APIKey == apiKey {
			matches = append(matches, model)
		}
	}
	return matches, nil
}

type activeModelHealthResult struct {
	index  int
	ready  bool
	maxLen string
}

var activeModelHealthProbe = dockerActiveModelHealthProbe

func healthCheckActiveModels(models []ActiveModel, fallbackAPIKey string) {
	results := make(chan activeModelHealthResult, len(models))
	launched := 0
	for i := range models {
		if models[i].Port == "unknown" || models[i].State != "running" || models[i].ContainerID == "" {
			continue
		}
		effectiveKey := models[i].APIKey
		if effectiveKey == "" {
			effectiveKey = fallbackAPIKey
		}
		go func(index int, containerID string, port string, currentMaxLen string, apiKey string) {
			result := activeModelHealthResult{index: index}
			result.ready, result.maxLen = activeModelHealthProbe(containerID, port, currentMaxLen, apiKey)
			select {
			case results <- result:
			default:
			}
		}(i, models[i].ContainerID, models[i].Port, models[i].MaxLen, effectiveKey)
		launched++
	}

	for i := 0; i < launched; i++ {
		result := <-results
		models[result.index].IsReady = result.ready
		if result.maxLen != "" {
			models[result.index].MaxLen = result.maxLen
		}
	}
}

func dockerActiveModelHealthProbe(containerID string, port string, currentMaxLen string, apiKey string) (bool, string) {
	ready := false
	maxLen := ""
	_ = withDockerClient(modelHealthCheckTimeout, func(dCtx context.Context, cli *dockerclient.Client) error {
		curlArgs := []string{
			"curl", "-sf",
			"--max-time", "4",
			"-o", "/dev/null",
			"-w", "%{http_code}",
		}
		if apiKey != "" {
			curlArgs = append(curlArgs, "-H", "Authorization: Bearer "+apiKey)
		}
		curlArgs = append(curlArgs, fmt.Sprintf("http://127.0.0.1:%s/v1/models", port))

		execCreate, err := cli.ExecCreate(dCtx, containerID, dockerclient.ExecCreateOptions{
			Cmd:          curlArgs,
			AttachStdout: true,
			AttachStderr: false,
		})
		if err != nil {
			return err
		}

		attach, err := cli.ExecAttach(dCtx, execCreate.ID, dockerclient.ExecAttachOptions{})
		if err != nil {
			return err
		}
		defer attach.Close()

		var stdoutBuf bytes.Buffer
		if _, err := stdcopy.StdCopy(&stdoutBuf, io.Discard, attach.Reader); err != nil {
			return err
		}
		attach.Close()

		statusCode := strings.TrimSpace(stdoutBuf.String())
		ready = statusCode == "200"

		if ready && currentMaxLen == "default" {
			var bodyBuf bytes.Buffer
			curlFull := append([]string{
				"curl", "-sf",
				"--max-time", "4",
			}, func() []string {
				if apiKey != "" {
					return []string{"-H", "Authorization: Bearer " + apiKey}
				}
				return nil
			}()...)
			curlFull = append(curlFull, fmt.Sprintf("http://127.0.0.1:%s/v1/models", port))

			execBody, err := cli.ExecCreate(dCtx, containerID, dockerclient.ExecCreateOptions{
				Cmd:          curlFull,
				AttachStdout: true,
				AttachStderr: false,
			})
			if err == nil {
				attachBody, err := cli.ExecAttach(dCtx, execBody.ID, dockerclient.ExecAttachOptions{})
				if err == nil {
					defer attachBody.Close()
					if _, err := stdcopy.StdCopy(&bodyBuf, io.Discard, attachBody.Reader); err == nil {
						var modelResp struct {
							Data []struct {
								MaxModelLen int `json:"max_model_len"`
							} `json:"data"`
						}
						if err := json.NewDecoder(&bodyBuf).Decode(&modelResp); err == nil && len(modelResp.Data) > 0 {
							maxLen = strconv.Itoa(modelResp.Data[0].MaxModelLen)
						}
					}
				}
			}
		}
		return nil
	})
	return ready, maxLen
}

func GetVLLMContainerNames() ([]string, error) {
	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return nil, err
	}
	return managedContainerNames(containers), nil
}

func normalizeHostBindIP(hostBindIP string) (netip.Addr, error) {
	hostBindIP = strings.TrimSpace(hostBindIP)
	if hostBindIP == "" {
		hostBindIP = "127.0.0.1"
	}
	if hostBindIP != "127.0.0.1" && hostBindIP != "0.0.0.0" {
		return netip.Addr{}, validationError("host bind IP must be 127.0.0.1 or 0.0.0.0")
	}
	return netip.MustParseAddr(hostBindIP), nil
}

// allowedToolParsers is the explicit allowlist of vLLM tool-call parsers. vLLM
// passes this string to its built-in registry and forwards it to Python, so
// anything outside this set is rejected as a hardening measure.
//
// Updating this list:
//
//  1. Check the vLLM release notes or source for new parser names.
//  2. Add the new parser to this map using the exact name vLLM expects.
//  3. Update the dashboard's LaunchDefaults (which reads this map) so the
//     frontend <select> stays in sync.
//  4. Bump the version label (vllm-manager.version) when releasing.
//
// The list is exposed to the dashboard via LaunchDefaults so the <select> in
// DeployServerCard cannot drift from the backend.
var allowedToolParsers = map[string]struct{}{
	"hermes":             {},
	"llama3_json":        {},
	"mistral":            {},
	"granite":            {},
	"granite4":           {},
	"granite-20b-fc":     {},
	"internlm":           {},
	"jamba":              {},
	"pythonic":           {},
	"python_tool":        {},
	"jinja":              {},
	"guidance":           {},
	"lm_format_enforcer": {},
	"xlam":               {},
	"qwen2":              {},
	"qwen3_coder":        {},
	"qwen3_xml":          {},
	"llama4_pythonic":    {},
	"deepseek_v3":        {},
	"deepseek_v31":       {},
	"openai":             {},
	"kimi_k2":            {},
	"hunyuan_a13b":       {},
	"cohere_command3":    {},
	"longcat":            {},
	"glm45":              {},
	"glm47":              {},
	"functiongemma":      {},
	"olmo3":              {},
	"gigachat3":          {},
	"apertus":            {},
}

// allowedToolParserNames returns the allowlist as a sorted slice so the
// dashboard can render an authoritative <select> without re-listing the values
// in two places.
func allowedToolParserNames() []string {
	names := make([]string, 0, len(allowedToolParsers))
	for name := range allowedToolParsers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// normalizeModelKey lowercases the user-visible model identifier and
// validates it against a conservative character class. It is used as the key
// for routing table lookups so case-different labels do not produce
// duplicate entries. It returns "" for invalid input; callers should treat
// that as a validation error.
func normalizeModelKey(model string) string {
	model = strings.TrimSpace(strings.ToLower(model))
	if model == "" || len(model) > 128 {
		return ""
	}
	for i, r := range model {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9':
		case r == '.' || r == '_' || r == '-' || r == '/':
		default:
			return ""
		}
		if i == 0 && (r == '.' || r == '_' || r == '-' || r == '/') {
			return ""
		}
	}
	return model
}

// NormalizeModelKey validates and canonicalizes a user-supplied model routing
// key.
func NormalizeModelKey(model string) string {
	return normalizeModelKey(model)
}

// NormalizeModelKeyForMatch is a looser, internal matcher used for comparing
// against values already stored in Docker labels. It lowercases and trims the
// input and returns "" for empty results, but it does NOT enforce the strict
// character class — labels written by older versions may contain characters
// we no longer accept on input, and we still want to match them.
func NormalizeModelKeyForMatch(model string) string {
	model = strings.TrimSpace(strings.ToLower(model))
	if model == "" {
		return ""
	}
	return model
}

func StartModel(model string, port int, gpus int, gpuIds string, memUtil string, maxLen string, maxNumSeqs string, kvCacheDType string, toolParser string, hfCache string, hfToken string, vllmAPIKey string, enableCUDACompatibility bool, hfHubOffline bool, hostBindIP string, serverAPIKey string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}
	vllmAPIKey = strings.TrimSpace(vllmAPIKey)
	if strings.IndexFunc(vllmAPIKey, unicode.IsSpace) >= 0 {
		return validationError("vLLM API key cannot contain whitespace")
	}
	// serverAPIKey is the optional per-container override. The caller is
	// expected to set either the global vllmAPIKey or a per-server key,
	// but never both. We forward whichever is set as `--api-key` so vLLM
	// applies the more specific secret.
	serverAPIKey = strings.TrimSpace(serverAPIKey)
	if strings.IndexFunc(serverAPIKey, unicode.IsSpace) >= 0 {
		return validationError("per-server API key cannot contain whitespace")
	}
	if vllmAPIKey != "" && serverAPIKey != "" {
		return validationError("vLLM API key and per-server API key are mutually exclusive; pass at most one")
	}
	effectiveAPIKey := vllmAPIKey
	if serverAPIKey != "" {
		effectiveAPIKey = serverAPIKey
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
	hostBindAddr, err := normalizeHostBindIP(hostBindIP)
	if err != nil {
		return err
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

	// Compute the container name up front so we can reject the request before
	// doing a net.Listen probe when a stale (stopped or running) container
	// already owns this name/port. Without this guard the user sees a
	// generic "container already exists" error only after the listener check
	// passes for a stopped container, which is confusing.
	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	shortName = sanitizeDockerNamePart(shortName)
	containerName := fmt.Sprintf("vllm-%s-%d", shortName, port)

	// Reject if any managed vLLM container (stopped or running) already
	// claims the target port via its name suffix. We use the suffix because
	// the port alone could be reused by a foreign process — the user
	// explicitly reserved `-<port>` for vllm-manager containers, and a
	// stale one would collide on the next start.
	portSuffix := fmt.Sprintf("-%d", port)
	for _, vc := range snapshot {
		if vc.name == containerName {
			return validationError("port %d is reserved by existing container %q (%s); remove it first or pick a different port", port, vc.name, vc.state)
		}
		if strings.HasSuffix(vc.name, portSuffix) && vc.name != containerName {
			return validationError("port %d is reserved by existing container %q (%s) with a different model name; remove it first or pick a different port", port, vc.name, vc.state)
		}
	}

	utilization, err := parseGPUMemoryUtilization(memUtil)
	if err != nil {
		return err
	}
	memUtil = strconv.FormatFloat(utilization, 'f', -1, 64)

	if _, err := parseMaxModelLen(maxLen); err != nil {
		return err
	}

	if _, err := parseMaxNumSeqs(maxNumSeqs); err != nil {
		return err
	}

	kvCacheDType, err = normalizeKVCacheDType(kvCacheDType)
	if err != nil {
		return err
	}

	if toolParser != "" {
		if _, ok := allowedToolParsers[toolParser]; !ok {
			allowed := allowedToolParserNames()
			return validationError("unsupported tool parser %q; allowed: %s", toolParser, strings.Join(allowed, ", "))
		}
	}

	// Check if port is already in use and hold the listener until the container
	// is created.  This eliminates the TOCTOU window where another process could
	// bind the port between the listen check and ContainerCreate.  If container
	// creation fails the listener is closed and the port is released; if it
	// succeeds the listener is closed after the container is started (the
	// container's own port binding takes over).
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return validationError("port %d is already in use", port)
	}
	listenerOpen := true
	closeReservedPort := func() {
		if listenerOpen {
			_ = l.Close()
			listenerOpen = false
		}
	}
	defer closeReservedPort()

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

	// The container-name / port-suffix collision check moved earlier in
	// StartModel so the user gets a clear "reserved by existing container"
	// error before we even probe the listener.

	labels := map[string]string{
		"app":                  "vllm-manager",
		managedContainerLabel:  "true",
		containerKindLabel:     containerKindServer,
		containerModelLabel:    model,
		"vllm-manager.version": managerVersion,
	}
	env := []string{"HF_HOME=/hf_cache"}

	if hfToken != "" {
		env = append(env, fmt.Sprintf("HF_TOKEN=%s", hfToken))
	}
	if enableCUDACompatibility {
		env = append(env, "VLLM_ENABLE_CUDA_COMPATIBILITY=1")
	}
	if hfHubOffline {
		env = append(env, "HF_HUB_OFFLINE=1")
	}

	// Warn if the configured image is not present locally.
	checkVLLMImageExists()

	cmd := []string{
		"--model", model,
		"--host", "0.0.0.0",
		"--tensor-parallel-size", strconv.Itoa(gpus),
		"--port", strconv.Itoa(port),
		"--gpu-memory-utilization", memUtil,
		"--enable-prefix-caching",
	}

	if maxLen != "" {
		cmd = append(cmd, "--max-model-len", maxLen)
	}
	if maxNumSeqs != "" {
		cmd = append(cmd, "--max-num-seqs", maxNumSeqs)
	}
	if kvCacheDType != "auto" {
		cmd = append(cmd, "--kv-cache-dtype", kvCacheDType)
	}
	if toolParser != "" {
		cmd = append(cmd, "--tool-call-parser", toolParser, "--enable-auto-tool-choice")
	}
	if effectiveAPIKey != "" {
		cmd = append(cmd, "--api-key", effectiveAPIKey)
	}

	return withDockerClient(execTimeoutStart, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, VLLMImage); err != nil {
			return fmt.Errorf("failed to ensure vLLM image: %v", err)
		}

		portStr := strconv.Itoa(port)
		logDockerAPI(
			"container create",
			containerName,
			"image="+VLLMImage,
			"ipc=host",
			"runtime=nvidia",
			"gpus="+gpuFlag,
			"ports="+hostBindAddr.String()+":"+portStr+":"+portStr,
			"network=bridge",
			"bind="+fmt.Sprintf("%s:/hf_cache", hfCache),
			"cmd="+strings.Join(cmd, " "),
		)
		p, _ := network.ParsePort(portStr + "/tcp")
		var portBinding network.PortMap
		exposedPorts := network.PortSet{}
		networkMode := container.NetworkMode("bridge")

		portBinding = network.PortMap{
			p: []network.PortBinding{
				{
					HostIP:   hostBindAddr,
					HostPort: portStr,
				},
			},
		}
		exposedPorts = network.PortSet{
			p: struct{}{},
		}

		created, err := cli.ContainerCreate(ctx, dockerclient.ContainerCreateOptions{
			Name: containerName,
			Config: &container.Config{
				Image:        VLLMImage,
				Cmd:          cmd,
				Env:          env,
				Labels:       labels,
				ExposedPorts: exposedPorts,
			},
			HostConfig: &container.HostConfig{
				IpcMode:      container.IPCModeHost,
				Runtime:      "nvidia",
				NetworkMode:  networkMode,
				PortBindings: portBinding,
				Binds:        []string{fmt.Sprintf("%s:/hf_cache", hfCache)},
				Resources: container.Resources{
					DeviceRequests: []container.DeviceRequest{dockerDeviceRequest(gpuFlag)},
				},
			},
		})
		if err != nil {
			return err
		}
		closeReservedPort()

		logDockerAPI("container start", created.ID)
		if _, err := cli.ContainerStart(ctx, created.ID, dockerclient.ContainerStartOptions{}); err != nil {
			logDockerAPI("container remove", created.ID, "force=true")
			if _, removeErr := cli.ContainerRemove(context.Background(), created.ID, dockerclient.ContainerRemoveOptions{Force: true}); removeErr != nil {
				return fmt.Errorf("failed to start container %s: %v (cleanup failed: %v)", created.ID, err, removeErr)
			}
			return fmt.Errorf("failed to start container %s: %v", created.ID, err)
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
	containerID = strings.TrimSpace(containerID)
	if containerID == "" {
		return validationError("container id is required")
	}

	return withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		metadata, err := inspectContainer(ctx, cli, containerID)
		if err != nil {
			return fmt.Errorf("failed to inspect container %s: %v", containerID, err)
		}
		if !isManagedContainer(containerLabels(metadata)) {
			return validationError("container %s is not managed by vllm-manager", containerID)
		}
		if metadata.State != nil && (metadata.State.Running || metadata.State.Paused || metadata.State.Restarting) {
			status := string(metadata.State.Status)
			if status == "" {
				status = "active"
			}
			return validationError("container %s is %s; stop it before removing", containerID, status)
		}

		logDockerAPI("container remove", containerID, "force=false")
		if _, err := cli.ContainerRemove(ctx, containerID, dockerclient.ContainerRemoveOptions{}); err != nil {
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
		timeout := int(execTimeoutLong.Seconds())
		logDockerAPI("container restart", containerID)
		if _, err := cli.ContainerRestart(ctx, containerID, dockerclient.ContainerRestartOptions{Timeout: &timeout}); err != nil {
			return fmt.Errorf("failed to restart container %s: %v", containerID, err)
		}
		return nil
	})
}
