package api

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

func GetModelDirectorySize(model, hfCache string) (uint64, error) {
	hubPath := HFHubPath(hfCache)
	if hubPath == "" {
		return 0, fmt.Errorf("HF cache path not configured")
	}

	size, err := calculateDirectorySize(filepath.Join(hubPath, modelCacheDirName(model)))
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	return size, nil
}

func calculateDirectorySize(path string) (uint64, error) {
	var size uint64
	err := filepath.WalkDir(path, func(_ string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Size() > 0 {
			size += uint64(info.Size())
		}
		return nil
	})
	return size, err
}

func GetDownloadedModelSize(hfCache string, model string) (uint64, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return 0, "", validationError("model is required")
	}

	path := modelCacheDir(hfCache, model)
	stat, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, "", validationError("model %s is not downloaded in the configured Hugging Face home", model)
		}
		return 0, "", fmt.Errorf("failed to inspect model cache: %v", err)
	}
	if !stat.IsDir() {
		return 0, "", validationError("model cache path is not a directory: %s", path)
	}

	bytes, err := calculateDirectorySize(path)
	if err != nil {
		return 0, "", fmt.Errorf("failed to calculate model size: %v", err)
	}

	return bytes, FormatBytes(bytes), nil
}

// downloadContainerModelName extracts the model a download container is fetching
// from its labels or `hf download <model>` args, using metadata already captured
// in the shared snapshot (no extra docker inspect).
func downloadContainerModelName(vc managedVLLMContainer) string {
	if model := vc.labels[containerModelLabel]; model != "" {
		return model
	}

	args := vc.metadata.Args
	if len(args) == 0 && vc.metadata.Config != nil {
		args = vc.metadata.Config.Cmd
	}
	for i := 0; i < len(args)-1; i++ {
		if args[i] == "download" {
			return args[i+1]
		}
	}
	return ""
}

// downloadContainersForModel returns every download container (any state) for
// the model in a snapshot taken by the caller. Matching uses the lowercased
// normalized model key so case-different labels are the same download.
func downloadContainersForModel(containers []managedVLLMContainer, model string) []managedVLLMContainer {
	want := NormalizeModelKeyForMatch(model)
	if want == "" {
		return nil
	}
	var result []managedVLLMContainer
	for _, vc := range containers {
		if !strings.HasPrefix(vc.name, "vllm-download-") {
			continue
		}
		if NormalizeModelKeyForMatch(downloadContainerModelName(vc)) == want {
			result = append(result, vc)
		}
	}
	return result
}

func isDownloadRunning(vc managedVLLMContainer) bool {
	return vc.state == "running" || vc.state == "created" || vc.state == "restarting"
}

// downloadContainerForModel returns the name of a still-running download
// container for the model, or "". Finished containers are kept until their
// result is read (see GetDownloadStatus) and do not block anything.
func downloadContainerForModel(containers []managedVLLMContainer, model string) string {
	for _, vc := range downloadContainersForModel(containers, model) {
		if isDownloadRunning(vc) {
			return vc.name
		}
	}
	return ""
}

func ensureModelCacheCanBeDeleted(model string) error {
	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return fmt.Errorf("failed to verify containers before deleting cache: %v", err)
	}

	for _, activeModel := range activeModelsFromContainers(containers) {
		if activeModel.Model == model {
			return validationError(
				"Cannot delete %s because a container still references it; Remove the container first",
				model,
			)
		}
	}

	if containerName := downloadContainerForModel(containers, model); containerName != "" {
		return validationError("cannot delete %s because it is currently being downloaded by %s", model, containerName)
	}

	return nil
}

func DeleteDownloadedModel(model string, hfCache string) error {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return validationError("hugging face cache directory is required")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return validationError("model is required")
	}
	if strings.ContainsRune(model, 0) {
		return validationError("model contains an invalid null byte")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	if err := ensureModelCacheCanBeDeleted(model); err != nil {
		return err
	}

	hubPath := filepath.Clean(HFHubPath(hfCache))
	hubStat, err := os.Stat(hubPath)
	if err != nil {
		if os.IsNotExist(err) {
			return validationError("Hugging Face hub directory does not exist: %s", hubPath)
		}
		return fmt.Errorf("failed to inspect Hugging Face hub directory: %v", err)
	}
	if !hubStat.IsDir() {
		return validationError("Hugging Face hub path is not a directory: %s", hubPath)
	}

	modelDirName := modelCacheDirName(model)
	modelPath := filepath.Clean(filepath.Join(hubPath, modelDirName))
	if filepath.Dir(modelPath) != hubPath {
		return validationError("resolved model cache path escapes Hugging Face hub directory")
	}

	modelStat, err := os.Stat(modelPath)
	if err != nil {
		if os.IsNotExist(err) {
			return validationError("model %s is not downloaded in the configured Hugging Face cache", model)
		}
		return fmt.Errorf("failed to inspect model cache directory: %v", err)
	}
	if !modelStat.IsDir() {
		return validationError("model cache path is not a directory: %s", modelPath)
	}

	if err := withDockerClient(execTimeoutStart, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, "alpine"); err != nil {
			return fmt.Errorf("failed to ensure alpine image: %v", err)
		}

		containerName := fmt.Sprintf("vllm-cache-delete-%s-%d", SanitizeDockerNamePart(modelDirName), time.Now().Unix())
		logDockerAPI("container create", containerName, "image=alpine", "bind="+fmt.Sprintf("%s:/target", hubPath))
		created, err := cli.ContainerCreate(ctx, dockerclient.ContainerCreateOptions{
			Name: containerName,
			Config: &container.Config{
				Image: "alpine",
				Cmd:   []string{"rm", "-rf", "/target/" + modelDirName},
			},
			HostConfig: &container.HostConfig{
				Binds: []string{fmt.Sprintf("%s:/target", hubPath)},
			},
		})
		if err != nil {
			return err
		}
		defer func() {
			logDockerAPI("container remove", created.ID, "force=true")
			_, _ = cli.ContainerRemove(context.Background(), created.ID, dockerclient.ContainerRemoveOptions{Force: true, RemoveVolumes: true})
		}()

		logDockerAPI("container start", created.ID)
		if _, err := cli.ContainerStart(ctx, created.ID, dockerclient.ContainerStartOptions{}); err != nil {
			return err
		}

		waitResult, err := waitForContainerExit(ctx, cli, created.ID)
		if err != nil {
			return err
		}
		if waitResult.StatusCode != 0 {
			return fmt.Errorf("cache delete container exited with status %d: %s", waitResult.StatusCode, dockerContainerLogs(ctx, cli, created.ID))
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to delete downloaded model cache: %v", err)
	}

	return nil
}

func GetDownloadedModels(hfCache string) ([]DownloadedModel, error) {
	hubPath := HFHubPath(hfCache)
	entries, err := os.ReadDir(hubPath)
	if err != nil {
		return nil, nil // Return empty if dir doesn't exist
	}

	var models []DownloadedModel
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelName := strings.ReplaceAll(strings.TrimPrefix(entry.Name(), "models--"), "--", "/")
			size := "unknown"
			var sizeBytes uint64
			if bytes, display, err := GetDownloadedModelSize(hfCache, modelName); err == nil {
				size = display
				sizeBytes = bytes
			}
			kvEstimate, _ := GetModelKVCacheEstimate(hfCache, modelName)

			models = append(models, DownloadedModel{
				Name:                 modelName,
				Size:                 size,
				SizeBytes:            sizeBytes,
				KVCacheBytesPerToken: kvEstimate.fullBytesPerToken(),
				KVCacheLayers:        kvEstimate.Layers,
				KVCacheKeyValueHeads: kvEstimate.KeyValueHeads,
				KVCacheHeadDim:       kvEstimate.HeadDim,
				KVCacheDTypeBytes:    kvEstimate.DTypeBytes,
				DefaultMaxModelLen:   kvEstimate.DefaultMaxModelLen,
				IsMultimodal:         kvEstimate.IsMultimodal,
			})
		}
	}
	return models, nil
}

// IsModelDownloaded reports whether the model is fully present in the cache.
// It returns the bytes on disk even for a partial download, and an error only
// if the cache directory itself is misconfigured. Complete means a snapshot
// has config.json (or Mistral's params.json) and no blob is still
// "*.incomplete", which is what hf download leaves behind when it stops early.
func IsModelDownloaded(model string, hfCache string) (bool, uint64, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return false, 0, nil
	}
	hubPath := HFHubPath(hfCache)
	if hubPath == "" || hubPath == "hub" {
		// HFHubPath returns "hub" (relative) when NormalizeHFHome yielded
		// the empty string. That's a misconfiguration, not a normal miss.
		return false, 0, fmt.Errorf("hugging face cache directory is required")
	}
	dirName := modelCacheDirName(model)
	if dirName == "" || dirName == "models--" {
		return false, 0, nil
	}
	dir := filepath.Join(hubPath, dirName)
	info, err := os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, 0, nil
		}
		return false, 0, err
	}
	if !info.IsDir() {
		return false, 0, nil
	}
	bytes, _, sizeErr := GetDownloadedModelSize(hfCache, model)
	return modelCacheComplete(dir), bytes, sizeErr
}

func modelCacheComplete(dir string) bool {
	if partial, _ := filepath.Glob(filepath.Join(dir, "blobs", "*.incomplete")); len(partial) > 0 {
		return false
	}
	for _, name := range []string{"config.json", "params.json"} {
		// Glob follows the snapshot symlinks, so a dangling link to a
		// missing blob does not count.
		matches, _ := filepath.Glob(filepath.Join(dir, "snapshots", "*", name))
		for _, match := range matches {
			if _, err := os.Stat(match); err == nil {
				return true
			}
		}
	}
	return false
}

// AlreadyDownloadedError means the model is fully cached; delete it first to
// download again.
type AlreadyDownloadedError struct {
	Model string
	Bytes uint64
}

func (e AlreadyDownloadedError) Error() string {
	return fmt.Sprintf("%s is already downloaded (%s on disk); delete it from the cache first to re-download", e.Model, FormatBytes(e.Bytes))
}

func DownloadModel(model string, hfCache string, hfToken string) (string, error) {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return "", validationError("hugging face cache directory is required")
	}
	model = strings.TrimSpace(model)
	if err := ValidateModelID(model); err != nil {
		return "", err
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	// Refuse to re-download a model that is fully present. A partial cache is
	// not "downloaded": hf download resumes it.
	if downloaded, size, err := IsModelDownloaded(model, hfCache); err != nil {
		return "", fmt.Errorf("failed to check existing cache: %v", err)
	} else if downloaded {
		return "", AlreadyDownloadedError{Model: model, Bytes: size}
	}

	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return "", fmt.Errorf("failed to verify download containers before downloading model: %v", err)
	}
	if containerName := downloadContainerForModel(containers, model); containerName != "" {
		return "", validationError("%s is already being downloaded by %s", model, containerName)
	}

	hfInfo := LazyHFInfo(model, hfToken)
	if warning, err := CheckDownloadPreflight(model, hfCache, hfInfo); err != nil {
		return "", err
	} else if warning != "" {
		logDockerMessage("WARNING: %s", warning)
	}
	if gpus, err := GetGPUInfo(); err == nil {
		if warning, err := CheckModelQuantizationCompatibility(model, hfCache, hfInfo, gpus); err != nil {
			return "", err
		} else if warning != "" {
			logDockerMessage("WARNING: %s", warning)
		}
	}

	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	containerName := fmt.Sprintf("vllm-download-%s-%d", SanitizeDockerNamePart(shortName), time.Now().Unix())

	labels := map[string]string{
		"app":                 "vllm-manager",
		managedContainerLabel: "true",
		containerKindLabel:    containerKindDownload,
		containerModelLabel:   model,
	}
	env := []string{
		"HF_HOME=/hf_cache",
		"MODEL_ID=" + model,
		"PYTHONUNBUFFERED=1",
	}

	if hfToken != "" {
		env = append(env, fmt.Sprintf("HF_TOKEN=%s", hfToken))
	}

	// Unknown file list (Hub unreachable): download everything, as before.
	var excludes []string
	if info, err := hfInfo(); err == nil {
		excludes = info.DownloadExcludes
	}

	var containerID string
	err = withDockerClient(execTimeoutStart, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, VLLMImage); err != nil {
			return fmt.Errorf("failed to ensure vLLM image: %v", err)
		}
		// Clear finished download containers for this model so a retry
		// starts clean.
		for _, old := range downloadContainersForModel(containers, model) {
			if !isDownloadRunning(old) {
				logDockerAPI("container remove", old.id, "force=true")
				_, _ = cli.ContainerRemove(ctx, old.id, dockerclient.ContainerRemoveOptions{Force: true})
			}
		}

		// Exclude patterns arrive as positional args ("$@"), never through
		// the shell's word splitting. `hf download` takes one pattern per
		// --exclude and treats any further bare argument as a file name to
		// fetch, so each pattern gets its own flag. The older huggingface-cli
		// takes them all after a single --exclude.
		downloadScript := strings.Join([]string{
			"if command -v hf >/dev/null 2>&1; then",
			"  for pattern in \"$@\"; do set -- \"$@\" --exclude \"$pattern\"; shift; done;",
			"  exec hf download \"$MODEL_ID\" \"$@\";",
			"elif command -v huggingface-cli >/dev/null 2>&1; then",
			"  if [ \"$#\" -gt 0 ]; then set -- --exclude \"$@\"; fi;",
			"  exec huggingface-cli download \"$MODEL_ID\" \"$@\";",
			"else",
			"  echo \"neither hf nor huggingface-cli is available in the vLLM image\" >&2;",
			"  exit 127;",
			"fi",
		}, "\n")

		logDockerAPI(
			"container create",
			containerName,
			"image="+VLLMImage,
			"network=host",
			"entrypoint=sh -lc",
			"bind="+fmt.Sprintf("%s:/hf_cache", hfCache),
			"cmd=download "+model,
			"exclude="+strings.Join(excludes, ","),
		)
		created, err := cli.ContainerCreate(ctx, dockerclient.ContainerCreateOptions{
			Name: containerName,
			Config: &container.Config{
				Image:      VLLMImage,
				Entrypoint: []string{"sh", "-lc"},
				Cmd:        append([]string{downloadScript, "vllm-download"}, excludes...),
				Env:        env,
				Labels:     labels,
			},
			// No AutoRemove: a failed download must keep its exit code and
			// logs until GetDownloadStatus reports them.
			HostConfig: &container.HostConfig{
				NetworkMode: container.NetworkMode("host"),
				Binds:       []string{fmt.Sprintf("%s:/hf_cache", hfCache)},
			},
		})
		if err != nil {
			return err
		}

		logDockerAPI("container start", created.ID)
		if _, err := cli.ContainerStart(ctx, created.ID, dockerclient.ContainerStartOptions{}); err != nil {
			return err
		}
		containerID = created.ID
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("failed to start download container: %v", err)
	}
	return containerID, nil
}

// DownloadStatus is what a progress poll returns for one model.
type DownloadStatus struct {
	// State is "running", "succeeded", "failed" or "stopped" (no download
	// container and the cache is not complete).
	State           string `json:"state"`
	DownloadedBytes uint64 `json:"downloaded_bytes"`
	ExitCode        int    `json:"exit_code,omitempty"`
	Error           string `json:"error,omitempty"`
}

const downloadErrorTailBytes = 2000

// GetDownloadStatus reports the model's download. A finished download
// container is removed once its result has been read.
func GetDownloadStatus(model string, hfCache string) (DownloadStatus, error) {
	model = strings.TrimSpace(model)
	if err := ValidateModelID(model); err != nil {
		return DownloadStatus{}, err
	}
	size, err := GetModelDirectorySize(model, hfCache)
	if err != nil {
		return DownloadStatus{}, err
	}
	status := DownloadStatus{DownloadedBytes: size}
	// Log tail of a download container that exited 0; used if the cache is
	// still incomplete, which means the downloader fetched nothing useful.
	cleanExitLogs, sawCleanExit := "", false

	err = withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		containers, err := listManagedVLLMContainers(ctx, cli)
		if err != nil {
			return err
		}
		matches := downloadContainersForModel(containers, model)
		for _, vc := range matches {
			if isDownloadRunning(vc) {
				status.State = "running"
				return nil
			}
		}
		for _, vc := range matches {
			exitCode := 0
			if vc.metadata.State != nil {
				exitCode = vc.metadata.State.ExitCode
			}
			if exitCode != 0 && status.State != "failed" {
				status.State = "failed"
				status.ExitCode = exitCode
				status.Error = downloadLogTail(ctx, cli, vc.id)
			} else if exitCode == 0 && !sawCleanExit {
				sawCleanExit = true
				cleanExitLogs = downloadLogTail(ctx, cli, vc.id)
			}
			logDockerAPI("container remove", vc.id, "force=true")
			_, _ = cli.ContainerRemove(ctx, vc.id, dockerclient.ContainerRemoveOptions{Force: true})
		}
		return nil
	})
	if err != nil {
		return DownloadStatus{}, fmt.Errorf("failed to read download container state: %v", err)
	}
	if status.State != "" {
		return status, nil
	}
	complete, _, _ := IsModelDownloaded(model, hfCache)
	switch {
	case complete:
		status.State = "succeeded"
	case sawCleanExit:
		status.State = "failed"
		status.Error = strings.TrimSpace("the downloader exited without fetching the model files\n" + cleanExitLogs)
	default:
		status.State = "stopped"
	}
	return status, nil
}

func downloadLogTail(ctx context.Context, cli *dockerclient.Client, containerID string) string {
	logs := dockerContainerLogs(ctx, cli, containerID)
	if len(logs) > downloadErrorTailBytes {
		logs = logs[len(logs)-downloadErrorTailBytes:]
	}
	return logs
}
