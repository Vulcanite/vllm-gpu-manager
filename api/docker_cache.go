package api

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

type modelSizeInfo struct {
	Display  string
	Bytes    uint64
	CachedAt time.Time
}

var (
	modelSizeCache   = make(map[string]modelSizeInfo)
	modelSizeCacheMu sync.RWMutex
)

const modelSizeCacheTTL = 2 * time.Second

func GetModelDirectorySize(model, hfCache string) (uint64, error) {
	hubPath := HFHubPath(hfCache)
	if hubPath == "" {
		return 0, fmt.Errorf("HF cache path not configured")
	}

	modelPath := filepath.Join(hubPath, modelCacheDirName(model))

	var totalSize int64
	err := filepath.Walk(modelPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if !info.IsDir() {
			totalSize += info.Size()
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return uint64(totalSize), nil
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
	cacheKey := filepath.Clean(path)

	modelSizeCacheMu.RLock()
	cached, ok := modelSizeCache[cacheKey]
	modelSizeCacheMu.RUnlock()
	if ok && time.Since(cached.CachedAt) < modelSizeCacheTTL {
		return cached.Bytes, cached.Display, nil
	}

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

	display := FormatBytes(bytes)
	modelSizeCacheMu.Lock()
	modelSizeCache[cacheKey] = modelSizeInfo{
		Display:  display,
		Bytes:    bytes,
		CachedAt: time.Now(),
	}
	modelSizeCacheMu.Unlock()
	return bytes, display, nil
}

func clearDownloadedModelSizeCache(path string) {
	cacheKey := filepath.Clean(path)
	modelSizeCacheMu.Lock()
	delete(modelSizeCache, cacheKey)
	modelSizeCacheMu.Unlock()
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

// downloadContainerForModel returns the name of an in-progress download
// container for the given model, or "" if none, scanning a snapshot taken by
// the caller. The comparison is done on the lowercased normalized model key
// so case-different labels are treated as the same download.
func downloadContainerForModel(containers []managedVLLMContainer, model string) string {
	want := NormalizeModelKeyForMatch(model)
	if want == "" {
		return ""
	}
	for _, vc := range containers {
		if !strings.HasPrefix(vc.name, "vllm-download-") {
			continue
		}
		if NormalizeModelKeyForMatch(downloadContainerModelName(vc)) == want {
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

		containerName := fmt.Sprintf("vllm-cache-delete-%s-%d", sanitizeDockerNamePart(modelDirName), time.Now().Unix())
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

	clearDownloadedModelSizeCache(modelPath)
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
			})
		}
	}
	return models, nil
}

// hubModelDirName converts a user-visible model identifier
// ("Qwen/Qwen2.5-7B-Instruct") into the directory name used by the
// Hugging Face hub cache ("models--Qwen--Qwen2.5-7B-Instruct"). It is the
// inverse of the conversion done in GetDownloadedModels.
func hubModelDirName(model string) string {
	model = strings.TrimSpace(model)
	if model == "" {
		return ""
	}
	return "models--" + strings.ReplaceAll(model, "/", "--")
}

// IsModelDownloaded reports whether a model identifier is already present in
// the configured Hugging Face cache. It returns (true, on-disk size in bytes,
// nil) on a hit, (false, 0, nil) on a miss, and an error only if the cache
// directory itself is misconfigured (not if the model is merely absent).
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
	dirName := hubModelDirName(model)
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
	// Reuse the existing size calculator so the on-disk bytes include all
	// the snapshots/weights/refs the cache may have.
	bytes, _, sizeErr := GetDownloadedModelSize(hfCache, model)
	return true, bytes, sizeErr
}

func DownloadModel(model string, hfCache string, hfToken string) (string, error) {
	hfCache = NormalizeHFHome(hfCache)
	if hfCache == "" {
		return "", validationError("hugging face cache directory is required")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return "", validationError("model is required")
	}

	lock := getModelOperationLock(hfCache, model)
	lock.Lock()
	defer lock.Unlock()

	// Refuse to re-download a model that is already fully present on disk.
	// Re-running the downloader would still be a no-op (HF skips existing
	// files) but it spins up a Docker container and burns time, and the
	// user almost certainly hit "Download" by accident. Callers that
	// genuinely need to force a fresh download should remove the cache
	// directory first.
	if downloaded, _, err := IsModelDownloaded(model, hfCache); err != nil {
		return "", fmt.Errorf("failed to check existing cache: %v", err)
	} else if downloaded {
		return "", validationError("%s is already downloaded; remove it first if you want to re-download", model)
	}

	containers, err := snapshotManagedVLLMContainers()
	if err != nil {
		return "", fmt.Errorf("failed to verify download containers before downloading model: %v", err)
	}
	if containerName := downloadContainerForModel(containers, model); containerName != "" {
		return "", validationError("%s is already being downloaded by %s", model, containerName)
	}

	if warning, err := CheckDownloadPreflight(model, hfCache, hfToken); err != nil {
		return "", err
	} else if warning != "" {
		logDockerMessage("WARNING: %s", warning)
	}
	if gpus, err := GetGPUInfo(); err == nil {
		if warning, err := CheckModelQuantizationCompatibility(model, hfCache, hfToken, gpus); err != nil {
			return "", err
		} else if warning != "" {
			logDockerMessage("WARNING: %s", warning)
		}
	}

	shortName := model
	if idx := strings.LastIndex(model, "/"); idx != -1 {
		shortName = model[idx+1:]
	}
	containerName := fmt.Sprintf("vllm-download-%s-%d", sanitizeDockerNamePart(shortName), time.Now().Unix())

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

	var containerID string
	err = withDockerClient(execTimeoutStart, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, VLLMImage); err != nil {
			return fmt.Errorf("failed to ensure vLLM image: %v", err)
		}

		downloadScript := strings.Join([]string{
			"if command -v hf >/dev/null 2>&1; then",
			"  exec hf download \"$MODEL_ID\";",
			"elif command -v huggingface-cli >/dev/null 2>&1; then",
			"  exec huggingface-cli download \"$MODEL_ID\";",
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
			"autoremove=true",
			"entrypoint=sh -lc",
			"bind="+fmt.Sprintf("%s:/hf_cache", hfCache),
			"cmd=download "+model,
		)
		created, err := cli.ContainerCreate(ctx, dockerclient.ContainerCreateOptions{
			Name: containerName,
			Config: &container.Config{
				Image:      VLLMImage,
				Entrypoint: []string{"sh", "-lc"},
				Cmd:        []string{downloadScript},
				Env:        env,
				Labels:     labels,
			},
			HostConfig: &container.HostConfig{
				AutoRemove:  true,
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
