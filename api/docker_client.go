package api

import (
	"bytes"
	"context"
	"io"
	"log"
	"os"
	"strings"
	"time"

	cerrdefs "github.com/containerd/errdefs"
	"github.com/moby/moby/api/pkg/stdcopy"
	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

const (
	execTimeoutShort = 10 * time.Second // ps, inspect, rm, image inspect
	execTimeoutLong  = 30 * time.Second // stop
)

func dockerClient(ctx context.Context) (*dockerclient.Client, error) {
	cli, err := dockerclient.New(dockerclient.FromEnv)
	if err != nil {
		return nil, err
	}
	_, err = cli.Ping(ctx, dockerclient.PingOptions{})
	return cli, err
}

func withDockerClient(timeout time.Duration, fn func(context.Context, *dockerclient.Client) error) error {
	ctx := context.Background()
	cancel := func() {}
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), timeout)
	}
	defer cancel()

	cli, err := dockerClient(ctx)
	if err != nil {
		return err
	}
	defer cli.Close()

	return fn(ctx, cli)
}

func dockerContainerName(summary container.Summary) string {
	if len(summary.Names) == 0 {
		return ""
	}
	return strings.TrimPrefix(summary.Names[0], "/")
}

func listContainers(ctx context.Context, cli *dockerclient.Client) ([]container.Summary, error) {
	logDockerAPI("container list", "all=true")
	opts := dockerclient.ContainerListOptions{All: true}
	result, err := cli.ContainerList(ctx, opts)
	if err != nil {
		return nil, err
	}
	return result.Items, nil
}

func inspectContainer(ctx context.Context, cli *dockerclient.Client, containerID string) (container.InspectResponse, error) {
	logDockerAPI("container inspect", containerID)
	result, err := cli.ContainerInspect(ctx, containerID, dockerclient.ContainerInspectOptions{})
	if err != nil {
		return container.InspectResponse{}, err
	}
	return result.Container, nil
}

func ensureDockerImage(ctx context.Context, cli *dockerclient.Client, imageName string) error {
	logDockerAPI("image inspect", imageName)
	if _, err := cli.ImageInspect(ctx, imageName); err == nil {
		return nil
	} else if !cerrdefs.IsNotFound(err) {
		return err
	}

	logDockerMessage("Image %s not found locally; pulling through Docker API", imageName)
	logDockerAPI("image pull", imageName)
	pull, err := cli.ImagePull(ctx, imageName, dockerclient.ImagePullOptions{})
	if err != nil {
		return err
	}
	defer pull.Close()
	_, err = io.Copy(io.Discard, pull)
	return err
}

func dockerContainerLogs(ctx context.Context, cli *dockerclient.Client, containerID string) string {
	logs, err := cli.ContainerLogs(ctx, containerID, dockerclient.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Tail:       "all",
	})
	if err != nil {
		return ""
	}
	defer logs.Close()

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	if _, err := stdcopy.StdCopy(&stdout, &stderr, logs); err != nil {
		return ""
	}
	return strings.TrimSpace(stdout.String() + stderr.String())
}

func waitForContainerExit(ctx context.Context, cli *dockerclient.Client, containerID string) (container.WaitResponse, error) {
	wait := cli.ContainerWait(ctx, containerID, dockerclient.ContainerWaitOptions{
		Condition: container.WaitConditionNotRunning,
	})
	select {
	case result := <-wait.Result:
		return result, nil
	case err := <-wait.Error:
		return container.WaitResponse{}, err
	case <-ctx.Done():
		return container.WaitResponse{}, ctx.Err()
	}
}

func dockerDeviceRequest(gpuFlag string) container.DeviceRequest {
	capabilities := [][]string{{"gpu"}}
	if devices, ok := strings.CutPrefix(gpuFlag, "device="); ok {
		var deviceIDs []string
		for _, id := range strings.Split(devices, ",") {
			id = strings.TrimSpace(id)
			if id != "" {
				deviceIDs = append(deviceIDs, id)
			}
		}
		if len(deviceIDs) > 0 {
			return container.DeviceRequest{
				Driver:       "nvidia",
				DeviceIDs:    deviceIDs,
				Capabilities: capabilities,
			}
		}
	}

	return container.DeviceRequest{
		Driver:       "nvidia",
		Count:        -1,
		Capabilities: capabilities,
	}
}

func logDockerAPI(operation string, args ...string) {
	if len(args) == 0 {
		logDockerMessage("API: %s", operation)
		return
	}
	logDockerMessage("API: %s %s", operation, redactDockerArgs(args))
}

func logDockerMessage(format string, args ...interface{}) {
	if !dockerAPILoggingEnabled() {
		return
	}
	log.Printf("[DOCKER] "+format, args...)
}

func dockerAPILoggingEnabled() bool {
	value := strings.TrimSpace(os.Getenv("VLLM_MANAGER_DOCKER_LOG"))
	return value == "1" || strings.EqualFold(value, "true")
}

func redactDockerArgs(args []string) string {
	redacted := append([]string(nil), args...)
	for i := range redacted {
		redacted[i] = redactDockerSecret(redacted[i])
		if redacted[i] == "-e" && i+1 < len(redacted) && strings.HasPrefix(redacted[i+1], "HF_TOKEN=") {
			redacted[i+1] = "HF_TOKEN=<redacted>"
		}
		if redacted[i] == "--api-key" && i+1 < len(redacted) {
			redacted[i+1] = "<redacted>"
		}
	}
	return strings.Join(redacted, " ")
}

func redactDockerSecret(value string) string {
	if strings.HasPrefix(value, "HF_TOKEN=") {
		return "HF_TOKEN=<redacted>"
	}
	return redactFlagValue(value, "--api-key")
}

func redactFlagValue(value string, flag string) string {
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return value
	}
	changed := false
	for i := range fields {
		prefix := ""
		token := fields[i]
		if trimmed, ok := strings.CutPrefix(token, "cmd="); ok {
			prefix = "cmd="
			token = trimmed
		}
		if token == flag && i+1 < len(fields) {
			fields[i+1] = "<redacted>"
			changed = true
			continue
		}
		if strings.HasPrefix(token, flag+"=") {
			fields[i] = prefix + flag + "=<redacted>"
			changed = true
		}
	}
	if !changed {
		return value
	}
	return strings.Join(fields, " ")
}

// checkVLLMImageExists verifies the configured vLLM image is present locally.
// It warns but does not hard-fail — the Docker API launch path pulls the image
// if the image is absent and network is available.
func checkVLLMImageExists() {
	_ = withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		logDockerAPI("image inspect", vllmImage)
		if _, err := cli.ImageInspect(ctx, vllmImage); err != nil {
			logDockerMessage("WARNING: image %s not found locally — Docker API will attempt to pull it, which may take several minutes", vllmImage)
		}
		return nil
	})
}
