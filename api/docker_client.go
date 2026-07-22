package api

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
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
	execTimeoutStart = 5 * time.Minute  // image pull + container create + start; bound for the model lock
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

// withDockerClientCtx is like withDockerClient but uses an external context
// instead of creating one. This is needed for streaming where the context
// must be tied to the HTTP request so client disconnects cancel the stream.
func withDockerClientCtx(ctx context.Context, fn func(context.Context, *dockerclient.Client) error) error {
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

// GetContainerLogs fetches logs for a container. For running containers it
// returns the last 200 lines (like tail -200). For stopped/exited containers
// it returns all logs since there won't be any more.
func GetContainerLogs(containerID string) (string, error) {
	var result string
	err := withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		// Check if the container is still running.
		info, err := cli.ContainerInspect(ctx, containerID, dockerclient.ContainerInspectOptions{})
		if err != nil {
			return err
		}

		tail := "all"
		if info.Container.State != nil && info.Container.State.Running {
			tail = "200"
		}

		logReader, err := cli.ContainerLogs(ctx, containerID, dockerclient.ContainerLogsOptions{
			ShowStdout: true,
			ShowStderr: true,
			Tail:       tail,
		})
		if err != nil {
			return err
		}
		defer logReader.Close()

		var stdout bytes.Buffer
		var stderr bytes.Buffer
		if _, err := stdcopy.StdCopy(&stdout, &stderr, logReader); err != nil {
			return err
		}
		result = strings.TrimSpace(stdout.String() + stderr.String())
		return nil
	})
	return result, err
}

// LogChunk represents a chunk of container log output.
type LogChunk struct {
	Stdout bool
	Data   string
	Err    error
}

// StreamContainerLogs streams container logs in real-time, sending each
// chunk on the returned channel. The stream follows the container output
// until the context is cancelled (e.g. client disconnect). Callers should
// range over the channel and handle LogChunk.Err to detect errors.
func StreamContainerLogs(ctx context.Context, containerID string) <-chan LogChunk {
	ch := make(chan LogChunk, 64)

	go func() {
		defer close(ch)

		err := withDockerClientCtx(ctx, func(ctx context.Context, cli *dockerclient.Client) error {
			logs, err := cli.ContainerLogs(ctx, containerID, dockerclient.ContainerLogsOptions{
				ShowStdout: true,
				ShowStderr: true,
				Follow:     true,
				Tail:       "0",
			})
			if err != nil {
				return err
			}
			defer logs.Close()

			return streamDockerLogs(ctx, logs, ch)
		})

		if err != nil && !errors.Is(err, context.Canceled) {
			select {
			case ch <- LogChunk{Err: err}:
			default:
			}
		}
	}()

	return ch
}

// streamDockerLogs reads Docker's multiplexed log stream frame-by-frame
// and sends each chunk on the provided channel. It parses the 8-byte
// frame headers directly so it can return data incrementally instead of
// blocking until the stream closes (which stdcopy.StdCopy does).
//
// Docker multiplexed log frame (see moby/api/pkg/stdcopy):
//
//	[0]   = stream type (1=stdout, 2=stderr)
//	[1:4] = reserved (zeros)
//	[4:8] = big-endian uint32 payload size
//	[8:]  = payload bytes
func streamDockerLogs(ctx context.Context, reader io.Reader, ch chan<- LogChunk) error {
	header := make([]byte, 8)
	buf := make([]byte, 1024*1024) // 1MB max frame

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Read the 8-byte header.
		n, err := io.ReadFull(reader, header)
		if err != nil {
			if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			return err
		}
		if n < 8 {
			return nil
		}

		// Parse payload size from header bytes 4-7 (big-endian uint32).
		payloadSize := int(binary.BigEndian.Uint32(header[4:8]))
		if payloadSize <= 0 || payloadSize > len(buf) {
			// Discard the oversized payload to avoid desyncing the frame
			// boundary. Without this, the next io.ReadFull would interpret
			// payload bytes as a header and corrupt the stream.
			if payloadSize > 0 {
				if _, err := io.CopyN(io.Discard, reader, int64(payloadSize)); err != nil {
					return err
				}
			}
			continue
		}

		// Read the payload.
		n, err = io.ReadFull(reader, buf[:payloadSize])
		if err != nil {
			if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			return err
		}

		// Determine stream type from header byte 0.
		isStdout := header[0] == 1

		chunk := string(buf[:n])
		if chunk == "" {
			continue
		}

		select {
		case ch <- LogChunk{Stdout: isStdout, Data: chunk}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
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
		logDockerAPI("image inspect", VLLMImage)
		if _, err := cli.ImageInspect(ctx, VLLMImage); err != nil {
			logDockerMessage("WARNING: image %s not found locally — Docker API will attempt to pull it, which may take several minutes", VLLMImage)
		}
		return nil
	})
}
