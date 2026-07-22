package api

import (
	"context"
	"fmt"
	"time"

	"github.com/moby/moby/api/types/container"
	dockerclient "github.com/moby/moby/client"
)

const gpuProbeImage = "alpine:latest"

func CheckDockerAPI() error {
	if err := withDockerClient(execTimeoutShort, func(context.Context, *dockerclient.Client) error { return nil }); err != nil {
		return fmt.Errorf("Docker API ping failed: %v", err)
	}
	return nil
}

func CheckDockerImageLocal(image string) error {
	return withDockerClient(execTimeoutShort, func(ctx context.Context, cli *dockerclient.Client) error {
		_, err := cli.ImageInspect(ctx, image)
		return err
	})
}

// CheckDockerGPUAccess verifies that Docker can satisfy an NVIDIA GPU device
// request. Alpine is intentionally used because the probe only needs the
// container runtime to inject GPU devices; it does not need CUDA userspace.
func CheckDockerGPUAccess() error {
	return withDockerClient(execTimeoutStart, func(ctx context.Context, cli *dockerclient.Client) error {
		if err := ensureDockerImage(ctx, cli, gpuProbeImage); err != nil {
			return fmt.Errorf("failed to ensure GPU probe image %s: %v", gpuProbeImage, err)
		}

		name := fmt.Sprintf("vllm-manager-gpu-probe-%d", time.Now().UnixNano())
		logDockerAPI("container create", name, "image="+gpuProbeImage, "gpu=all")
		created, err := cli.ContainerCreate(ctx, gpuProbeCreateOptions(name))
		if err != nil {
			return fmt.Errorf("failed to create Docker GPU probe: %v", err)
		}
		defer func() {
			removeCtx, cancel := context.WithTimeout(context.Background(), execTimeoutShort)
			defer cancel()
			logDockerAPI("container remove", created.ID, "force=true")
			_, _ = cli.ContainerRemove(removeCtx, created.ID, dockerclient.ContainerRemoveOptions{Force: true, RemoveVolumes: true})
		}()

		logDockerAPI("container start", created.ID)
		if _, err := cli.ContainerStart(ctx, created.ID, dockerclient.ContainerStartOptions{}); err != nil {
			return fmt.Errorf("Docker could not provision NVIDIA GPUs: %v", err)
		}
		result, err := waitForContainerExit(ctx, cli, created.ID)
		if err != nil {
			return fmt.Errorf("failed waiting for Docker GPU probe: %v", err)
		}
		if result.StatusCode != 0 {
			return fmt.Errorf("Docker GPU probe exited with status %d: %s", result.StatusCode, dockerContainerLogs(ctx, cli, created.ID))
		}
		return nil
	})
}

func gpuProbeCreateOptions(name string) dockerclient.ContainerCreateOptions {
	return dockerclient.ContainerCreateOptions{
		Name: name,
		Config: &container.Config{
			Image: gpuProbeImage,
			Cmd:   []string{"true"},
		},
		HostConfig: &container.HostConfig{
			Resources: container.Resources{
				DeviceRequests: []container.DeviceRequest{dockerDeviceRequest("all")},
			},
		},
	}
}
