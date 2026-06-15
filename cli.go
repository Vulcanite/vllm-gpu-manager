package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/spf13/cobra"
	"golang.org/x/term"

	"vllm-manager/api"
)

type cliOptions struct {
	JSON bool
}

func runCLI(args []string) error {
	cmd := newCLICommand(os.Stdout, os.Stderr)
	cmd.SetArgs(args)
	return cmd.Execute()
}

func printRootHelp(w io.Writer) {
	_ = newCLICommand(w, w).Help()
}

func newCLICommand(out io.Writer, errOut io.Writer) *cobra.Command {
	opts := &cliOptions{}
	cmd := &cobra.Command{
		Use:           "vllm-manager",
		Short:         "Manage local vLLM serving containers",
		SilenceErrors: true,
		SilenceUsage:  true,
		RunE:          showCommandHelp,
	}
	cmd.SetOut(out)
	cmd.SetErr(errOut)
	cmd.Flags().Bool("dashboard", false, "run the web dashboard")
	cmd.Flags().String("port", defaultDashboardPort, "dashboard port")
	cmd.Flags().Bool("proxy", false, "enable the /v1/ proxy gateway")
	cmd.PersistentFlags().BoolVar(&opts.JSON, "json", false, "print JSON output")
	cmd.AddCommand(
		newGPUCommand(opts),
		newModelCommand(opts),
		newCacheCommand(opts),
		newSettingsCommand(opts),
	)
	cmd.Example = strings.Join([]string{
		"  vllm-manager gpu list",
		"  vllm-manager model active",
		"  vllm-manager model start Qwen/Qwen2.5-7B-Instruct --port 8080 --gpus 1",
		"  vllm-manager settings get",
		"  vllm-manager --dashboard --port 9090",
		"  vllm-manager --dashboard --port 9090 --proxy",
	}, "\n")
	return cmd
}

func showCommandHelp(cmd *cobra.Command, _ []string) error {
	return cmd.Help()
}

func newGPUCommand(opts *cliOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "gpu",
		Short: "Inspect NVIDIA GPUs",
		RunE:  showCommandHelp,
	}
	cmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List detected GPUs",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, args []string) error {
			return cliGPUList(*opts)
		},
	})
	return cmd
}

func newModelCommand(opts *cliOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model",
		Short: "Manage vLLM containers and Hugging Face models",
		RunE:  showCommandHelp,
	}
	cmd.AddCommand(
		&cobra.Command{
			Use:   "active",
			Short: "Show active vLLM serving containers",
			Args:  cobra.NoArgs,
			RunE: func(_ *cobra.Command, args []string) error {
				return cliModelActive(args, *opts)
			},
		},
		&cobra.Command{
			Use:   "cached",
			Short: "Show downloaded models in the Hugging Face cache",
			Args:  cobra.NoArgs,
			RunE: func(_ *cobra.Command, args []string) error {
				return cliModelCached(args, *opts)
			},
		},
		&cobra.Command{
			Use:   "info <model>",
			Short: "Fetch Hugging Face model metadata",
			Args:  cobra.ExactArgs(1),
			RunE: func(_ *cobra.Command, args []string) error {
				return cliModelInfo(args, *opts)
			},
		},
		&cobra.Command{
			Use:   "download <model>",
			Short: "Download a model into the configured cache",
			Args:  cobra.ExactArgs(1),
			RunE: func(_ *cobra.Command, args []string) error {
				return cliModelDownload(args, *opts)
			},
		},
		&cobra.Command{
			Use:   "progress <model>",
			Short: "Show cached bytes for a model download",
			Args:  cobra.ExactArgs(1),
			RunE: func(_ *cobra.Command, args []string) error {
				return cliModelProgress(args, *opts)
			},
		},
		newModelRecommendCommand(opts),
		newModelStartCommand(opts),
		newModelStopCommand(opts),
		newModelRestartCommand(opts),
		newModelRemoveCommand(opts),
	)
	return cmd
}

func newModelRecommendCommand(opts *cliOptions) *cobra.Command {
	var useCase string
	var fitOnly bool
	var limit int
	cmd := &cobra.Command{
		Use:   "recommend",
		Short: "Recommend models for the detected GPUs",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			return cliModelRecommend(*opts, useCase, fitOnly, limit)
		},
	}
	cmd.Flags().StringVar(&useCase, "use-case", "general", "recommendation use case: general, coding, reasoning, chat, vision")
	cmd.Flags().BoolVar(&fitOnly, "fit-only", false, "hide models that do not fit")
	cmd.Flags().IntVar(&limit, "limit", 20, "maximum recommendations to print")
	return cmd
}

func newModelStartCommand(opts *cliOptions) *cobra.Command {
	var port int
	var gpus int
	var gpuIDs string
	var memUtil string
	var maxLen string
	var maxNumSeqs string
	var kvCacheDType string
	var toolParser string
	cmd := &cobra.Command{
		Use:   "start <model>",
		Short: "Start a vLLM OpenAI-compatible server",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			if port == 0 {
				return errors.New("model start requires --port")
			}
			if gpus == 0 {
				return errors.New("model start requires --gpus")
			}
			settings := getSettings()
			if err := api.StartModel(args[0], port, gpus, gpuIDs, memUtil, maxLen, maxNumSeqs, kvCacheDType, toolParser, settings.HFCache, settings.HFToken, settings.VLLMAPIKey, settings.EnableCUDACompatibility, settings.HFHubOffline, "0.0.0.0"); err != nil {
				return err
			}
			upsertRoute(args[0], port)
			return printMutationResult(*opts, map[string]interface{}{
				"status":         "started",
				"model":          args[0],
				"port":           port,
				"gpus":           gpus,
				"gpu_ids":        emptyAsDefault(gpuIDs, "auto"),
				"mem_util":       memUtil,
				"max_len":        emptyAsDefault(maxLen, "default"),
				"max_num_seqs":   emptyAsDefault(maxNumSeqs, "default"),
				"kv_cache_dtype": kvCacheDType,
			})
		},
	}
	cmd.Flags().IntVar(&port, "port", 0, "port for the vLLM OpenAI server")
	cmd.Flags().IntVar(&gpus, "gpus", 0, "tensor parallel GPU count")
	cmd.Flags().StringVar(&gpuIDs, "gpu-ids", "", "comma-separated GPU IDs")
	cmd.Flags().StringVar(&memUtil, "mem-util", "0.9", "vLLM GPU memory utilization")
	cmd.Flags().StringVar(&maxLen, "max-len", "", "vLLM max model length")
	cmd.Flags().StringVar(&maxNumSeqs, "max-num-seqs", "", "vLLM max number of sequences")
	cmd.Flags().StringVar(&kvCacheDType, "kv-cache-dtype", "auto", "vLLM KV cache dtype: auto, fp8, fp8_e5m2, fp8_e4m3")
	cmd.Flags().StringVar(&toolParser, "tool-parser", "", "vLLM tool call parser")
	return cmd
}

func newModelStopCommand(opts *cliOptions) *cobra.Command {
	var port int
	cmd := &cobra.Command{
		Use:   "stop",
		Short: "Stop the managed vLLM server on a port",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			if port == 0 {
				return errors.New("model stop requires --port")
			}
			if err := api.StopModel(port); err != nil {
				return err
			}
			return printMutationResult(*opts, map[string]interface{}{
				"status": "stopped",
				"port":   port,
			})
		},
	}
	cmd.Flags().IntVar(&port, "port", 0, "port for the vLLM server")
	return cmd
}

func newModelRestartCommand(opts *cliOptions) *cobra.Command {
	var containerID string
	cmd := &cobra.Command{
		Use:   "restart",
		Short: "Restart a managed container",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			if strings.TrimSpace(containerID) == "" {
				return errors.New("model restart requires --container-id")
			}
			if err := api.RestartModel(containerID); err != nil {
				return err
			}
			return printMutationResult(*opts, map[string]interface{}{
				"status":       "restarted",
				"container_id": containerID,
			})
		},
	}
	cmd.Flags().StringVar(&containerID, "container-id", "", "managed container ID")
	return cmd
}

func newModelRemoveCommand(opts *cliOptions) *cobra.Command {
	var containerID string
	var yes bool
	cmd := &cobra.Command{
		Use:   "remove",
		Short: "Remove a managed container",
		Args:  cobra.NoArgs,
		RunE: func(_ *cobra.Command, _ []string) error {
			if strings.TrimSpace(containerID) == "" {
				return errors.New("model remove requires --container-id")
			}
			if err := confirmDestructive(yes, fmt.Sprintf("Remove managed container %s?", containerID)); err != nil {
				return err
			}
			if err := api.RemoveModel(containerID); err != nil {
				return err
			}
			return printMutationResult(*opts, map[string]interface{}{
				"status":       "removed",
				"container_id": containerID,
			})
		},
	}
	cmd.Flags().StringVar(&containerID, "container-id", "", "managed container ID")
	cmd.Flags().BoolVar(&yes, "yes", false, "skip confirmation prompt")
	return cmd
}

func newCacheCommand(opts *cliOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cache",
		Short: "Manage Hugging Face cache contents",
		RunE:  showCommandHelp,
	}
	cmd.AddCommand(newCacheDeleteCommand(opts))
	return cmd
}

func newCacheDeleteCommand(opts *cliOptions) *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "delete <model>",
		Short: "Delete a cached model",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			if err := confirmDestructive(yes, fmt.Sprintf("Delete cached model %s?", args[0])); err != nil {
				return err
			}
			if err := api.DeleteDownloadedModel(args[0], getSettings().HFCache); err != nil {
				return err
			}
			return printMutationResult(*opts, map[string]interface{}{
				"status": "deleted",
				"model":  args[0],
			})
		},
	}
	cmd.Flags().BoolVar(&yes, "yes", false, "skip confirmation prompt")
	return cmd
}

func newSettingsCommand(opts *cliOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "settings",
		Short: "View and update local settings",
		RunE:  showCommandHelp,
	}
	cmd.AddCommand(
		&cobra.Command{
			Use:   "get",
			Short: "Show local settings and system resources",
			Args:  cobra.NoArgs,
			RunE: func(_ *cobra.Command, args []string) error {
				return cliSettingsGet(args, *opts)
			},
		},
		newSettingsSetCommand(opts),
	)
	return cmd
}

func newSettingsSetCommand(opts *cliOptions) *cobra.Command {
	var hfCache string
	var hfToken string
	var clearHFToken bool
	var vllmAPIKey string
	var clearVLLMAPIKey bool
	var enableCUDACompatibility bool
	var hfHubOffline bool
	cmd := &cobra.Command{
		Use:   "set",
		Short: "Update local settings",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if strings.TrimSpace(hfCache) == "" {
				return errors.New("settings set requires --hf-cache")
			}
			if clearHFToken && hfToken != "" {
				return errors.New("settings set cannot use --hf-token and --clear-hf-token together")
			}
			if clearVLLMAPIKey && vllmAPIKey != "" {
				return errors.New("settings set cannot use --vllm-api-key and --clear-vllm-api-key together")
			}
			settings, err := updateSettings(SettingsUpdate{
				HFCache:                 hfCache,
				HFToken:                 hfToken,
				ClearHFToken:            clearHFToken,
				VLLMAPIKey:              vllmAPIKey,
				ClearVLLMAPIKey:         clearVLLMAPIKey,
				EnableCUDACompatibility: enableCUDACompatibility,
				SetCUDACompatibility:    cmd.Flags().Changed("vllm-cuda-compatibility"),
				HFHubOffline:            hfHubOffline,
				SetHFHubOffline:         cmd.Flags().Changed("hf-hub-offline"),
			})
			if err != nil {
				return err
			}
			return printMutationResult(*opts, map[string]interface{}{
				"status":                    "saved",
				"hf_cache":                  settings.HFCache,
				"hf_token_set":              settings.HFToken != "",
				"vllm_api_key_set":          settings.VLLMAPIKey != "",
				"enable_cuda_compatibility": settings.EnableCUDACompatibility,
				"hf_hub_offline":            settings.HFHubOffline,
			})
		},
	}
	cmd.Flags().StringVar(&hfCache, "hf-cache", "", "Hugging Face home directory")
	cmd.Flags().StringVar(&hfToken, "hf-token", "", "Hugging Face token")
	cmd.Flags().BoolVar(&clearHFToken, "clear-hf-token", false, "clear stored Hugging Face token")
	cmd.Flags().StringVar(&vllmAPIKey, "vllm-api-key", "", "vLLM OpenAI-compatible API key")
	cmd.Flags().BoolVar(&clearVLLMAPIKey, "clear-vllm-api-key", false, "clear stored vLLM API key")
	cmd.Flags().BoolVar(&enableCUDACompatibility, "vllm-cuda-compatibility", false, "set VLLM_ENABLE_CUDA_COMPATIBILITY=1 on new serving containers")
	cmd.Flags().BoolVar(&hfHubOffline, "hf-hub-offline", false, "set HF_HUB_OFFLINE=1 on new serving containers")
	return cmd
}

func cliGPUList(opts cliOptions) error {
	gpus, err := api.GetGPUInfo()
	if err != nil {
		return err
	}
	if opts.JSON {
		return printJSON(gpus)
	}
	rows := make([][]string, 0, len(gpus))
	for _, gpu := range gpus {
		rows = append(rows, []string{
			strconv.Itoa(gpu.ID),
			gpu.Name,
			api.FormatBytes(gpu.MemoryUsed),
			api.FormatBytes(gpu.MemoryTotal),
			fmt.Sprintf("%.1f%%", gpu.MemoryPercent),
			fmt.Sprintf("%d%%", gpu.GPUUtilization),
		})
	}
	printMarkdownTable(os.Stdout, []string{"ID", "Name", "Memory Used", "Memory Total", "Memory", "GPU Util"}, rows)
	return nil
}

func cliModelActive(args []string, opts cliOptions) error {
	if len(args) != 0 {
		return fmt.Errorf("model active does not accept arguments: %s", strings.Join(args, " "))
	}
	models, err := api.GetActiveModels(getSettings().VLLMAPIKey)
	if err != nil {
		return err
	}
	if opts.JSON {
		if models == nil {
			models = []api.ActiveModel{}
		}
		return printJSON(models)
	}
	printActiveModelSections(os.Stdout, models)
	return nil
}

func cliModelCached(args []string, opts cliOptions) error {
	if len(args) != 0 {
		return fmt.Errorf("model cached does not accept arguments: %s", strings.Join(args, " "))
	}
	models, err := api.GetDownloadedModels(getSettings().HFCache)
	if err != nil {
		return err
	}
	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})
	if opts.JSON {
		if models == nil {
			models = []api.DownloadedModel{}
		}
		return printJSON(models)
	}
	rows := make([][]string, 0, len(models))
	for _, model := range models {
		rows = append(rows, []string{
			model.Name,
			model.Size,
			zeroAsDash(model.DefaultMaxModelLen),
			api.FormatBytes(model.KVCacheBytesPerToken),
		})
	}
	printMarkdownTable(os.Stdout, []string{"Model", "Size", "Default Max Len", "KV Bytes/Token"}, rows)
	return nil
}

func cliModelInfo(args []string, opts cliOptions) error {
	if len(args) != 1 {
		return errors.New("usage: model info <model>")
	}
	info, err := fetchHFModelInfo(args[0])
	if err != nil {
		return err
	}
	if opts.JSON {
		return printJSON(info)
	}
	printMarkdownTable(os.Stdout, []string{"Model", "Size", "Downloads", "Likes"}, [][]string{{
		info.Model,
		api.FormatBytes(uint64(info.SizeBytes)),
		strconv.Itoa(info.Downloads),
		strconv.Itoa(info.Likes),
	}})
	return nil
}

func cliModelDownload(args []string, opts cliOptions) error {
	if len(args) != 1 {
		return errors.New("usage: model download <model>")
	}
	settings := getSettings()
	containerID, err := api.DownloadModel(args[0], settings.HFCache, settings.HFToken)
	if err != nil {
		return err
	}
	return printMutationResult(opts, map[string]interface{}{
		"status":       "download_started",
		"model":        args[0],
		"container_id": containerID,
	})
}

func cliModelProgress(args []string, opts cliOptions) error {
	if len(args) != 1 {
		return errors.New("usage: model progress <model>")
	}
	size, err := api.GetModelDirectorySize(args[0], getSettings().HFCache)
	if err != nil {
		return err
	}
	payload := struct {
		Model           string `json:"model"`
		DownloadedBytes uint64 `json:"downloaded_bytes"`
	}{
		Model:           args[0],
		DownloadedBytes: size,
	}
	if opts.JSON {
		return printJSON(payload)
	}
	printMarkdownTable(os.Stdout, []string{"Model", "Downloaded"}, [][]string{{args[0], api.FormatBytes(size)}})
	return nil
}

func cliModelRecommend(opts cliOptions, useCase string, fitOnly bool, limit int) error {
	recommendations, err := api.RecommendModels(api.RecommendationOptions{
		UseCase: useCase,
		FitOnly: fitOnly,
		Limit:   limit,
	})
	if err != nil {
		return err
	}
	if opts.JSON {
		return printJSON(recommendations)
	}
	rows := make([][]string, 0, len(recommendations.Models))
	for _, model := range recommendations.Models {
		rows = append(rows, []string{
			model.Name,
			model.ParameterCount,
			model.Quantization,
			model.FitLevel,
			fmt.Sprintf("%.1f", model.Score),
			fmt.Sprintf("%.0f", model.ModelQuality),
			fmt.Sprintf("%d", model.RecommendedGPUs),
			model.RecommendedGPUIDs,
			fmt.Sprintf("%.1f GiB", model.RequiredGBPerGPU),
			model.Reason,
		})
	}
	printMarkdownTable(os.Stdout, []string{"Model", "Params", "Quant", "Fit", "Score", "Quality", "GPUs", "GPU IDs", "Need/GPU", "Reason"}, rows)
	return nil
}

func cliSettingsGet(args []string, opts cliOptions) error {
	if len(args) != 0 {
		return fmt.Errorf("settings get does not accept arguments: %s", strings.Join(args, " "))
	}
	status := getSettingsStatus()
	if opts.JSON {
		return printJSON(status)
	}
	printMarkdownTable(os.Stdout, []string{"Setting", "Value"}, [][]string{
		{"HF Cache", status.HFCache},
		{"HF Token", setUnset(status.HFTokenSet)},
		{"vLLM API Key", setUnset(status.VLLMAPIKeySet)},
		{"vLLM CUDA Compatibility", enabledDisabled(status.EnableCUDACompatibility)},
		{"HF Hub Offline", enabledDisabled(status.HFHubOffline)},
		{"Cache Disk Used", api.FormatBytes(status.DiskUsed)},
		{"Cache Disk Total", api.FormatBytes(status.DiskTotal)},
		{"Cache Disk Free", api.FormatBytes(status.DiskFree)},
		{"System RAM Used", api.FormatBytes(status.RAMUsed)},
		{"System RAM Total", api.FormatBytes(status.RAMTotal)},
		{"System RAM Free", api.FormatBytes(status.RAMFree)},
	})
	return nil
}

func confirmDestructive(yes bool, prompt string) error {
	if yes {
		return nil
	}
	if !isTerminal(os.Stdin) {
		return errors.New("confirmation required; rerun with --yes in non-interactive shells")
	}
	fmt.Fprintf(os.Stderr, "%s [y/N]: ", prompt)
	reader := bufio.NewReader(os.Stdin)
	answer, err := reader.ReadString('\n')
	if err != nil {
		if errors.Is(err, io.EOF) {
			return errors.New("confirmation required; rerun with --yes in non-interactive shells")
		}
		return fmt.Errorf("failed to read confirmation: %v", err)
	}
	answer = strings.ToLower(strings.TrimSpace(answer))
	if answer == "y" || answer == "yes" {
		return nil
	}
	return errors.New("cancelled")
}

func isTerminal(file *os.File) bool {
	info, err := file.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

func printMutationResult(opts cliOptions, fields map[string]interface{}) error {
	if opts.JSON {
		return printJSON(fields)
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	rows := make([][]string, 0, len(keys))
	for _, key := range keys {
		rows = append(rows, []string{key, fmt.Sprint(fields[key])})
	}
	printMarkdownTable(os.Stdout, []string{"Field", "Value"}, rows)
	return nil
}

func printJSON(value interface{}) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func printMarkdownTable(w io.Writer, headers []string, rows [][]string) {
	renderMarkdown(w, markdownTable(headers, rows))
}

func printActiveModelSections(w io.Writer, models []api.ActiveModel) {
	if len(models) == 0 {
		renderMarkdown(w, "No active vLLM models.\n")
		return
	}

	var b strings.Builder
	for i, model := range models {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "## %s\n\n", markdownLine(cliContainerName(model)))
		fmt.Fprintf(&b, "- Model: %s\n", markdownLine(model.Model))
		fmt.Fprintf(&b, "- Status: %s (ready: %s)\n", markdownLine(model.State), yesNo(model.IsReady))
		fmt.Fprintf(&b, "- Serve: %s\n", markdownLine(model.APIUrl))
		fmt.Fprintf(&b, "- GPUs: %s, tensor parallel %s, memory utilization %s\n",
			markdownLine(model.GPUIDs),
			markdownLine(model.GPUs),
			markdownLine(model.MemUtil),
		)
		fmt.Fprintf(&b, "- Runtime: max length %s, max seqs %s, KV cache %s, parser %s\n",
			markdownLine(model.MaxLen),
			markdownLine(emptyAsDefault(model.MaxNumSeqs, "default")),
			markdownLine(emptyAsDefault(model.KVCacheDType, "auto")),
			markdownLine(model.ToolParser),
		)
	}
	renderMarkdown(w, b.String())
}

func markdownTable(headers []string, rows [][]string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "| %s |\n", strings.Join(escapeCells(headers), " | "))
	separators := make([]string, len(headers))
	for i := range separators {
		separators[i] = "---"
	}
	fmt.Fprintf(&b, "| %s |\n", strings.Join(separators, " | "))
	for _, row := range rows {
		padded := make([]string, len(headers))
		for i := range padded {
			if i < len(row) {
				padded[i] = row[i]
			}
		}
		fmt.Fprintf(&b, "| %s |\n", strings.Join(escapeCells(padded), " | "))
	}
	return b.String()
}

func renderMarkdown(w io.Writer, markdown string) {
	file, ok := w.(*os.File)
	if !ok || !isTerminal(file) {
		fmt.Fprint(w, markdown)
		return
	}

	renderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(markdownRenderWidth(file)),
	)
	if err != nil {
		fmt.Fprint(w, markdown)
		return
	}

	out, err := renderer.Render(markdown)
	if err != nil {
		fmt.Fprint(w, markdown)
		return
	}
	fmt.Fprint(w, out)
}

func markdownRenderWidth(file *os.File) int {
	width, _, err := term.GetSize(int(file.Fd()))
	if err == nil && width > 0 {
		return width
	}
	return 120
}

func cliContainerName(model api.ActiveModel) string {
	if strings.TrimSpace(model.ContainerName) != "" {
		return model.ContainerName
	}
	if len(model.ContainerID) > 12 {
		return model.ContainerID[:12]
	}
	return model.ContainerID
}

func markdownLine(value string) string {
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	return value
}

func escapeCells(cells []string) []string {
	escaped := make([]string, len(cells))
	for i, cell := range cells {
		cell = strings.ReplaceAll(cell, "\n", " ")
		cell = strings.ReplaceAll(cell, "|", `\|`)
		escaped[i] = cell
	}
	return escaped
}

func zeroAsDash(value int) string {
	if value == 0 {
		return "-"
	}
	return strconv.Itoa(value)
}

func emptyAsDefault(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func yesNo(value bool) string {
	if value {
		return "yes"
	}
	return "no"
}

func setUnset(value bool) string {
	if value {
		return "set"
	}
	return "unset"
}

func enabledDisabled(value bool) string {
	if value {
		return "enabled"
	}
	return "disabled"
}
