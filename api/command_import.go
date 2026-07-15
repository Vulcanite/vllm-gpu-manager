package api

import (
	"fmt"
	"strconv"
	"strings"
)

type ImportedCommandFields struct {
	Model                     string `json:"model,omitempty"`
	ServedModelName           string `json:"served_model_name,omitempty"`
	Port                      int    `json:"port,omitempty"`
	GPUs                      int    `json:"gpus,omitempty"`
	GPUIDs                    string `json:"gpu_ids,omitempty"`
	MemUtil                   string `json:"mem_util,omitempty"`
	MaxLen                    string `json:"max_len,omitempty"`
	MaxNumSeqs                string `json:"max_num_seqs,omitempty"`
	KVCacheDType              string `json:"kv_cache_dtype,omitempty"`
	ToolParser                string `json:"tool_parser,omitempty"`
	TokenizerMode             string `json:"tokenizer_mode,omitempty"`
	ReasoningParser           string `json:"reasoning_parser,omitempty"`
	TrustRemoteCode           bool   `json:"trust_remote_code,omitempty"`
	BlockSize                 string `json:"block_size,omitempty"`
	MoeBackend                string `json:"moe_backend,omitempty"`
	ExpertParallel            bool   `json:"enable_expert_parallel,omitempty"`
	DisableFlashInferAutotune bool   `json:"disable_flashinfer_autotune,omitempty"`
	SpeculativeConfig         string `json:"speculative_config,omitempty"`
}

type ImportedCommand struct {
	Fields        ImportedCommandFields `json:"fields"`
	Image         string                `json:"image,omitempty"`
	ExtraVLLMArgs []string              `json:"extra_vllm_args"`
	Warnings      []string              `json:"warnings"`
	Ignored       []string              `json:"ignored"`
	Unsupported   []string              `json:"unsupported"`
}

type importedCommandBuilder struct {
	result ImportedCommand
}

type vllmFlagSpec struct {
	hasValue bool
}

var managedVLLMValueFlags = map[string]func(*ImportedCommandFields, string){
	"--model":                  func(f *ImportedCommandFields, v string) { f.Model = v },
	"--served-model-name":      func(f *ImportedCommandFields, v string) { f.ServedModelName = v },
	"--host":                   func(*ImportedCommandFields, string) {},
	"--port":                   func(f *ImportedCommandFields, v string) { f.Port, _ = strconv.Atoi(v) },
	"--tensor-parallel-size":   func(f *ImportedCommandFields, v string) { f.GPUs, _ = strconv.Atoi(v) },
	"--gpu-memory-utilization": func(f *ImportedCommandFields, v string) { f.MemUtil = v },
	"--max-model-len":          func(f *ImportedCommandFields, v string) { f.MaxLen = v },
	"--max-num-seqs":           func(f *ImportedCommandFields, v string) { f.MaxNumSeqs = v },
	"--kv-cache-dtype":         func(f *ImportedCommandFields, v string) { f.KVCacheDType = v },
	"--tool-call-parser":       func(f *ImportedCommandFields, v string) { f.ToolParser = v },
	"--tokenizer-mode":         func(f *ImportedCommandFields, v string) { f.TokenizerMode = v },
	"--reasoning-parser":       func(f *ImportedCommandFields, v string) { f.ReasoningParser = v },
	"--block-size":             func(f *ImportedCommandFields, v string) { f.BlockSize = v },
	"--moe-backend":            func(f *ImportedCommandFields, v string) { f.MoeBackend = v },
	"--speculative-config":     func(f *ImportedCommandFields, v string) { f.SpeculativeConfig = v },
	"--api-key":                func(*ImportedCommandFields, string) {},
}

var managedVLLMBoolFlags = map[string]func(*ImportedCommandFields){
	"--trust-remote-code":                func(f *ImportedCommandFields) { f.TrustRemoteCode = true },
	"--enable-expert-parallel":           func(f *ImportedCommandFields) { f.ExpertParallel = true },
	"--no-enable-flashinfer-autotune":    func(f *ImportedCommandFields) { f.DisableFlashInferAutotune = true },
	"--enable-auto-tool-choice":          func(*ImportedCommandFields) {},
	"--enable-prefix-caching":            func(*ImportedCommandFields) {},
	"--no-enable-prefix-caching":         func(*ImportedCommandFields) {},
	"--disable-log-requests":             nil,
	"--disable-log-stats":                nil,
	"--enable-chunked-prefill":           nil,
	"--enable-prompt-embeds":             nil,
	"--enforce-eager":                    nil,
	"--disable-frontend-multiprocessing": nil,
}

var allowedExtraVLLMFlags = map[string]vllmFlagSpec{
	"--max-num-batched-tokens":       {hasValue: true},
	"--max-num-partial-prefills":     {hasValue: true},
	"--max-long-partial-prefills":    {hasValue: true},
	"--long-prefill-token-threshold": {hasValue: true},
	"--max-seq-len-to-capture":       {hasValue: true},
	"--max-logprobs":                 {hasValue: true},
	"--chat-template":                {hasValue: true},
	"--chat-template-content-format": {hasValue: true},
	"--response-role":                {hasValue: true},
	"--generation-config":            {hasValue: true},
	"--guided-decoding-backend":      {hasValue: true},
	"--rope-scaling":                 {hasValue: true},
	"--rope-theta":                   {hasValue: true},
	"--limit-mm-per-prompt":          {hasValue: true},
	"--mm-processor-kwargs":          {hasValue: true},
	"--scheduler-delay-factor":       {hasValue: true},
	"--swap-space":                   {hasValue: true},
	"--cpu-offload-gb":               {hasValue: true},
	"--compilation-config":           {hasValue: true},
	"--override-generation-config":   {hasValue: true},
	"--override-pooler-config":       {hasValue: true},
	"--hf-overrides":                 {hasValue: true},
	"--disable-log-requests":         {hasValue: false},
	"--disable-log-stats":            {hasValue: false},
	"--enable-chunked-prefill":       {hasValue: false},
	"--enable-prompt-embeds":         {hasValue: false},
	"--enforce-eager":                {hasValue: false},
	"--language-model-only":          {hasValue: false},
}

var dockerValueFlags = map[string]struct{}{
	"--gpus": {}, "--name": {}, "--publish": {}, "-p": {}, "--volume": {}, "-v": {}, "--env": {}, "-e": {},
	"--ipc": {}, "--network": {}, "--entrypoint": {}, "--mount": {}, "--cap-add": {}, "--device": {},
	"--pid": {}, "--user": {}, "--security-opt": {}, "--restart": {}, "--workdir": {}, "-w": {},
	"--hostname": {}, "--add-host": {}, "--ulimit": {}, "--shm-size": {}, "--runtime": {},
	"--pull": {}, "--platform": {}, "--env-file": {}, "--label": {}, "-l": {},
}

var dockerBoolFlags = map[string]struct{}{
	"-d": {}, "--detach": {}, "--rm": {}, "--privileged": {}, "--init": {}, "-it": {}, "-i": {}, "-t": {},
}

var blockedDockerFlags = map[string]struct{}{
	"--privileged": {}, "--network": {}, "--entrypoint": {}, "--mount": {}, "--cap-add": {}, "--device": {},
	"--pid": {}, "--user": {}, "--security-opt": {}, "--restart": {},
}

func ParseImportedCommand(command string) (ImportedCommand, error) {
	tokens, err := shellFields(command)
	if err != nil {
		return ImportedCommand{}, err
	}
	if len(tokens) == 0 {
		return ImportedCommand{}, validationError("command is required")
	}

	builder := importedCommandBuilder{
		result: ImportedCommand{
			ExtraVLLMArgs: []string{},
			Warnings:      []string{},
			Ignored:       []string{},
			Unsupported:   []string{},
		},
	}

	if len(tokens) >= 2 && tokens[0] == "sudo" {
		tokens = tokens[1:]
	}
	if len(tokens) >= 2 && tokens[0] == "docker" && tokens[1] == "run" {
		if err := builder.parseDockerRun(tokens[2:]); err != nil {
			return ImportedCommand{}, err
		}
	} else if len(tokens) >= 2 && tokens[0] == "vllm" && tokens[1] == "serve" {
		builder.parseVLLMArgs(tokens[2:])
	} else if len(tokens) >= 4 && tokens[0] == "python" && tokens[1] == "-m" && strings.HasPrefix(tokens[2], "vllm.") {
		builder.parseVLLMArgs(tokens[3:])
	} else {
		return ImportedCommand{}, validationError("command must start with docker run, vllm serve, or python -m vllm...")
	}

	if builder.result.Fields.Model == "" {
		builder.result.Warnings = append(builder.result.Warnings, "No model was found in the imported command.")
	}
	return builder.result, nil
}

func (b *importedCommandBuilder) parseDockerRun(tokens []string) error {
	for i := 0; i < len(tokens); i++ {
		token := tokens[i]
		if token == "" {
			continue
		}
		if !strings.HasPrefix(token, "-") {
			b.result.Image = token
			remaining := tokens[i+1:]
			if len(remaining) >= 2 && remaining[0] == "vllm" && remaining[1] == "serve" {
				remaining = remaining[2:]
			}
			b.parseVLLMArgs(remaining)
			return nil
		}

		flag, value, hasInlineValue := splitFlagValue(token)
		if _, blocked := blockedDockerFlags[flag]; blocked {
			b.result.Unsupported = append(b.result.Unsupported, token)
			b.result.Warnings = append(b.result.Warnings, fmt.Sprintf("%s was not imported because the manager owns Docker security settings.", flag))
		}

		if _, ok := dockerBoolFlags[flag]; ok {
			if !hasInlineValue {
				continue
			}
		}

		if _, ok := dockerValueFlags[flag]; ok {
			if !hasInlineValue {
				if i+1 >= len(tokens) {
					return validationError("%s requires a value", flag)
				}
				value = tokens[i+1]
				i++
			}
			b.handleDockerValue(flag, value)
			continue
		}

		b.result.Unsupported = append(b.result.Unsupported, token)
	}
	return validationError("docker run command does not include an image")
}

func (b *importedCommandBuilder) handleDockerValue(flag string, value string) {
	switch flag {
	case "--gpus":
		b.parseDockerGPUs(value)
	case "-p", "--publish":
		b.parseDockerPort(value)
	case "-v", "--volume":
		b.result.Ignored = append(b.result.Ignored, fmt.Sprintf("%s %s", flag, value))
		b.result.Warnings = append(b.result.Warnings, "Volume mounts were ignored; the manager uses the configured Hugging Face cache.")
	case "-e", "--env":
		if strings.HasPrefix(value, "HF_HOME=") {
			b.result.Ignored = append(b.result.Ignored, fmt.Sprintf("%s %s", flag, value))
			return
		}
		b.result.Unsupported = append(b.result.Unsupported, fmt.Sprintf("%s %s", flag, value))
		b.result.Warnings = append(b.result.Warnings, "Arbitrary Docker environment variables were not imported.")
	default:
		b.result.Ignored = append(b.result.Ignored, fmt.Sprintf("%s %s", flag, value))
	}
}

func (b *importedCommandBuilder) parseDockerGPUs(value string) {
	value = strings.TrimSpace(value)
	if value == "all" {
		b.result.Fields.GPUIDs = "all"
		return
	}
	if n, err := strconv.Atoi(value); err == nil && n > 0 {
		b.result.Fields.GPUs = n
		return
	}
	if strings.HasPrefix(value, "device=") {
		ids := strings.TrimPrefix(value, "device=")
		parts := strings.Split(ids, ",")
		valid := make([]string, 0, len(parts))
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if _, err := strconv.Atoi(part); err == nil {
				valid = append(valid, part)
			}
		}
		if len(valid) > 0 {
			b.result.Fields.GPUs = len(valid)
			b.result.Fields.GPUIDs = strings.Join(valid, ",")
		}
	}
}

func (b *importedCommandBuilder) parseDockerPort(value string) {
	parts := strings.Split(value, ":")
	if len(parts) < 2 {
		return
	}
	hostPort := parts[len(parts)-2]
	if port, err := strconv.Atoi(hostPort); err == nil {
		b.result.Fields.Port = port
	}
}

func (b *importedCommandBuilder) parseVLLMArgs(tokens []string) {
	for i := 0; i < len(tokens); i++ {
		token := tokens[i]
		if token == "" {
			continue
		}
		if !strings.HasPrefix(token, "-") {
			if b.result.Fields.Model == "" {
				b.result.Fields.Model = token
			} else {
				b.result.Unsupported = append(b.result.Unsupported, token)
			}
			continue
		}

		flag, value, hasInlineValue := splitFlagValue(token)
		if setter, ok := managedVLLMValueFlags[flag]; ok {
			if !hasInlineValue {
				if i+1 >= len(tokens) {
					b.result.Unsupported = append(b.result.Unsupported, flag)
					continue
				}
				value = tokens[i+1]
				i++
			}
			if flag == "--api-key" {
				b.result.Ignored = append(b.result.Ignored, flag)
				b.result.Warnings = append(b.result.Warnings, "--api-key was ignored; API keys are managed by vLLM Manager.")
				continue
			}
			setter(&b.result.Fields, value)
			continue
		}
		if setter, ok := managedVLLMBoolFlags[flag]; ok {
			if setter != nil {
				setter(&b.result.Fields)
			}
			if _, extra := allowedExtraVLLMFlags[flag]; !extra {
				b.result.Ignored = append(b.result.Ignored, flag)
			} else {
				b.result.ExtraVLLMArgs = append(b.result.ExtraVLLMArgs, flag)
			}
			continue
		}
		if spec, ok := allowedExtraVLLMFlags[flag]; ok {
			b.result.ExtraVLLMArgs = append(b.result.ExtraVLLMArgs, flag)
			if spec.hasValue {
				if !hasInlineValue {
					if i+1 >= len(tokens) {
						b.result.Unsupported = append(b.result.Unsupported, flag)
						continue
					}
					value = tokens[i+1]
					i++
				}
				b.result.ExtraVLLMArgs = append(b.result.ExtraVLLMArgs, value)
			}
			continue
		}

		b.result.Unsupported = append(b.result.Unsupported, token)
		if !hasInlineValue && i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1], "-") {
			i++
		}
	}
}

func NormalizeExtraVLLMArgs(args []string) ([]string, error) {
	normalized := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		token := strings.TrimSpace(args[i])
		if token == "" {
			continue
		}
		if !strings.HasPrefix(token, "-") {
			return nil, validationError("extra vLLM arg %q is missing a flag", token)
		}
		flag, value, hasInlineValue := splitFlagValue(token)
		if _, managed := managedVLLMValueFlags[flag]; managed {
			return nil, validationError("extra vLLM arg %q is managed by vLLM Manager", flag)
		}
		if _, managed := managedVLLMBoolFlags[flag]; managed {
			if _, ok := allowedExtraVLLMFlags[flag]; !ok {
				return nil, validationError("extra vLLM arg %q is managed by vLLM Manager", flag)
			}
		}
		spec, ok := allowedExtraVLLMFlags[flag]
		if !ok {
			return nil, validationError("extra vLLM arg %q is not allowed", flag)
		}
		normalized = append(normalized, flag)
		if spec.hasValue {
			if !hasInlineValue {
				if i+1 >= len(args) {
					return nil, validationError("extra vLLM arg %q requires a value", flag)
				}
				value = args[i+1]
				i++
			}
			if strings.TrimSpace(value) == "" {
				return nil, validationError("extra vLLM arg %q requires a value", flag)
			}
			normalized = append(normalized, value)
		} else if hasInlineValue {
			return nil, validationError("extra vLLM arg %q does not accept a value", flag)
		}
	}
	return normalized, nil
}

func splitFlagValue(token string) (string, string, bool) {
	if idx := strings.Index(token, "="); idx > 0 {
		return token[:idx], token[idx+1:], true
	}
	return token, "", false
}

func shellFields(input string) ([]string, error) {
	var tokens []string
	var b strings.Builder
	inSingle := false
	inDouble := false
	haveToken := false
	for i := 0; i < len(input); i++ {
		ch := input[i]
		switch {
		case inSingle:
			if ch == '\'' {
				inSingle = false
			} else {
				b.WriteByte(ch)
			}
			haveToken = true
		case inDouble:
			if ch == '"' {
				inDouble = false
			} else if ch == '\\' {
				if i+1 >= len(input) {
					return nil, validationError("unterminated escape in command")
				}
				i++
				b.WriteByte(input[i])
			} else {
				b.WriteByte(ch)
			}
			haveToken = true
		default:
			switch ch {
			case '\'':
				inSingle = true
				haveToken = true
			case '"':
				inDouble = true
				haveToken = true
			case '\\':
				if i+1 >= len(input) {
					return nil, validationError("unterminated escape in command")
				}
				if input[i+1] == '\n' {
					i++
					continue
				}
				if input[i+1] == '\r' && i+2 < len(input) && input[i+2] == '\n' {
					i += 2
					continue
				}
				i++
				b.WriteByte(input[i])
				haveToken = true
			case ' ', '\t', '\n', '\r':
				if haveToken {
					tokens = append(tokens, b.String())
					b.Reset()
					haveToken = false
				}
			default:
				b.WriteByte(ch)
				haveToken = true
			}
		}
	}
	if inSingle || inDouble {
		return nil, validationError("unterminated quote in command")
	}
	if haveToken {
		tokens = append(tokens, b.String())
	}
	return tokens, nil
}
