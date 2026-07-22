import { useCallback, useState } from 'react';
import { createPortal } from 'react-dom';
import { Info, Play, RefreshCw } from 'lucide-react';

import { generateVLLMToken } from '../../lib/tokens.js';

const FIELD_HELP = {
  model: 'The downloaded Hugging Face model to serve with vLLM.',
  servedModelName: 'Optional public model alias passed to vLLM as --served-model-name.',
  port: 'The host port that clients or the proxy use to reach this model server.',
  gpus: 'Number of GPUs assigned to this vLLM server.',
  gpuPinning: 'Choose the exact GPU IDs assigned to this server when you are not using every GPU.',
  memory: 'Fraction of each selected GPU memory that vLLM may use.',
  maxLen: 'Maximum context length passed to vLLM as --max-model-len.',
  maxNumSeqs: 'Maximum number of concurrent sequences. Defaults to 256.',
  cpuOffloadGB: 'Offloads model weights to CPU memory per GPU. This does not replace KV cache memory and needs fast CPU-GPU interconnect.',
  dtype: 'Model weight and activation data type. Auto uses the type selected by vLLM and the model configuration.',
  parallelismMode: 'Tensor parallelism splits each layer across GPUs. Pipeline parallelism assigns different layers to different GPUs. Multiple GPUs are required.',
  moeBackend: 'Backend implementation used by vLLM for Mixture-of-Experts models. Enable Expert Parallel first.',
  expertParallel: 'Enable vLLM expert parallelism for MoE models. Requires multiple GPUs in Tensor mode.',
  languageModelOnly: 'Disable multimodal inputs and serve only the language model component.',
  trustRemoteCode: 'Allow model repositories to execute custom Python code inside the vLLM container.',
  prefixCaching: 'Reuse KV cache blocks when requests share the same token prefix. Disable it when cache isolation is preferred.',
  speculativeConfig: 'JSON object passed directly to vLLM --speculative-config for speculative decoding.',
  speculativeDecoding: 'Enable speculative decoding and provide its vLLM JSON configuration.',
  extraVLLMArgs: 'Validated vLLM flags imported from a command that do not have dedicated fields yet.',
  tokenizerMode: 'Tokenizer implementation mode passed to vLLM.',
  reasoningParser: 'Parser used by vLLM to extract reasoning content from supported model outputs.',
  toolParser: 'Parser used by vLLM to convert model tool-call text into structured tool calls.',
  kvCacheDType: 'KV cache data type. FP8 modes can reduce memory usage when supported.',
  bindIP: 'Host IP address for direct server port binding outside proxy mode.',
  apiKeyAuth: 'Require clients to send an API key when calling this direct vLLM server.',
  customApiKey: 'Use a per-server API key instead of the global vLLM API key.',
  serverApiKey: 'The custom per-server API key that will be passed to vLLM.',
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const MEMORY_UTIL_STEP = 0.05;
const MIN_MEMORY_UTIL = 0.05;
const MAX_MEMORY_UTIL = 1;

const formatMemoryUtil = (value) => {
  const rounded = Math.round(value * 100) / 100;
  return rounded.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
};

const FieldLabel = ({ children, help, htmlFor }) => {
  const [tooltip, setTooltip] = useState(null);

  const showTooltip = useCallback((event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const estimatedHeight = 72;
    const margin = 12;
    const openBelow = rect.bottom + estimatedHeight + margin <= window.innerHeight;
    const left = clamp(rect.left + rect.width / 2, margin + 136, window.innerWidth - margin - 136);
    setTooltip({
      text: help,
      left,
      top: openBelow ? rect.bottom + 8 : rect.top - 8,
      placement: openBelow ? 'below' : 'above',
    });
  }, [help]);

  const hideTooltip = useCallback(() => setTooltip(null), []);

  return (
    <div className="form-label field-label-with-help">
      {htmlFor ? (
        <label className="field-label-text" htmlFor={htmlFor}>{children}</label>
      ) : (
        <span className="field-label-text">{children}</span>
      )}
      <button
        type="button"
        className="field-info"
        aria-label={`${children}: ${help}`}
        onBlur={hideTooltip}
        onClick={showTooltip}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
      >
        <Info size={13} aria-hidden="true" />
      </button>
      {tooltip && createPortal(
        <div
          className={`field-tooltip-portal ${tooltip.placement === 'above' ? 'above' : 'below'}`}
          style={{ left: tooltip.left, top: tooltip.top }}
          role="tooltip"
        >
          {tooltip.text}
        </div>,
        document.body,
      )}
    </div>
  );
};

export const DeployServerCard = ({
  downloadedModels,
  embedded = false,
  gpuList,
  gpuUnavailableMessage,
  handleGpuCountChange,
  handleParallelismModeChange,
  handleStart,
  hasGlobalVllmAPIKey,
  kvCacheDTypeOptions,
  dtypeOptions,
  launchError,
  launching,
  maxGpus,
  modelLocked = false,
  noGpuDetected,
  proxyEnabled = false,
  scaleMaxLen,
  selectedGpuIds,
  serverApiKey,
  setServerApiKey,
  setStartDType,
  setStartEnableExpertParallel,
  setStartLanguageModelOnly,
  setStartMoeBackend,
  setStartSpeculativeConfig,
  setUseServerApiKey,
  setUseCustomServerApiKey,
  setStartKvCacheDType,
  setStartMaxLen,
  setStartMaxNumSeqs,
  setStartCPUOffloadGB,
  setStartServedModelName,
  setStartMem,
  setStartBindIP,
  setStartModelName,
  setStartParser,
  setStartReasoningParser,
  setStartTokenizerMode,
  setStartTrustRemoteCode,
  setStartEnablePrefixCaching,
  setStartExtraVLLMArgs,
  setStartPort,
  startGpus,
  startDType,
  startParallelismMode,
  startKvCacheDType,
  startMaxLen,
  startMaxNumSeqs,
  startCPUOffloadGB,
  cpuOffloadMaxGB,
  startServedModelName,
  startMem,
  startBindIP,
  startModelName,
  startEnableExpertParallel,
  startLanguageModelOnly,
  startExtraVLLMArgs,
  startMoeBackend,
  startParser,
  startPort,
  startReasoningParser,
  startSpeculativeConfig,
  startTokenizerMode,
  startTrustRemoteCode,
  startEnablePrefixCaching,
  selectedModelIsMultimodal,
  submitLabel = 'Deploy Server',
  useServerApiKey,
  useCustomServerApiKey,
  moeBackendOptions,
  reasoningParserOptions,
  toggleGpu,
  tokenizerModeOptions,
  toolParserOptions,
}) => {
  const [extraArgsExpanded, setExtraArgsExpanded] = useState(false);
  const [speculativeExpanded, setSpeculativeExpanded] = useState(Boolean(String(startSpeculativeConfig || '').trim()));
  const showExtraVLLMArgs = extraArgsExpanded || Boolean(String(startExtraVLLMArgs || '').trim());
  const showSpeculativeConfig = speculativeExpanded || Boolean(String(startSpeculativeConfig || '').trim());

  const handleServerKeyToggle = (event) => {
    const checked = event.target.checked;
    setUseServerApiKey(checked);
    if (!checked) {
      setUseCustomServerApiKey(false);
      setServerApiKey('');
    }
  };

  const handleCustomServerKeyToggle = (event) => {
    const checked = event.target.checked;
    setUseCustomServerApiKey(checked);
    if (!checked) {
      setServerApiKey('');
    }
  };

  const handleGenerateServerApiKey = () => {
    setServerApiKey(generateVLLMToken());
  };

  const handleExtraArgsToggle = (event) => {
    const checked = event.target.checked;
    setExtraArgsExpanded(checked);
    if (!checked) {
      setStartExtraVLLMArgs('');
    }
  };

  const handleSpeculativeToggle = (event) => {
    const checked = event.target.checked;
    setSpeculativeExpanded(checked);
    if (!checked) setStartSpeculativeConfig('');
  };

  const handleExpertParallelToggle = (event) => {
    const checked = event.target.checked;
    setStartEnableExpertParallel(checked);
    if (!checked) setStartMoeBackend('auto');
  };

  const currentGpuCount = Number(startGpus) || 1;
  const selectedModelIsMissing = startModelName && !downloadedModels.some((model) => model.name === startModelName);
  const canDecreaseGpus = !noGpuDetected && currentGpuCount > 1;
  const canIncreaseGpus = !noGpuDetected && currentGpuCount < maxGpus;
  const setGpuCount = (value) => {
    handleGpuCountChange({ target: { value: String(value) } });
  };
  const decreaseGpus = () => {
    if (!canDecreaseGpus) return;
    setGpuCount(currentGpuCount - 1);
  };
  const increaseGpus = () => {
    if (!canIncreaseGpus) return;
    setGpuCount(currentGpuCount + 1);
  };

  const currentMem = Number(startMem);
  const normalizedMem = Number.isNaN(currentMem) ? 0.5 : currentMem;
  const canDecreaseMem = normalizedMem > MIN_MEMORY_UTIL;
  const canIncreaseMem = normalizedMem < MAX_MEMORY_UTIL;
  const stepMemoryUtil = (direction) => {
    const next = clamp(normalizedMem + (MEMORY_UTIL_STEP * direction), MIN_MEMORY_UTIL, MAX_MEMORY_UTIL);
    setStartMem(formatMemoryUtil(next));
  };

  const currentCPUOffload = Number(startCPUOffloadGB);
  const normalizedCPUOffload = Number.isInteger(currentCPUOffload) && currentCPUOffload >= 0 ? currentCPUOffload : 0;
  const canDecreaseCPUOffload = normalizedCPUOffload > 0;
  const canIncreaseCPUOffload = cpuOffloadMaxGB === null || normalizedCPUOffload < cpuOffloadMaxGB;
  const setCPUOffload = (value) => {
    const next = Math.max(0, Math.trunc(Number(value) || 0));
    if (cpuOffloadMaxGB !== null) {
      setStartCPUOffloadGB(String(Math.min(next, Math.max(0, cpuOffloadMaxGB))));
      return;
    }
    setStartCPUOffloadGB(String(next));
  };
  const decreaseCPUOffload = () => {
    if (!canDecreaseCPUOffload) return;
    setCPUOffload(normalizedCPUOffload - 1);
  };
  const increaseCPUOffload = () => {
    if (!canIncreaseCPUOffload) return;
    setCPUOffload(normalizedCPUOffload + 1);
  };
  const effectiveDTypeOptions = dtypeOptions.includes(startDType)
    ? dtypeOptions
    : [...dtypeOptions, startDType].filter(Boolean);
  const parallelismEligible = currentGpuCount >= 2;
  const expertParallelEligible = parallelismEligible && startParallelismMode === 'tensor';
  const moeBackendEligible = expertParallelEligible && startEnableExpertParallel;
  const showLanguageModelOnly = selectedModelIsMultimodal || startLanguageModelOnly;

  return (
    <div className={embedded ? 'launch-deploy-card deploy-server-panel' : 'card launch-deploy-card'}>
      {!embedded && (
        <div className="card-header">
          <h2 className="card-title">
            <Play size={18} /> Deploy Server
          </h2>
        </div>
      )}
      {launchError && <div className="alert alert-danger">{launchError}</div>}
      {noGpuDetected && !launchError && <div className="alert alert-danger">{gpuUnavailableMessage}</div>}
      <form onSubmit={handleStart}>
        <div className="form-group">
          <FieldLabel htmlFor="startModelName" help={FIELD_HELP.model}>Downloaded Model</FieldLabel>
          <select id="startModelName" className="form-control" value={startModelName} onChange={(event) => setStartModelName(event.target.value)} disabled={modelLocked || launching} required>
            <option value="" disabled>
              Select a model
            </option>
            {selectedModelIsMissing && (
              <option value={startModelName}>
                {startModelName}
              </option>
            )}
            {downloadedModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} ({model.size})
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <FieldLabel htmlFor="startServedModelName" help={FIELD_HELP.servedModelName}>Served Model Name</FieldLabel>
          <input
            id="startServedModelName"
            type="text"
            className="form-control"
            value={startServedModelName}
            onChange={(event) => setStartServedModelName(event.target.value)}
            placeholder="Optional alias exposed by vLLM"
            disabled={launching}
          />
        </div>

        <div className="form-grid">
          <div className="form-group">
            <FieldLabel htmlFor="startPort" help={FIELD_HELP.port}>Port</FieldLabel>
            <input id="startPort" type="number" className="form-control" value={startPort} onChange={(event) => setStartPort(event.target.value)} required />
          </div>
          <div className="form-group">
            <FieldLabel htmlFor="startGpus" help={FIELD_HELP.gpus}>GPUs</FieldLabel>
            <div className="inline-control gpu-count-control">
              <button className="btn btn-outline btn-compact" type="button" onClick={decreaseGpus} disabled={!canDecreaseGpus}>
                -
              </button>
              <input
                id="startGpus"
                type="number"
                className="form-control"
                value={startGpus}
                onChange={handleGpuCountChange}
                min="1"
                max={noGpuDetected ? 1 : maxGpus}
                step="1"
                disabled={noGpuDetected}
                required
              />
              <button className="btn btn-outline btn-compact" type="button" onClick={increaseGpus} disabled={!canIncreaseGpus}>
                +
              </button>
            </div>
          </div>
        </div>

        <div className={maxGpus > 1 ? 'form-grid' : 'form-group'}>
          {maxGpus > 1 && (
            <div className="form-group">
              <FieldLabel help={FIELD_HELP.gpuPinning}>GPU Pinning</FieldLabel>
              <div className="pill-container">
                {noGpuDetected ? (
                  <span className="text-secondary text-sm">No GPUs detected</span>
                ) : (
                  gpuList.map((gpu) => (
                    <button key={gpu.id} className={`pill ${selectedGpuIds.includes(gpu.id) ? 'selected' : ''}`} type="button" onClick={() => toggleGpu(gpu.id)}>
                      GPU {gpu.id}
                    </button>
                  ))
                )}
              </div>
            </div>
          )}
          <div className="form-group">
            <FieldLabel htmlFor="startMem" help={FIELD_HELP.memory}>Memory Utilization</FieldLabel>
            <div className="inline-control memory-util-control">
              <button className="btn btn-outline btn-compact" type="button" onClick={() => stepMemoryUtil(-1)} disabled={!canDecreaseMem}>
                -
              </button>
              <input
                id="startMem"
                type="number"
                className="form-control"
                value={startMem}
                onChange={(event) => setStartMem(event.target.value)}
                max={MAX_MEMORY_UTIL}
                step="any"
                required
              />
              <button className="btn btn-outline btn-compact" type="button" onClick={() => stepMemoryUtil(1)} disabled={!canIncreaseMem}>
                +
              </button>
            </div>
          </div>
        </div>

        <div className="form-group">
          <FieldLabel htmlFor="startMaxLen" help={FIELD_HELP.maxLen}>Max Context Length</FieldLabel>
          <div className="inline-control max-len-control">
            <button className="btn btn-outline btn-compact" type="button" onClick={() => scaleMaxLen(0.5)}>
              -
            </button>
            <input
              id="startMaxLen"
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              className="form-control"
              value={startMaxLen}
              onChange={(event) => setStartMaxLen(event.target.value)}
            />
            <button className="btn btn-outline btn-compact" type="button" onClick={() => scaleMaxLen(2)}>
              +
            </button>
          </div>
        </div>

        <details className="advanced-settings-accordion">
          <summary>Advanced Settings</summary>
          <div className="advanced-settings-body">
            <div className="form-grid">
              <div className="form-group">
                <FieldLabel htmlFor="startMaxNumSeqs" help={FIELD_HELP.maxNumSeqs}>Max Num Seqs</FieldLabel>
                <input
                  id="startMaxNumSeqs"
                  type="text"
                  className="form-control centered-input no-native-spinner"
                  value={startMaxNumSeqs}
                  onChange={(event) => setStartMaxNumSeqs(event.target.value)}
                  inputMode="numeric"
                />
              </div>
              <div className="form-group">
                <FieldLabel htmlFor="startCPUOffloadGB" help={FIELD_HELP.cpuOffloadGB}>CPU Offload per GPU (GiB)</FieldLabel>
                <div className="inline-control cpu-offload-control">
                  <button className="btn btn-outline btn-compact" type="button" onClick={decreaseCPUOffload} disabled={!canDecreaseCPUOffload}>
                    -
                  </button>
                  <input
                    id="startCPUOffloadGB"
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    className="form-control"
                    value={startCPUOffloadGB}
                    onChange={(event) => setCPUOffload(event.target.value)}
                    disabled={launching}
                  />
                  <button className="btn btn-outline btn-compact" type="button" onClick={increaseCPUOffload} disabled={!canIncreaseCPUOffload}>
                    +
                  </button>
                </div>
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <FieldLabel htmlFor="startDType" help={FIELD_HELP.dtype}>DType</FieldLabel>
                <select id="startDType" className="form-control" value={startDType} onChange={(event) => setStartDType(event.target.value)}>
                  {effectiveDTypeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <FieldLabel htmlFor="startParallelismMode" help={FIELD_HELP.parallelismMode}>Parallelism Mode</FieldLabel>
                <select
                  id="startParallelismMode"
                  className="form-control"
                  value={startParallelismMode}
                  onChange={(event) => handleParallelismModeChange(event.target.value)}
                  disabled={launching || !parallelismEligible}
                >
                  <option value="tensor">Tensor</option>
                  <option value="pipeline">Pipeline</option>
                </select>
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group launch-toggle-group">
                <div className="setting-row launch-setting-row">
                  <div>
                    <FieldLabel htmlFor="startEnableExpertParallel" help={FIELD_HELP.expertParallel}>Expert Parallel</FieldLabel>
                  </div>
                  <label className="toggle-control" aria-label="Enable expert parallelism for this server">
                    <input
                      id="startEnableExpertParallel"
                      type="checkbox"
                      checked={Boolean(startEnableExpertParallel)}
                      onChange={handleExpertParallelToggle}
                      disabled={launching || !expertParallelEligible}
                    />
                    <span className="toggle-track" aria-hidden="true">
                      <span className="toggle-thumb" />
                    </span>
                  </label>
                </div>
              </div>
              <div className="form-group">
                <FieldLabel htmlFor="startMoeBackend" help={FIELD_HELP.moeBackend}>MoE Backend</FieldLabel>
                <select
                  id="startMoeBackend"
                  className="form-control"
                  value={startMoeBackend}
                  onChange={(event) => setStartMoeBackend(event.target.value)}
                  disabled={launching || !moeBackendEligible}
                >
                  {(moeBackendOptions && moeBackendOptions.length > 0 ? moeBackendOptions : ['auto']).map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group launch-toggle-group">
                <div className="setting-row launch-setting-row">
                  <div>
                    <FieldLabel htmlFor="startTrustRemoteCode" help={FIELD_HELP.trustRemoteCode}>Trust Remote Code</FieldLabel>
                  </div>
                  <label className="toggle-control" aria-label="Allow trust remote code for this server">
                    <input
                      id="startTrustRemoteCode"
                      type="checkbox"
                      checked={Boolean(startTrustRemoteCode)}
                      onChange={(event) => setStartTrustRemoteCode(event.target.checked)}
                      disabled={launching}
                    />
                    <span className="toggle-track" aria-hidden="true">
                      <span className="toggle-thumb" />
                    </span>
                  </label>
                </div>
              </div>
              <div className="form-group launch-toggle-group">
                <div className="setting-row launch-setting-row">
                  <div>
                    <FieldLabel htmlFor="startEnablePrefixCaching" help={FIELD_HELP.prefixCaching}>Prefix Caching</FieldLabel>
                  </div>
                  <label className="toggle-control" aria-label="Enable prefix caching for this server">
                    <input
                      id="startEnablePrefixCaching"
                      type="checkbox"
                      checked={Boolean(startEnablePrefixCaching)}
                      onChange={(event) => setStartEnablePrefixCaching(event.target.checked)}
                      disabled={launching}
                    />
                    <span className="toggle-track" aria-hidden="true">
                      <span className="toggle-thumb" />
                    </span>
                  </label>
                </div>
              </div>
            </div>
            {showLanguageModelOnly && (
              <div className="form-group launch-toggle-group">
                <div className="setting-row launch-setting-row">
                  <div>
                    <FieldLabel htmlFor="startLanguageModelOnly" help={FIELD_HELP.languageModelOnly}>Language Model Only</FieldLabel>
                  </div>
                  <label className="toggle-control" aria-label="Serve only the language model component">
                    <input
                      id="startLanguageModelOnly"
                      type="checkbox"
                      checked={Boolean(startLanguageModelOnly)}
                      onChange={(event) => setStartLanguageModelOnly(event.target.checked)}
                      disabled={launching}
                    />
                    <span className="toggle-track" aria-hidden="true">
                      <span className="toggle-thumb" />
                    </span>
                  </label>
                </div>
              </div>
            )}

            <div className="form-group extra-vllm-args-group">
              <div className="setting-row launch-setting-row">
                <FieldLabel htmlFor="enableSpeculativeDecoding" help={FIELD_HELP.speculativeDecoding}>Enable Speculative Decoding</FieldLabel>
                <label className="toggle-control" aria-label="Enable speculative decoding">
                  <input id="enableSpeculativeDecoding" type="checkbox" checked={showSpeculativeConfig} onChange={handleSpeculativeToggle} disabled={launching} />
                  <span className="toggle-track" aria-hidden="true"><span className="toggle-thumb" /></span>
                </label>
              </div>
              {showSpeculativeConfig && (
                <input
                  id="startSpeculativeConfig"
                  type="text"
                  className="form-control code-input"
                  value={startSpeculativeConfig}
                  onChange={(event) => setStartSpeculativeConfig(event.target.value)}
                  placeholder='{"model":"draft-model","num_speculative_tokens":4}'
                  spellCheck="false"
                  disabled={launching}
                  aria-label={FIELD_HELP.speculativeConfig}
                />
              )}
            </div>

            <div className="form-group extra-vllm-args-group">
              <div className="setting-row launch-setting-row">
                <div>
                  <FieldLabel htmlFor="showExtraVLLMArgs" help={FIELD_HELP.extraVLLMArgs}>Extra vLLM Args</FieldLabel>
                </div>
                <label className="toggle-control" aria-label="Show extra vLLM args field">
                  <input
                    id="showExtraVLLMArgs"
                    type="checkbox"
                    checked={showExtraVLLMArgs}
                    onChange={handleExtraArgsToggle}
                    disabled={launching}
                  />
                  <span className="toggle-track" aria-hidden="true">
                    <span className="toggle-thumb" />
                  </span>
                </label>
              </div>
              {showExtraVLLMArgs && (
                <textarea
                  id="startExtraVLLMArgs"
                  className="form-control code-textarea extra-vllm-args-textarea"
                  value={startExtraVLLMArgs}
                  onChange={(event) => setStartExtraVLLMArgs(event.target.value)}
                  placeholder={'--no-enable-flashinfer-autotune\n--max-num-batched-tokens 8192'}
                  rows={4}
                  spellCheck="false"
                  disabled={launching}
                />
              )}
            </div>

            <div className="form-grid">
              <div className="form-group">
                <FieldLabel htmlFor="startTokenizerMode" help={FIELD_HELP.tokenizerMode}>Tokenizer Mode</FieldLabel>
                <select id="startTokenizerMode" className="form-control" value={startTokenizerMode} onChange={(event) => setStartTokenizerMode(event.target.value)}>
                  {tokenizerModeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <FieldLabel htmlFor="startReasoningParser" help={FIELD_HELP.reasoningParser}>Reasoning Parser</FieldLabel>
                <select id="startReasoningParser" className="form-control" value={startReasoningParser} onChange={(event) => setStartReasoningParser(event.target.value)}>
                  <option value="">Disabled</option>
                  {reasoningParserOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <FieldLabel htmlFor="startParser" help={FIELD_HELP.toolParser}>Tool Call Parser</FieldLabel>
                <select id="startParser" className="form-control" value={startParser} onChange={(event) => setStartParser(event.target.value)}>
                  <option value="">Disabled</option>
                  {(toolParserOptions && toolParserOptions.length > 0 ? toolParserOptions : [
                    'hermes',
                    'llama3_json',
                    'mistral',
                    'granite',
                    'granite4',
                    'granite-20b-fc',
                    'internlm',
                    'jamba',
                    'pythonic',
                    'python_tool',
                    'jinja',
                    'guidance',
                    'lm_format_enforcer',
                    'xlam',
                    'qwen2',
                    'qwen3_coder',
                    'qwen3_xml',
                    'llama4_pythonic',
                    'deepseek_v3',
                    'deepseek_v31',
                    'deepseek_v4',
                    'openai',
                    'kimi_k2',
                    'hunyuan_a13b',
                    'cohere_command3',
                    'longcat',
                    'glm45',
                    'glm47',
                    'functiongemma',
                    'olmo3',
                    'gigachat3',
                    'apertus',
                  ]).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <FieldLabel htmlFor="startKvCacheDType" help={FIELD_HELP.kvCacheDType}>KV Cache DType</FieldLabel>
                <select id="startKvCacheDType" className="form-control" value={startKvCacheDType} onChange={(event) => setStartKvCacheDType(event.target.value)}>
                  {kvCacheDTypeOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </details>

        {!proxyEnabled && (
          <div className="form-group">
            <FieldLabel htmlFor="startBindIP" help={FIELD_HELP.bindIP}>Bind IP</FieldLabel>
            <input
              id="startBindIP"
              type="text"
              className="form-control"
              value={startBindIP}
              onChange={(event) => setStartBindIP(event.target.value)}
              placeholder="Use default server bind IP"
              disabled={launching}
            />
          </div>
        )}

        {hasGlobalVllmAPIKey && !proxyEnabled && (
          <div className="form-group">
            <div className="setting-row launch-setting-row">
              <div>
                <FieldLabel help={FIELD_HELP.apiKeyAuth}>API Key Authentication</FieldLabel>
                <p className="text-secondary text-sm">When enabled, the server uses the globally configured API key for authentication.</p>
              </div>
              <label className="toggle-control" aria-label="Enable API key authentication for this server">
                <input
                  id="startUseServerApiKey"
                  type="checkbox"
                  checked={Boolean(useServerApiKey)}
                  onChange={handleServerKeyToggle}
                  disabled={launching}
                />
                <span className="toggle-track" aria-hidden="true">
                  <span className="toggle-thumb" />
                </span>
              </label>
            </div>
            {useServerApiKey && (
              <div className="api-key-options">
                <div className="setting-row launch-setting-row mt-2">
                  <div>
                    <FieldLabel help={FIELD_HELP.customApiKey}>Custom API Key</FieldLabel>
                    <p className="text-secondary text-sm">Use a per-server key instead of the globally configured API key.</p>
                  </div>
                  <label className="toggle-control" aria-label="Use a custom API key for this server">
                    <input
                      id="startUseCustomServerApiKey"
                      type="checkbox"
                      checked={Boolean(useCustomServerApiKey)}
                      onChange={handleCustomServerKeyToggle}
                      disabled={launching}
                    />
                    <span className="toggle-track" aria-hidden="true">
                      <span className="toggle-thumb" />
                    </span>
                  </label>
                </div>
                {useCustomServerApiKey && (
                  <>
                    <FieldLabel htmlFor="startServerApiKey" help={FIELD_HELP.serverApiKey}>Server API Key</FieldLabel>
                    <div className="inline-control token-generator-control mt-2">
                      <input
                        id="startServerApiKey"
                        type="text"
                        className="form-control"
                        value={serverApiKey || ''}
                        onChange={(event) => setServerApiKey(event.target.value)}
                        aria-label="Custom per-server API key"
                        disabled={launching}
                        autoComplete="off"
                      />
                      <button className="btn btn-outline btn-nowrap" type="button" onClick={handleGenerateServerApiKey} disabled={launching}>
                        <RefreshCw size={14} /> Generate
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        <button type="submit" className="btn btn-primary mt-2 full-width" disabled={launching || noGpuDetected}>
          {launching ? (submitLabel === 'Update Server' ? 'Updating...' : 'Deploying...') : submitLabel}
        </button>
      </form>
    </div>
  );
};
