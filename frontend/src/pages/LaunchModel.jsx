import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { ActiveContainersCard } from '../components/launch/ActiveContainersCard.jsx';
import { CachedModelsCard } from '../components/launch/CachedModelsCard.jsx';
import { DeployServerCard } from '../components/launch/DeployServerCard.jsx';
import { DownloadModelCard } from '../components/launch/DownloadModelCard.jsx';
import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { deleteDownloadedModel, downloadModel, getErrorDetail, getModelInfo, removeModel, restartModel, startModel } from '../lib/api.js';
import { formatBytes } from '../lib/format.js';

const ACTIVE_DOWNLOAD_STORAGE_KEY = 'vllm-manager-active-download';

// Fallback values mirror api.LaunchDefaults(). The live values are fetched from
// /api/config so this preview cannot drift from the backend VRAM preflight; the
// constants below only apply if that request has not resolved yet.
const DEFAULT_LAUNCH_CONFIG = {
  model_vram_safety_margin: 1.08,
  default_gpu_memory_utilization: 0.9,
  default_max_model_len: 32768,
  kv_cache_dtype_options: ['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
  tokenizer_mode_options: ['auto', 'hf', 'slow', 'mistral', 'deepseek_v32', 'deepseek_v4'],
  reasoning_parser_options: ['deepseek_r1', 'deepseek_v4', 'ernie_x1', 'glm45', 'granite', 'hunyuan_a13b', 'mistral', 'openai_gptoss', 'qwen3'],
  block_size_options: [16, 32, 64, 128],
  moe_backend_options: ['auto', 'cutlass', 'deepep_high_throughput', 'deepep_low_latency', 'flashinfer', 'flashinfer_all2allv', 'naive', 'pplx', 'triton'],
};

const normalizeGpuPinning = (selectedIds, availableIds, requestedCount, useAll) => {
  if (useAll) return [];

  const validSelected = selectedIds.filter((id) => availableIds.includes(id));
  let next = validSelected.slice(0, requestedCount);
  if (next.length < requestedCount) {
    const remainingIds = availableIds.filter((id) => !next.includes(id));
    next = [...next, ...remainingIds.slice(0, requestedCount - next.length)];
  }
  return next;
};

const shellQuote = (value) => {
  const text = String(value ?? '');
  if (/^[A-Za-z0-9_./:=,@+-]+$/.test(text)) return text;
  return `'${text.replaceAll("'", "'\"'\"'")}'`;
};

const buildVLLMServeCommand = (payload, { includeAPIKey = false } = {}) => {
  const args = ['vllm', 'serve', payload.model];
  args.push('--host', '0.0.0.0');
  args.push('--port', String(payload.port));
  args.push('--tensor-parallel-size', String(payload.gpus));
  args.push('--gpu-memory-utilization', payload.mem_util);
  args.push('--enable-prefix-caching');
  if (payload.max_len) args.push('--max-model-len', payload.max_len);
  if (payload.max_num_seqs) args.push('--max-num-seqs', payload.max_num_seqs);
  if (payload.kv_cache_dtype && payload.kv_cache_dtype !== 'auto') args.push('--kv-cache-dtype', payload.kv_cache_dtype);
  if (payload.tool_parser) args.push('--tool-call-parser', payload.tool_parser, '--enable-auto-tool-choice');
  if (payload.tokenizer_mode && payload.tokenizer_mode !== 'auto') args.push('--tokenizer-mode', payload.tokenizer_mode);
  if (payload.reasoning_parser) args.push('--reasoning-parser', payload.reasoning_parser);
  if (payload.trust_remote_code) args.push('--trust-remote-code');
  if (payload.block_size) args.push('--block-size', payload.block_size);
  if (payload.moe_backend && payload.moe_backend !== 'auto') args.push('--moe-backend', payload.moe_backend);
  if (payload.enable_expert_parallel) args.push('--enable-expert-parallel');
  if (payload.speculative_config) args.push('--speculative-config', payload.speculative_config);
  if (payload.use_api_key) args.push('--api-key', includeAPIKey && payload.server_api_key ? payload.server_api_key : '<redacted>');

  const lines = [`${args[0]} ${args[1]} ${shellQuote(args[2])}`];
  for (let i = 3; i < args.length; i += 2) {
    if (args[i + 1] === undefined || String(args[i]).startsWith('--enable') || args[i] === '--trust-remote-code') {
      lines.push(`  ${shellQuote(args[i])}`);
      i -= 1;
    } else {
      lines.push(`  ${shellQuote(args[i])} ${shellQuote(args[i + 1])}`);
    }
  }
  return lines.join(' \\\n');
};

export const LaunchModel = ({ gpus, gpuError = '', downloadedModels, activeModels, launchConfig, fetchData, notify, vllmAPIKeySet = false, proxyEnabled = false }) => {
  const DEFAULT_MAX_MODEL_LEN = launchConfig?.default_max_model_len ?? DEFAULT_LAUNCH_CONFIG.default_max_model_len;
  const KV_CACHE_DTYPE_OPTIONS = launchConfig?.kv_cache_dtype_options?.length
    ? launchConfig.kv_cache_dtype_options
    : DEFAULT_LAUNCH_CONFIG.kv_cache_dtype_options;
  const TOKENIZER_MODE_OPTIONS = launchConfig?.tokenizer_mode_options?.length
    ? launchConfig.tokenizer_mode_options
    : DEFAULT_LAUNCH_CONFIG.tokenizer_mode_options;
  const REASONING_PARSER_OPTIONS = launchConfig?.reasoning_parser_options?.length
    ? launchConfig.reasoning_parser_options
    : DEFAULT_LAUNCH_CONFIG.reasoning_parser_options;
  const BLOCK_SIZE_OPTIONS = launchConfig?.block_size_options?.length
    ? launchConfig.block_size_options
    : DEFAULT_LAUNCH_CONFIG.block_size_options;
  const MOE_BACKEND_OPTIONS = launchConfig?.moe_backend_options?.length
    ? launchConfig.moe_backend_options
    : DEFAULT_LAUNCH_CONFIG.moe_backend_options;

  const [deployConfig, setDeployConfig] = useState({
    modelName: '',
    port: '8080',
    gpus: 1,
    gpuIds: [],
    mem: '0.5',
    maxLen: '32768',
    maxNumSeqs: '',
    kvCacheDType: 'auto',
    parser: '',
    tokenizerMode: 'auto',
    reasoningParser: '',
    trustRemoteCode: false,
    blockSize: '16',
    moeBackend: 'auto',
    enableExpertParallel: false,
    speculativeConfig: '',
    bindIP: '',
    useServerApiKey: false,
    useCustomServerApiKey: false,
    serverApiKey: '',
  });

  const updateDeployConfig = (field, value) => {
    setDeployConfig((prev) => ({ ...prev, [field]: value }));
  };

  const startModelName = deployConfig.modelName;
  const startPort = deployConfig.port;
  const startGpus = deployConfig.gpus;
  const startGpuIds = deployConfig.gpuIds;
  const startMem = deployConfig.mem;
  const startMaxLen = deployConfig.maxLen;
  const startMaxNumSeqs = deployConfig.maxNumSeqs;
  const startKvCacheDType = deployConfig.kvCacheDType;
  const startParser = deployConfig.parser;
  const startTokenizerMode = deployConfig.tokenizerMode;
  const startReasoningParser = deployConfig.reasoningParser;
  const startTrustRemoteCode = deployConfig.trustRemoteCode;
  const startBlockSize = deployConfig.blockSize;
  const startMoeBackend = deployConfig.moeBackend;
  const startEnableExpertParallel = deployConfig.enableExpertParallel;
  const startSpeculativeConfig = deployConfig.speculativeConfig;
  const startBindIP = deployConfig.bindIP;
  const startUseServerApiKey = deployConfig.useServerApiKey;
  const startUseCustomServerApiKey = deployConfig.useCustomServerApiKey;
  const startServerApiKey = deployConfig.serverApiKey;
  const [launchError, setLaunchError] = useState('');
  const [launching, setLaunching] = useState(false);
  const [launchConfirmData, setLaunchConfirmData] = useState(null);
  const [deployModalOpen, setDeployModalOpen] = useState(false);

  const [downloadConfirmData, setDownloadConfirmData] = useState(null);
  const [downloadName, setDownloadName] = useState('');
  const [downloadMessage, setDownloadMessage] = useState('');
  const [downloadError, setDownloadError] = useState('');
  const [checkingInfo, setCheckingInfo] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [activeDownloads, setActiveDownloads] = useState(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DOWNLOAD_STORAGE_KEY);
      const parsed = stored ? JSON.parse(stored) : null;
      if (Array.isArray(parsed)) {
        return parsed.filter((item) => item && typeof item.name === 'string');
      }
      return parsed && typeof parsed.name === 'string' ? [parsed] : [];
    } catch {
      return [];
    }
  });
  const [restartError, setRestartError] = useState('');
  const [restartingId, setRestartingId] = useState('');
  const [removeConfirmId, setRemoveConfirmId] = useState(null);
  const [removeError, setRemoveError] = useState('');
  const [removing, setRemoving] = useState(false);
  const [deleteCacheConfirmModel, setDeleteCacheConfirmModel] = useState(null);
  const [deletingCache, setDeletingCache] = useState(false);

  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const gpuList = useMemo(() => gpus || [], [gpus]);
  const noGpuDetected = gpuList.length === 0;
  const gpuUnavailableMessage = gpuError || 'No GPUs detected. Deployment is disabled until nvidia-smi reports at least one GPU.';
  const maxGpus = noGpuDetected ? 0 : gpuList.length;
  const availableGpuIds = useMemo(() => [...gpuList].sort((a, b) => a.id - b.id).map((gpu) => gpu.id), [gpuList]);
  const requestedGpuCount = noGpuDetected ? 0 : Math.min(maxGpus, Math.max(1, Number(startGpus) || 1));
  const usesAllGpus = !noGpuDetected && requestedGpuCount >= maxGpus;
  const selectedGpuIds = useMemo(
    () => normalizeGpuPinning(startGpuIds, availableGpuIds, requestedGpuCount, usesAllGpus),
    [availableGpuIds, requestedGpuCount, startGpuIds, usesAllGpus],
  );

  useAutoDismiss(launchError, setLaunchError);
  useAutoDismiss(restartError, setRestartError);
  useAutoDismiss(removeError, setRemoveError);
  useAutoDismiss(downloadMessage, setDownloadMessage);
  useAutoDismiss(downloadError, setDownloadError);

  useEffect(() => {
    const timeout = setTimeout(() => {
      const model = searchParams.get('model');
      const download = searchParams.get('download');
      if (download) {
        setDownloadName(download);
      }
      if (!model) return;

      const gpuCount = Number(searchParams.get('gpus')) || 1;
      const gpuIds = (searchParams.get('gpu_ids') || '')
        .split(',')
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isInteger(value));

      setDeployConfig((prev) => ({
        ...prev,
        modelName: model,
        gpus: gpuCount,
        gpuIds,
        mem: searchParams.get('mem_util') || prev.mem,
        maxLen: searchParams.get('max_len') || prev.maxLen,
        maxNumSeqs: searchParams.get('max_num_seqs') || prev.maxNumSeqs,
        kvCacheDType: searchParams.get('kv_cache_dtype') || prev.kvCacheDType,
        tokenizerMode: searchParams.get('tokenizer_mode') || prev.tokenizerMode,
        reasoningParser: searchParams.get('reasoning_parser') || prev.reasoningParser,
        trustRemoteCode: searchParams.get('trust_remote_code') === 'true' || prev.trustRemoteCode,
        blockSize: searchParams.get('block_size') || prev.blockSize,
        moeBackend: searchParams.get('moe_backend') || prev.moeBackend,
        enableExpertParallel: searchParams.get('enable_expert_parallel') === 'true' || prev.enableExpertParallel,
      }));
      setDeployModalOpen(true);
    }, 0);

    return () => clearTimeout(timeout);
  }, [searchParams]);

  useEffect(() => {
    try {
      if (activeDownloads.length > 0) {
        localStorage.setItem(ACTIVE_DOWNLOAD_STORAGE_KEY, JSON.stringify(activeDownloads));
      } else {
        localStorage.removeItem(ACTIVE_DOWNLOAD_STORAGE_KEY);
      }
    } catch {
      // Progress persistence is best-effort; polling still works without storage.
    }
  }, [activeDownloads]);

  const handleGpuCountChange = (event) => {
    if (noGpuDetected) {
      updateDeployConfig('gpuIds', []);
      return;
    }

    const rawValue = event.target.value;
    if (rawValue === '') {
      setDeployConfig((prev) => ({ ...prev, gpus: '', gpuIds: [] }));
      return;
    }

    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;

    const nextCount = Math.min(maxGpus, Math.max(1, Math.trunc(parsed)));
    const nextUsesAll = nextCount >= maxGpus;
    setDeployConfig((prev) => ({
      ...prev,
      gpus: nextCount,
      gpuIds: normalizeGpuPinning(prev.gpuIds, availableGpuIds, nextCount, nextUsesAll),
    }));
  };

  const toggleGpu = (id) => {
    if (usesAllGpus) return;
    if (selectedGpuIds.includes(id)) return;

    updateDeployConfig('gpuIds', [...selectedGpuIds.slice(0, requestedGpuCount - 1), id]);
  };

  const scaleMaxLen = (factor) => {
    const current = Number(startMaxLen) || DEFAULT_MAX_MODEL_LEN;
    const next = Math.max(1, Math.round(current * factor));
    updateDeployConfig('maxLen', String(next));
  };

  const validateLaunch = () => {
    const gpusToSend = Number(startGpus);
    if (noGpuDetected) return gpuUnavailableMessage;
    if (!startModelName) return 'Please select a model to launch';
    if (!Number.isInteger(gpusToSend) || gpusToSend < 1) return 'Must request at least 1 GPU.';
    if (gpusToSend > maxGpus) return `Cannot request ${gpusToSend} GPUs. Only ${maxGpus} available.`;

    const portNum = Number(startPort);
    if (!portNum || portNum < 1 || portNum > 65535) return 'Please enter a valid port number between 1 and 65535';

    const memNum = Number(startMem);
    if (startMem && (Number.isNaN(memNum) || memNum <= 0 || memNum > 1.0)) return 'Memory utilization must be between 0.01 and 1.0';

    if (startMaxLen) {
      const maxLenNum = Number(startMaxLen);
      if (Number.isNaN(maxLenNum) || maxLenNum <= 0 || !Number.isInteger(maxLenNum)) {
        return 'Max context length must be a positive integer';
      }
    }

    if (startMaxNumSeqs) {
      const maxNumSeqsNum = Number(startMaxNumSeqs);
      if (Number.isNaN(maxNumSeqsNum) || maxNumSeqsNum <= 0 || !Number.isInteger(maxNumSeqsNum)) {
        return 'Max num seqs must be a positive integer';
      }
    }

    if (startParser && !/^[a-zA-Z0-9_-]+$/.test(startParser)) return 'Tool parser must contain only letters, numbers, underscores, and hyphens';
    if (!TOKENIZER_MODE_OPTIONS.includes(startTokenizerMode)) return 'Tokenizer mode is not supported by this vLLM Manager build.';
    if (startReasoningParser && !REASONING_PARSER_OPTIONS.includes(startReasoningParser)) return 'Reasoning parser is not supported by this vLLM Manager build.';
    if (startBlockSize && !BLOCK_SIZE_OPTIONS.map(String).includes(String(startBlockSize))) return 'Block size must be a supported power of 2.';
    if (!MOE_BACKEND_OPTIONS.includes(startMoeBackend || 'auto')) return 'MoE backend is not supported by this vLLM Manager build.';
    if (startSpeculativeConfig.trim()) {
      try {
        JSON.parse(startSpeculativeConfig);
      } catch {
        return 'Speculative config must be valid JSON.';
      }
    }
    if (!proxyEnabled && startBindIP.trim() && !/^[0-9a-fA-F:.]+$/.test(startBindIP.trim())) return 'Bind IP must be a valid IPv4 or IPv6 address.';
    if (!KV_CACHE_DTYPE_OPTIONS.includes(startKvCacheDType)) return 'KV cache dtype must be auto, fp8, fp8_e5m2, or fp8_e4m3';
    if (!usesAllGpus && selectedGpuIds.length !== gpusToSend) return `Select exactly ${gpusToSend} GPU${gpusToSend === 1 ? '' : 's'} for pinning.`;
    if (!proxyEnabled && vllmAPIKeySet && startUseServerApiKey && startUseCustomServerApiKey && !startServerApiKey.trim()) return 'Enter a per-server API key or turn off custom API key.';
    return '';
  };

  const handleStart = async (event) => {
    event.preventDefault();
    setLaunchError('');

    const validationError = validateLaunch();
    if (validationError) {
      setLaunchError(validationError);
      return;
    }

    const gpuIdsParam = usesAllGpus ? 'all' : [...selectedGpuIds].sort((a, b) => a - b).join(',');
    const effectiveUseServerApiKey = !proxyEnabled && vllmAPIKeySet && startUseServerApiKey;
    const effectiveServerApiKey = effectiveUseServerApiKey && startUseCustomServerApiKey ? startServerApiKey.trim() : '';
    const compactSpeculativeConfig = startSpeculativeConfig.trim() ? JSON.stringify(JSON.parse(startSpeculativeConfig)) : '';
    const payload = {
      model: startModelName,
      port: Number(startPort),
      gpus: Number(startGpus),
      gpu_ids: gpuIdsParam,
      mem_util: startMem !== null && startMem !== undefined ? startMem.toString() : '',
      max_len: startMaxLen ? startMaxLen.toString() : '',
      max_num_seqs: startMaxNumSeqs ? startMaxNumSeqs.toString() : '',
      kv_cache_dtype: startKvCacheDType,
      tool_parser: startParser || '',
      tokenizer_mode: startTokenizerMode || 'auto',
      reasoning_parser: startReasoningParser || '',
      trust_remote_code: Boolean(startTrustRemoteCode),
      block_size: startBlockSize ? startBlockSize.toString() : '',
      moe_backend: startMoeBackend || 'auto',
      enable_expert_parallel: Boolean(startEnableExpertParallel),
      speculative_config: compactSpeculativeConfig,
      bind_ip: !proxyEnabled ? startBindIP.trim() : '',
      use_api_key: effectiveUseServerApiKey,
      server_api_key: effectiveServerApiKey,
    };

    setLaunchConfirmData({
      payload,
      command: buildVLLMServeCommand(payload),
      gpuIds: gpuIdsParam,
    });
    setDeployModalOpen(false);
  };

  const executeStart = async () => {
    if (!launchConfirmData) return;
    try {
      setLaunching(true);
      await startModel(launchConfirmData.payload);
      notify('Model starting. It may take a few minutes to be ready.');
      setLaunchConfirmData(null);
      fetchData();
      navigate('/');
    } catch (error) {
      setLaunchError(getErrorDetail(error, 'Failed to start model'));
      setLaunchConfirmData(null);
      setDeployModalOpen(true);
    } finally {
      setLaunching(false);
    }
  };

  const handleRestart = async (containerId) => {
    setRestartError('');
    try {
      setRestartingId(containerId);
      await restartModel(containerId);
      notify('Container restart requested.');
      fetchData();
    } catch (error) {
      setRestartError(getErrorDetail(error, 'Failed to restart model'));
    } finally {
      setRestartingId('');
    }
  };

  const handleRemove = async () => {
    if (!removeConfirmId) return;
    setRemoveError('');
    setRemoving(true);
    try {
      await removeModel(removeConfirmId);
      setRemoveConfirmId(null);
      notify('Container removed.');
      fetchData();
    } catch (error) {
      setRemoveError(getErrorDetail(error, 'Failed to remove container'));
      setRemoveConfirmId(null);
    } finally {
      setRemoving(false);
    }
  };

  const requestDeleteDownloadedModel = (model) => {
    setDeleteCacheConfirmModel(model);
  };

  const handleDeleteDownloadedModel = async () => {
    if (!deleteCacheConfirmModel) return;

    const modelName = deleteCacheConfirmModel.name;
    setDeletingCache(true);
    try {
      await deleteDownloadedModel(modelName);
      setDeleteCacheConfirmModel(null);
      setActiveDownloads((prev) => prev.filter((item) => item.name !== modelName));
      notify(`Deleted ${modelName} from cache.`);
      fetchData();
    } catch (error) {
      notify(getErrorDetail(error, 'Failed to delete downloaded model'), 'danger');
      setDeleteCacheConfirmModel(null);
    } finally {
      setDeletingCache(false);
    }
  };

  const handleDownload = async (event) => {
    event.preventDefault();
    setDownloadMessage('');
    setDownloadError('');

    const trimmedName = downloadName.trim();
    if (!trimmedName) {
      return;
    }

    // Pre-flight: consult the same endpoint the deploy card uses to see
    // whether the model is already on disk. If it is, we surface a clear
    // message instead of opening the confirm modal — the user almost
    // certainly did not intend to re-download. The server enforces the
    // same check, so this is purely a UX nicety.
    setCheckingInfo(true);
    let infoData = null;
    try {
      const infoRes = await getModelInfo(trimmedName);
      infoData = infoRes.data || null;
      if (infoData?.downloaded) {
        const cached = infoData.cached_bytes;
        const sizeText = typeof cached === 'number' && cached > 0
          ? `${formatBytes(cached)} on disk`
          : 'already on disk';
        setDownloadError(`${trimmedName} is ${sizeText}. Delete it from the cache below if you want to re-download.`);
        return;
      }
      if (infoData?.download_blocked_reason) {
        setDownloadError(infoData.download_blocked_reason);
        return;
      }
    } catch (error) {
      // A network or upstream error from getModelInfo shouldn't block the
      // user from at least seeing the existing confirm modal — the
      // server will still refuse with a 409 if the model is present.
      getErrorDetail(error, '');
    } finally {
      setCheckingInfo(false);
    }

    let sizeText;
    let sizeBytes = 0;
    const warnings = [infoData?.compatibility_warning, infoData?.download_warning].filter(Boolean);
    if (infoData?.size_bytes) {
      sizeBytes = infoData.size_bytes;
      sizeText = `Estimated file size: ${formatBytes(infoData.size_bytes)}.`;
    } else {
      sizeText = 'Model size could not be fetched before download.';
    }
    if (warnings.length > 0) {
      sizeText = `${sizeText} ${warnings.join(' ')}`;
    }

    setDownloadConfirmData({ model: trimmedName, sizeText, sizeBytes });
  };

  const executeDownload = async () => {
    if (!downloadConfirmData) return;
    setDownloadError('');
    setDownloading(true);
    try {
      const res = await downloadModel(downloadConfirmData.model);
      const nextDownload = {
        containerId: res.data?.container_id || '',
        name: downloadConfirmData.model,
        expectedSize: downloadConfirmData.sizeBytes,
      };
      setActiveDownloads((prev) => [nextDownload, ...prev.filter((item) => item.name !== nextDownload.name)]);
      setDownloadMessage('Download started. Progress will update as files appear in the cache.');
      setDownloadName('');
      setDownloadConfirmData(null);
      fetchData();
    } catch (error) {
      const status = error.response?.status;
      if (status === 409) {
        setDownloadError(`${downloadConfirmData.model} is already downloaded. Delete it from the cache below if you want to re-download.`);
      } else {
        setDownloadError(getErrorDetail(error, 'Failed to trigger download'));
      }
      setDownloadConfirmData(null);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="dashboard-grid launch-page-grid">
      <div className="stack launch-side-stack">
        <DownloadModelCard
          activeDownloads={activeDownloads}
          checkingInfo={checkingInfo}
          downloadError={downloadError}
          downloading={downloading}
          downloadMessage={downloadMessage}
          downloadName={downloadName}
          downloadedModels={downloadedModels}
          handleDownload={handleDownload}
          setActiveDownloads={setActiveDownloads}
          setDownloadName={setDownloadName}
        />

        <CachedModelsCard
          deletingCache={deletingCache}
          downloadedModels={downloadedModels}
          requestDeleteDownloadedModel={requestDeleteDownloadedModel}
        />
      </div>

      <div className="launch-servers-row">
        <ActiveContainersCard
          activeModels={activeModels}
          handleRestart={handleRestart}
          onDeployServer={() => {
            setLaunchError('');
            setDeployModalOpen(true);
          }}
          removing={removing}
          removeError={removeError}
          requestRemove={(containerId) => {
            setRemoveError('');
            setRemoveConfirmId(containerId);
          }}
          restartingId={restartingId}
          restartError={restartError}
          proxyEnabled={proxyEnabled}
        />
      </div>

      {deployModalOpen && (
        <ConfirmModal
          title="Deploy Server"
          contentClassName="deploy-server-modal"
          hideActions
          loading={launching}
          onCancel={() => {
            if (!launching) setDeployModalOpen(false);
          }}
        >
          <DeployServerCard
            downloadedModels={downloadedModels}
            embedded
            gpuList={gpuList}
            gpuUnavailableMessage={gpuUnavailableMessage}
            handleGpuCountChange={handleGpuCountChange}
            handleStart={handleStart}
            hasGlobalVllmAPIKey={vllmAPIKeySet}
            proxyEnabled={proxyEnabled}
            kvCacheDTypeOptions={KV_CACHE_DTYPE_OPTIONS}
            launchError={launchError}
            launching={launching}
            maxGpus={maxGpus}
            noGpuDetected={noGpuDetected}
            selectedGpuIds={selectedGpuIds}
            serverApiKey={startServerApiKey}
            setServerApiKey={(v) => updateDeployConfig('serverApiKey', v)}
            setStartBlockSize={(v) => updateDeployConfig('blockSize', v)}
            setStartMoeBackend={(v) => updateDeployConfig('moeBackend', v)}
            setStartEnableExpertParallel={(v) => updateDeployConfig('enableExpertParallel', v)}
            setStartSpeculativeConfig={(v) => updateDeployConfig('speculativeConfig', v)}
            setUseServerApiKey={(v) => updateDeployConfig('useServerApiKey', v)}
            setUseCustomServerApiKey={(v) => updateDeployConfig('useCustomServerApiKey', v)}
            setStartKvCacheDType={(v) => updateDeployConfig('kvCacheDType', v)}
            setStartMaxLen={(v) => updateDeployConfig('maxLen', v)}
            setStartMaxNumSeqs={(v) => updateDeployConfig('maxNumSeqs', v)}
            scaleMaxLen={scaleMaxLen}
            setStartMem={(v) => updateDeployConfig('mem', v)}
            setStartModelName={(v) => updateDeployConfig('modelName', v)}
            setStartParser={(v) => updateDeployConfig('parser', v)}
            setStartReasoningParser={(v) => updateDeployConfig('reasoningParser', v)}
            setStartTokenizerMode={(v) => updateDeployConfig('tokenizerMode', v)}
            setStartTrustRemoteCode={(v) => updateDeployConfig('trustRemoteCode', v)}
            setStartBindIP={(v) => updateDeployConfig('bindIP', v)}
            setStartPort={(v) => updateDeployConfig('port', v)}
            startBlockSize={startBlockSize}
            startMoeBackend={startMoeBackend}
            startEnableExpertParallel={startEnableExpertParallel}
            startSpeculativeConfig={startSpeculativeConfig}
            startGpus={startGpus}
            startKvCacheDType={startKvCacheDType}
            startMaxLen={startMaxLen}
            startMaxNumSeqs={startMaxNumSeqs}
            startMem={startMem}
            startModelName={startModelName}
            startParser={startParser}
            startReasoningParser={startReasoningParser}
            startTokenizerMode={startTokenizerMode}
            startTrustRemoteCode={startTrustRemoteCode}
            startBindIP={startBindIP}
            startPort={startPort}
            useServerApiKey={startUseServerApiKey}
            useCustomServerApiKey={startUseCustomServerApiKey}
            blockSizeOptions={BLOCK_SIZE_OPTIONS}
            moeBackendOptions={MOE_BACKEND_OPTIONS}
            reasoningParserOptions={REASONING_PARSER_OPTIONS}
            toggleGpu={toggleGpu}
            tokenizerModeOptions={TOKENIZER_MODE_OPTIONS}
            toolParserOptions={launchConfig?.tool_parser_options}
          />
        </ConfirmModal>
      )}

      {launchConfirmData && (
        <ConfirmModal
          title="Review Deploy Command"
          confirmLabel="Deploy Server"
          contentClassName="deploy-command-modal"
          loading={launching}
          onCancel={() => {
            setLaunchConfirmData(null);
            setDeployModalOpen(true);
          }}
          onConfirm={executeStart}
        >
          <div className="deploy-command-summary">
            <div>
              <span className="command-summary-label">Model</span>
              <span>{launchConfirmData.payload.model}</span>
            </div>
            <div>
              <span className="command-summary-label">GPUs</span>
              <span>{launchConfirmData.gpuIds === 'all' ? 'All selected GPUs' : launchConfirmData.gpuIds}</span>
            </div>
            <div>
              <span className="command-summary-label">Port</span>
              <span>{launchConfirmData.payload.port}</span>
            </div>
          </div>
          <pre className="deploy-command-preview">
            <code>{launchConfirmData.command}</code>
          </pre>
        </ConfirmModal>
      )}

      {downloadConfirmData && (
        <ConfirmModal
          title="Confirm Download"
          confirmLabel="Start Download"
          loading={downloading}
          onCancel={() => setDownloadConfirmData(null)}
          onConfirm={executeDownload}
        >
          <p>
            Do you want to proceed with downloading <strong>{downloadConfirmData.model}</strong>?
          </p>
          {downloadConfirmData.sizeText && <p className="modal-emphasis">{downloadConfirmData.sizeText}</p>}
        </ConfirmModal>
      )}

      {removeConfirmId && (
        <ConfirmModal
          title="Remove Container?"
          confirmLabel="Remove"
          danger
          loading={removing}
          onCancel={() => setRemoveConfirmId(null)}
          onConfirm={handleRemove}
        >
          <p>
            Are you sure you want to permanently remove this container? This cannot be undone.
          </p>
        </ConfirmModal>
      )}

      {deleteCacheConfirmModel && (
        <ConfirmModal
          title="Delete Downloaded Model?"
          confirmLabel="Delete Model"
          danger
          loading={deletingCache}
          onCancel={() => setDeleteCacheConfirmModel(null)}
          onConfirm={handleDeleteDownloadedModel}
        >
          <p>
            Delete <strong>{deleteCacheConfirmModel.name}</strong> from the configured Hugging Face cache?
          </p>
          <p className="modal-emphasis">This removes the cached model directory ({deleteCacheConfirmModel.size}). Containers that reference this model must be removed first.</p>
        </ConfirmModal>
      )}

    </div>
  );
};
