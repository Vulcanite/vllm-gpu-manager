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

export const LaunchModel = ({ gpus, gpuError = '', downloadedModels, activeModels, launchConfig, fetchData, notify, vllmAPIKeySet = false, proxyEnabled = false }) => {
  const DEFAULT_MAX_MODEL_LEN = launchConfig?.default_max_model_len ?? DEFAULT_LAUNCH_CONFIG.default_max_model_len;
  const KV_CACHE_DTYPE_OPTIONS = launchConfig?.kv_cache_dtype_options?.length
    ? launchConfig.kv_cache_dtype_options
    : DEFAULT_LAUNCH_CONFIG.kv_cache_dtype_options;

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
  const startUseServerApiKey = deployConfig.useServerApiKey;
  const startUseCustomServerApiKey = deployConfig.useCustomServerApiKey;
  const startServerApiKey = deployConfig.serverApiKey;
  const [launchError, setLaunchError] = useState('');
  const [launching, setLaunching] = useState(false);

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
      }));
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
      use_api_key: effectiveUseServerApiKey,
      server_api_key: effectiveServerApiKey,
    };

    try {
      setLaunching(true);
      await startModel(payload);
      notify('Model starting. It may take a few minutes to be ready.');
      fetchData();
      navigate('/');
    } catch (error) {
      setLaunchError(getErrorDetail(error, 'Failed to start model'));
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
    try {
      const infoRes = await getModelInfo(trimmedName);
      if (infoRes.data?.downloaded) {
        const cached = infoRes.data.cached_bytes;
        const sizeText = typeof cached === 'number' && cached > 0
          ? `${formatBytes(cached)} on disk`
          : 'already on disk';
        setDownloadError(`${trimmedName} is ${sizeText}. Delete it from the cache below if you want to re-download.`);
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

    let sizeText = '';
    let sizeBytes = 0;
    try {
      const infoRes = await getModelInfo(trimmedName);
      if (infoRes.data?.size_bytes) {
        sizeBytes = infoRes.data.size_bytes;
        sizeText = `Estimated file size: ${formatBytes(infoRes.data.size_bytes)}.`;
      }
    } catch {
      sizeText = 'Model size could not be fetched before download.';
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
      <div className="stack launch-deploy-stack">
        <DeployServerCard
          downloadedModels={downloadedModels}
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
          onClearGpuPinning={() => updateDeployConfig('gpuIds', [])}
          selectedGpuIds={selectedGpuIds}
          serverApiKey={startServerApiKey}
          setServerApiKey={(v) => updateDeployConfig('serverApiKey', v)}
          setUseServerApiKey={(v) => updateDeployConfig('useServerApiKey', v)}
          setUseCustomServerApiKey={(v) => updateDeployConfig('useCustomServerApiKey', v)}
          setStartKvCacheDType={(v) => updateDeployConfig('kvCacheDType', v)}
          setStartMaxLen={(v) => updateDeployConfig('maxLen', v)}
          setStartMaxNumSeqs={(v) => updateDeployConfig('maxNumSeqs', v)}
          scaleMaxLen={scaleMaxLen}
          setStartMem={(v) => updateDeployConfig('mem', v)}
          setStartModelName={(v) => updateDeployConfig('modelName', v)}
          setStartParser={(v) => updateDeployConfig('parser', v)}
          setStartPort={(v) => updateDeployConfig('port', v)}
          startGpus={startGpus}
          startKvCacheDType={startKvCacheDType}
          startMaxLen={startMaxLen}
          startMaxNumSeqs={startMaxNumSeqs}
          startMem={startMem}
          startModelName={startModelName}
          startParser={startParser}
          startPort={startPort}
          useServerApiKey={startUseServerApiKey}
          useCustomServerApiKey={startUseCustomServerApiKey}
          toggleGpu={toggleGpu}
          toolParserOptions={launchConfig?.tool_parser_options}
          usesAllGpus={usesAllGpus}
        />
      </div>

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
