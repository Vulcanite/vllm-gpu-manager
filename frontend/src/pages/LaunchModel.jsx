import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, HardDrive, Play, Server, Trash2 } from 'lucide-react';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { DownloadProgressPanel } from '../components/DownloadProgressPanel.jsx';
import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { deleteDownloadedModel, downloadModel, getErrorDetail, getModelInfo, removeModel, restartModel, startModel } from '../lib/api.js';
import { formatBytes } from '../lib/format.js';

const MODEL_VRAM_SAFETY_MARGIN = 1.08;
const DEFAULT_GPU_MEMORY_UTILIZATION = 0.9;
const ACTIVE_DOWNLOAD_STORAGE_KEY = 'vllm-manager-active-download';

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

export const LaunchModel = ({ gpus, gpuError = '', downloadedModels, activeModels, fetchData, notify }) => {
  const [startModelName, setStartModelName] = useState('');
  const [startPort, setStartPort] = useState(8080);
  const [startGpus, setStartGpus] = useState(1);
  const [startGpuIds, setStartGpuIds] = useState([]);
  const [startMem, setStartMem] = useState(0.9);
  const [startMaxLen, setStartMaxLen] = useState('');
  const [startParser, setStartParser] = useState('');
  const [launchError, setLaunchError] = useState('');
  const [launching, setLaunching] = useState(false);

  const [downloadConfirmData, setDownloadConfirmData] = useState(null);
  const [downloadName, setDownloadName] = useState('');
  const [downloadMessage, setDownloadMessage] = useState('');
  const [downloadError, setDownloadError] = useState('');
  const [checkingInfo, setCheckingInfo] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [activeDownload, setActiveDownload] = useState(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DOWNLOAD_STORAGE_KEY);
      const parsed = stored ? JSON.parse(stored) : null;
      return parsed && typeof parsed.name === 'string' ? parsed : null;
    } catch {
      return null;
    }
  });
  const [restartError, setRestartError] = useState('');
  const [restartingId, setRestartingId] = useState('');
  const [removeConfirmId, setRemoveConfirmId] = useState(null);
  const [removeError, setRemoveError] = useState('');
  const [removing, setRemoving] = useState(false);
  const [deleteCacheConfirmModel, setDeleteCacheConfirmModel] = useState(null);
  const [deleteCacheError, setDeleteCacheError] = useState('');
  const [deletingCache, setDeletingCache] = useState(false);

  const navigate = useNavigate();
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
  useAutoDismiss(deleteCacheError, setDeleteCacheError);

  useEffect(() => {
    try {
      if (activeDownload) {
        localStorage.setItem(ACTIVE_DOWNLOAD_STORAGE_KEY, JSON.stringify(activeDownload));
      } else {
        localStorage.removeItem(ACTIVE_DOWNLOAD_STORAGE_KEY);
      }
    } catch {
      // Progress persistence is best-effort; polling still works without storage.
    }
  }, [activeDownload]);

  const getKVCacheBytesPerGpu = (selectedModel, requestedGpus, maxModelLen) => {
    if (!selectedModel || !requestedGpus || !maxModelLen) return 0;

    const layers = selectedModel.kv_cache_layers || 0;
    const keyValueHeads = selectedModel.kv_cache_key_value_heads || 0;
    const headDim = selectedModel.kv_cache_head_dim || 0;
    const dtypeBytes = selectedModel.kv_cache_dtype_bytes || 0;

    if (layers > 0 && keyValueHeads > 0 && headDim > 0 && dtypeBytes > 0) {
      const keyValueHeadsPerGpu = Math.max(1, Math.ceil(keyValueHeads / requestedGpus));
      const bytesPerTokenPerGpu = 2 * layers * keyValueHeadsPerGpu * headDim * dtypeBytes;
      return bytesPerTokenPerGpu * maxModelLen;
    }

    const bytesPerToken = selectedModel.kv_cache_bytes_per_token || 0;
    return bytesPerToken > 0 ? (bytesPerToken / requestedGpus) * maxModelLen : 0;
  };

  const vramPreflightError = useMemo(() => {
    if (noGpuDetected) return '';

    const selectedModel = downloadedModels.find((m) => m.name === startModelName);
    const modelBytes = selectedModel?.size_bytes || 0;
    const requestedGpus = Number(startGpus);
    if (!modelBytes || !requestedGpus || requestedGpus < 1) return '';

    const memUtil = startMem ? Number(startMem) : DEFAULT_GPU_MEMORY_UTILIZATION;
    if (Number.isNaN(memUtil) || memUtil <= 0 || memUtil > 1.0) return '';

    const selectedGpus = (() => {
      if (selectedGpuIds.length > 0) {
        return gpuList.filter((gpu) => selectedGpuIds.includes(gpu.id)).sort((a, b) => a.id - b.id);
      }
      return [...gpuList].sort((a, b) => a.id - b.id).slice(0, requestedGpus);
    })();
    if (selectedGpus.length !== requestedGpus) return '';

    const modelBytesPerGpu = (modelBytes / requestedGpus) * MODEL_VRAM_SAFETY_MARGIN;
    const effectiveMaxLen = (() => {
      if (startMaxLen) {
        const parsed = Number(startMaxLen);
        return Number.isInteger(parsed) && parsed > 0 ? parsed : 0;
      }
      return selectedModel?.default_max_model_len || 0;
    })();
    const kvCacheBytesPerGpu = getKVCacheBytesPerGpu(selectedModel, requestedGpus, effectiveMaxLen);
    const requiredPerGpu = modelBytesPerGpu + kvCacheBytesPerGpu;
    const limitingGpu = selectedGpus.find((gpu) => {
      const utilizationBudget = gpu.memory_total * memUtil;
      const availableBudget = Math.min(gpu.memory_free, utilizationBudget);
      return requiredPerGpu > availableBudget;
    });

    if (!limitingGpu) return '';

    const availableBudget = Math.min(limitingGpu.memory_free, limitingGpu.memory_total * memUtil);
    if (kvCacheBytesPerGpu > 0) {
      return `${selectedModel.name} needs about ${formatBytes(requiredPerGpu)} per GPU (${formatBytes(modelBytesPerGpu)} model shard plus ${formatBytes(kvCacheBytesPerGpu)} KV cache at max_model_len ${effectiveMaxLen}), but GPU ${limitingGpu.id} only has ${formatBytes(availableBudget)} available within the ${memUtil.toFixed(2)} memory utilization budget.`;
    }
    return `${selectedModel.name} needs about ${formatBytes(requiredPerGpu)} per GPU after margin, but GPU ${limitingGpu.id} only has ${formatBytes(availableBudget)} available within the ${memUtil.toFixed(2)} memory utilization budget.`;
  }, [noGpuDetected, startModelName, startGpus, selectedGpuIds, startMem, startMaxLen, gpuList, downloadedModels]);

  const handleGpuCountChange = (event) => {
    if (noGpuDetected) {
      setStartGpuIds([]);
      return;
    }

    const rawValue = event.target.value;
    if (rawValue === '') {
      setStartGpus('');
      setStartGpuIds([]);
      return;
    }

    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;

    const nextCount = Math.min(maxGpus, Math.max(1, Math.trunc(parsed)));
    const nextUsesAll = nextCount >= maxGpus;
    setStartGpus(nextCount);
    setStartGpuIds((current) => normalizeGpuPinning(current, availableGpuIds, nextCount, nextUsesAll));
  };

  const toggleGpu = (id) => {
    if (usesAllGpus) return;
    if (selectedGpuIds.includes(id)) return;

    setStartGpuIds([...selectedGpuIds.slice(0, requestedGpuCount - 1), id]);
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

    if (startParser && !/^[a-zA-Z0-9-]+$/.test(startParser)) return 'Tool parser must contain only letters, numbers, and hyphens';
    if (!usesAllGpus && selectedGpuIds.length !== gpusToSend) return `Select exactly ${gpusToSend} GPU${gpusToSend === 1 ? '' : 's'} for pinning.`;
    if (vramPreflightError) return vramPreflightError;
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
    const payload = {
      model: startModelName,
      port: Number(startPort),
      gpus: Number(startGpus),
      gpu_ids: gpuIdsParam,
      mem_util: startMem !== null && startMem !== undefined ? startMem.toString() : '',
      max_len: startMaxLen ? startMaxLen.toString() : '',
      tool_parser: startParser || '',
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
    setDeleteCacheError('');
    setDeleteCacheConfirmModel(model);
  };

  const handleDeleteDownloadedModel = async () => {
    if (!deleteCacheConfirmModel) return;

    const modelName = deleteCacheConfirmModel.name;
    setDeleteCacheError('');
    setDeletingCache(true);
    try {
      await deleteDownloadedModel(modelName);
      setDeleteCacheConfirmModel(null);
      if (activeDownload?.name === modelName) {
        setActiveDownload(null);
      }
      notify(`Deleted ${modelName} from cache.`);
      fetchData();
    } catch (error) {
      setDeleteCacheError(getErrorDetail(error, 'Failed to delete downloaded model'));
      setDeleteCacheConfirmModel(null);
    } finally {
      setDeletingCache(false);
    }
  };

  const handleDownload = async (event) => {
    event.preventDefault();
    setDownloadMessage('');
    setDownloadError('');
    setCheckingInfo(true);

    let sizeText = '';
    let sizeBytes = 0;
    try {
      const infoRes = await getModelInfo(downloadName);
      if (infoRes.data?.size_bytes) {
        sizeBytes = infoRes.data.size_bytes;
        sizeText = `Estimated file size: ${formatBytes(infoRes.data.size_bytes)}.`;
      }
    } catch {
      sizeText = 'Model size could not be fetched before download.';
    } finally {
      setCheckingInfo(false);
    }

    setDownloadConfirmData({ model: downloadName, sizeText, sizeBytes });
  };

  const executeDownload = async () => {
    if (!downloadConfirmData) return;
    setDownloadError('');
    setDownloading(true);
    try {
      const res = await downloadModel(downloadConfirmData.model);
      setActiveDownload({
        containerId: res.data?.container_id || '',
        name: downloadConfirmData.model,
        expectedSize: downloadConfirmData.sizeBytes,
      });
      setDownloadMessage('Download started. Progress will update as files appear in the cache.');
      setDownloadName('');
      setDownloadConfirmData(null);
      fetchData();
    } catch (error) {
      setDownloadError(getErrorDetail(error, 'Failed to trigger download'));
      setDownloadConfirmData(null);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="dashboard-grid">
      <div className="stack">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Play size={18} /> Deploy Server
            </h2>
          </div>
          {launchError && <div className="alert alert-danger">{launchError}</div>}
          {noGpuDetected && !launchError && <div className="alert alert-danger">{gpuUnavailableMessage}</div>}
          {vramPreflightError && !launchError && <div className="alert alert-danger">{vramPreflightError}</div>}
          <form onSubmit={handleStart}>
            <div className="form-group">
              <label className="form-label">Downloaded Model</label>
              <select className="form-control" value={startModelName} onChange={(event) => setStartModelName(event.target.value)} required>
                <option value="" disabled>
                  Select a model
                </option>
                {downloadedModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({model.size})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Port</label>
                <input type="number" className="form-control" value={startPort} onChange={(event) => setStartPort(event.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label">GPUs</label>
                <input
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
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">GPU Pinning</label>
                <div className="pill-container">
                  {noGpuDetected ? (
                    <span className="text-secondary text-sm">No GPUs detected</span>
                  ) : usesAllGpus ? (
                    <button className="pill selected" type="button" onClick={() => setStartGpuIds([])}>
                      All
                    </button>
                  ) : (
                    gpuList.map((gpu) => (
                      <button key={gpu.id} className={`pill ${selectedGpuIds.includes(gpu.id) ? 'selected' : ''}`} type="button" onClick={() => toggleGpu(gpu.id)}>
                        GPU {gpu.id}
                      </button>
                    ))
                  )}
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Memory Utilization</label>
                <input type="text" className="form-control" value={startMem} onChange={(event) => setStartMem(event.target.value)} required />
              </div>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Max Context Length</label>
                <input type="number" className="form-control" value={startMaxLen} onChange={(event) => setStartMaxLen(event.target.value)} placeholder="Model default" />
              </div>
              <div className="form-group">
                <label className="form-label">Tool Call Parser</label>
                <select className="form-control" value={startParser} onChange={(event) => setStartParser(event.target.value)}>
                  <option value="">Disabled</option>
                  <option value="hermes">hermes</option>
                  <option value="llama3_json">llama3_json</option>
                  <option value="mistral">mistral</option>
                </select>
              </div>
            </div>

            <button type="submit" className="btn btn-primary mt-2 full-width" disabled={launching || noGpuDetected || Boolean(vramPreflightError)}>
              {launching ? 'Deploying...' : 'Deploy Server'}
            </button>
          </form>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Server size={20} /> Existing Model Containers
            </h2>
          </div>
          {restartError && <div className="alert alert-danger">{restartError}</div>}
          {removeError && <div className="alert alert-danger">{removeError}</div>}
          <div className="model-list compact-list">
            {activeModels.map((model) => (
              <div key={model.container_id} className="model-item model-item-stacked">
                <div className="model-title-row">
                  <span className="model-name model-name-small">{model.model}</span>
                  <span className="badge badge-active">Port {model.port}</span>
                </div>
                <div className="text-secondary text-sm">
                  GPUs: {model.gpus} | Mem: {model.mem_util} | MaxLen: {model.max_len} | Parser: {model.tool_parser}
                </div>
                <div className="model-actions-row">
                  {model.state !== 'running' && (
                    <button className="btn btn-outline" type="button" onClick={() => handleRestart(model.container_id)} disabled={restartingId === model.container_id}>
                      {restartingId === model.container_id ? 'Restarting...' : 'Restart'}
                    </button>
                  )}
                </div>
              </div>
            ))}
            {activeModels.length === 0 && <p className="text-secondary text-sm">No existing configurations found.</p>}
          </div>
        </div>
      </div>

      <div className="stack">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <Download size={24} /> Download New Model
            </h2>
          </div>
          <p className="text-secondary mb-4 text-sm">Downloads occur in the background via Docker. Progress is estimated from cached bytes on disk.</p>
          {downloadMessage && <div className="alert alert-success">{downloadMessage}</div>}
          {downloadError && <div className="alert alert-danger">{downloadError}</div>}
          <form onSubmit={handleDownload}>
            <div className="form-group">
              <label className="form-label">HuggingFace Model ID</label>
              <div className="inline-control">
                <input
                  type="text"
                  className="form-control"
                  value={downloadName}
                  onChange={(event) => setDownloadName(event.target.value)}
                  placeholder="e.g. meta-llama/Llama-3-8B-Instruct"
                  required
                />
                <button type="submit" className="btn btn-primary" disabled={checkingInfo || downloading}>
                  <Download size={18} />
                  {checkingInfo ? 'Checking...' : 'Download'}
                </button>
              </div>
            </div>
          </form>
          {activeDownload && (
            <DownloadProgressPanel
              key={activeDownload.name}
              modelName={activeDownload.name}
              expectedSize={activeDownload.expectedSize}
              onDismiss={() => setActiveDownload(null)}
            />
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">
              <HardDrive size={20} /> Downloaded Models
            </h2>
          </div>
          {deleteCacheError && <div className="alert alert-danger">{deleteCacheError}</div>}
          <div className="model-list compact-list">
            {downloadedModels.map((model) => (
              <div key={model.name} className="model-item compact-model-item cache-model-item">
                <span className="model-name model-name-small cache-model-name">{model.name}</span>
                <div className="cache-model-actions">
                  <span className="badge badge-active">{model.size}</span>
                  <button className="btn btn-danger btn-compact" type="button" onClick={() => requestDeleteDownloadedModel(model)} disabled={deletingCache}>
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </div>
            ))}
            {downloadedModels.length === 0 && <p className="text-secondary text-sm">No models found in cache.</p>}
          </div>
        </div>
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
