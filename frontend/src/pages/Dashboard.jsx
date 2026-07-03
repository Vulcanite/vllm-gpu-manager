import { useMemo } from 'react';
import { Cpu, Server, Square, FileText, ListTree } from 'lucide-react';
import { Link } from 'react-router-dom';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { formatBytes, getMemoryColor } from '../lib/format.js';

// Build a map from GPU ID -> list of ACTIVE (running) containers pinned to that GPU.
// Only running models are shown; inactive containers are excluded.
// Returns map of gpuId -> [{ model, port, pid }]
const buildPinningByGPU = (activeModels, gpus) => {
  const map = new Map();
  
  // Build a map of process name -> pid for quick lookup
  const pidMap = new Map();
  for (const gpu of gpus) {
    const processes = Array.isArray(gpu.processes) ? gpu.processes : [];
    for (const proc of processes) {
      const name = (proc.name || '').toLowerCase();
      if (name) {
        pidMap.set(name, proc.pid);
      }
    }
  }
  
  for (const model of activeModels) {
    // Only show running models
    if (model.state !== 'running') continue;
    if (!model.gpu_ids) continue;
    for (const id of String(model.gpu_ids)
      .split(',')
      .map((part) => Number(part.trim()))
      .filter((id) => Number.isInteger(id))) {
      if (!map.has(id)) map.set(id, []);
      
      // Try to find PID by matching model name to process name
      const modelName = (model.model || '').toLowerCase();
      let pid = null;
      for (const [procName, procPid] of pidMap) {
        if (modelName.includes(procName) || procName.includes(modelName.split('/').pop())) {
          pid = procPid;
          break;
        }
      }
      
      map.get(id).push({
        model: model.model,
        port: model.port,
        pid: pid,
      });
    }
  }
  return map;
};

export const Dashboard = ({
  gpus,
  activeModels,
  stopConfirmPort,
  stopError,
  stopping,
  onCancelStop,
  onConfirmStop,
  onRequestStop,
}) => {
  const runningModels = activeModels.filter((model) => model.state === 'running');
  const pinningByGPU = useMemo(() => buildPinningByGPU(activeModels, gpus), [activeModels, gpus]);
  const gpuProcesses = useMemo(() => (
    gpus.flatMap((gpu) => {
      const processes = Array.isArray(gpu.processes) ? gpu.processes : [];
      return processes.map((proc) => ({
        gpuId: gpu.id,
        pid: proc.pid,
        name: proc.name || 'unknown',
        memory: proc.used_memory_human || (proc.used_memory ? formatBytes(proc.used_memory) : '0 B'),
      }));
    })
  ), [gpus]);

  return (
    <div className="dashboard-grid dashboard-overview-grid">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Cpu size={18} /> Hardware
          </h2>
        </div>
        <div className="gpu-list">
          {gpus.map((gpu) => {
            const pinned = pinningByGPU.get(gpu.id) || [];
            return (
              <div key={gpu.id} className="gpu-item">
                <div className="gpu-header">
                  <span className="gpu-name">{gpu.name}</span>
                  <span className="gpu-id">GPU {gpu.id}</span>
                </div>

                <div className="metric-row">
                  <span>Memory</span>
                  <span className="metric-value">
                    {formatBytes(gpu.memory_used)} / {formatBytes(gpu.memory_total)}
                  </span>
                </div>
                <div className="progress-bar-bg">
                  <div className={`progress-bar-fill ${getMemoryColor(gpu.memory_percent)}`} style={{ width: `${gpu.memory_percent}%` }} />
                </div>

                <div className="metric-row">
                  <span>Compute</span>
                  <span className="metric-value">{gpu.gpu_utilization}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className={`progress-bar-fill ${getMemoryColor(gpu.gpu_utilization)}`} style={{ width: `${gpu.gpu_utilization}%` }} />
                </div>

                {pinned.length > 0 && (
                  <div className="model-meta mt-2">
                    <span>PINNED BY</span>
                    {pinned.map((item) => (
                      <span key={`${item.model}-${item.port}`}>
                        {item.model} |
                      </span>
                    ))}
                  </div>
                )}

              </div>
            );
          })}
          {gpus.length === 0 && <p className="text-secondary text-sm">No GPUs detected.</p>}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Server size={18} /> Active Deployments
          </h2>
          <span className="badge badge-active">{runningModels.length} Active</span>
        </div>
        {stopError && <div className="alert alert-danger">{stopError}</div>}
        <div className="model-list">
          {runningModels.map((model) => (
            <div key={model.container_id} className="model-item">
              <div className="model-info">
                <div className="model-title-row">
                  <span className="model-name">{model.model}</span>
                  {model.is_ready ? (
                    <span className="badge badge-ready">Ready</span>
                  ) : (
                    <span className="badge badge-starting">Starting...</span>
                  )}
                </div>
                <div className="deployment-meta mt-4">
                  <span className="deployment-meta-item"><span>PORT</span>{model.port}</span>
                  <span className="deployment-meta-item"><span>GPUs</span>{model.gpus}</span>
                  <span className="deployment-meta-item"><span>ARCH</span>{model.arch}</span>
                  <span className="deployment-meta-item"><span>CTX</span>{model.max_len}</span>
                  <span className="deployment-meta-item"><span>SEQS</span>{model.max_num_seqs || 'default'}</span>
                  <span className="deployment-meta-item"><span>KV</span>{model.kv_cache_dtype || 'auto'}</span>
                </div>
                <div className="model-meta model-url mt-2"><span>API URL :</span>{model.api_url}</div>
              </div>
              <div className="model-actions">
                <Link to={`/logs?container_id=${encodeURIComponent(model.container_id)}`} className="btn btn-outline btn-link-clean" style={{ padding: '6px 10px', marginRight: '6px' }}>
                  <FileText size={14} /> Logs
                </Link>
                <button onClick={() => onRequestStop(model.port)} className="btn btn-danger" disabled={stopping}>
                  <Square size={14} /> Stop
                </button>
              </div>
            </div>
          ))}
          {runningModels.length === 0 && (
            <div className="empty-state">
              <p className="text-sm">No models currently running.</p>
            </div>
          )}
        </div>
      </div>

      <div className="card full-span">
        <div className="card-header">
          <h2 className="card-title">
            <ListTree size={18} /> Processes
          </h2>
          <span className="badge badge-active">{gpuProcesses.length} Running</span>
        </div>
        {gpuProcesses.length > 0 ? (
          <div className="process-table-wrap">
            <div className="process-table">
              <div className="process-row process-table-head">
                <span>GPU ID</span>
                <span>PID</span>
                <span>Process Name</span>
                <span>Mem Used</span>
              </div>
              {gpuProcesses.map((proc) => (
                <div key={`${proc.gpuId}-${proc.pid}-${proc.name}`} className="process-row">
                  <span>GPU {proc.gpuId}</span>
                  <span>{proc.pid}</span>
                  <span className="process-name-cell">{proc.name}</span>
                  <span>{proc.memory}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <p className="text-sm">No GPU processes detected.</p>
          </div>
        )}
      </div>

      {stopConfirmPort && (
        <ConfirmModal
          title="Stop Server?"
          confirmLabel="Stop Server"
          danger
          loading={stopping}
          onCancel={onCancelStop}
          onConfirm={onConfirmStop}
        >
          <p>
            Are you sure you want to stop the model running on port <strong>{stopConfirmPort}</strong>? This will kill the container and free up GPU memory.
          </p>
        </ConfirmModal>
      )}
    </div>
  );
};
