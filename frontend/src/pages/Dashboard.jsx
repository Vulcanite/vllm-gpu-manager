import { Cpu, Server, Square } from 'lucide-react';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { formatBytes, getMemoryColor } from '../lib/format.js';

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

  return (
    <div className="dashboard-grid">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Cpu size={18} /> Hardware
          </h2>
        </div>
        <div className="gpu-list">
          {gpus.map((gpu) => (
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
            </div>
          ))}
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
                <div className="model-meta mt-2">
                  <span>PORT {model.port}</span>
                  <span>GPUs {model.gpus}</span>
                  <span>ARCH {model.arch}</span>
                </div>
                <div className="model-meta mt-1">
                  <span>CTX {model.max_len}</span>
                  <span>PARSER {model.tool_parser || 'none'}</span>
                </div>
                <div className="model-meta model-url mt-2">{model.api_url}</div>
              </div>
              <div className="model-actions">
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
