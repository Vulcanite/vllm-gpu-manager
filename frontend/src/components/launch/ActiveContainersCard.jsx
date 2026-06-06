import { Server } from 'lucide-react';

export const ActiveContainersCard = ({ activeModels, handleRestart, removeError, restartingId, restartError }) => {
  return (
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
              GPUs: {model.gpus} | Mem: {model.mem_util} | MaxLen: {model.max_len} | KV: {model.kv_cache_dtype || 'auto'} | Parser: {model.tool_parser}
            </div>
            <div className="model-actions-row">
              <button className="btn btn-outline" type="button" onClick={() => handleRestart(model.container_id)} disabled={restartingId === model.container_id}>
                {restartingId === model.container_id ? 'Restarting...' : model.state === 'running' ? 'Restart' : 'Start'}
              </button>
            </div>
          </div>
        ))}
        {activeModels.length === 0 && <p className="text-secondary text-sm">No existing configurations found.</p>}
      </div>
    </div>
  );
};
