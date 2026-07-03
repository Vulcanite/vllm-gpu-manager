import { Play, RefreshCw, Server, Trash2 } from 'lucide-react';

const formatGPUIds = (model) => {
  const gpuIds = String(model.gpu_ids || '').trim();
  if (!gpuIds) return '-';
  if (gpuIds === 'all') return 'All';
  return gpuIds
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean)
    .map((id) => `GPU ${id}`)
    .join(', ');
};

const formatAPIKeyMode = (model) => {
  if (model.api_key_mode === 'custom' || model.has_server_api_key) return 'Custom';
  if (model.api_key_mode === 'global') return 'Global';
  return 'None';
};

export const ActiveContainersCard = ({ activeModels, handleRestart, removing, removeError, requestRemove, restartingId, restartError, proxyEnabled = false }) => {
  const columnCount = proxyEnabled ? 8 : 9;

  return (
    <div className="card active-containers-card">
      <div className="card-header">
        <h2 className="card-title">
          <Server size={20} /> Existing Model Servers
        </h2>
      </div>
      {restartError && <div className="alert alert-danger">{restartError}</div>}
      {removeError && <div className="alert alert-danger">{removeError}</div>}
      <div className="active-containers-table-wrap">
        <table className="active-containers-table">
          <thead>
            <tr>
              <th scope="col">Model Name</th>
              <th scope="col">GPU ID</th>
              <th scope="col">Port</th>
              <th scope="col">Max Mem Util</th>
              <th scope="col">Max Context Length</th>
              <th scope="col">Max Num Seqs</th>
              <th scope="col">Tool Call</th>
              {!proxyEnabled && <th scope="col">API Key</th>}
              <th scope="col">Actions</th>
            </tr>
          </thead>
          <tbody>
            {activeModels.map((model) => {
              const active = ['running', 'restarting', 'paused'].includes(model.state);
              const isRestarting = restartingId === model.container_id;
              const actionLabel = model.state === 'running' ? 'Restart' : 'Start';
              return (
                <tr key={model.container_id}>
                  <td className="active-container-model">{model.model}</td>
                  <td>{formatGPUIds(model)}</td>
                  <td>{model.port}</td>
                  <td>{model.mem_util || '-'}</td>
                  <td>{model.max_len || '-'}</td>
                  <td>{model.max_num_seqs || 'Default'}</td>
                  <td>{model.tool_parser || 'Disabled'}</td>
                  {!proxyEnabled && <td>{formatAPIKeyMode(model)}</td>}
                  <td>
                    <div className="active-container-actions">
                      <button className="btn btn-outline" type="button" onClick={() => handleRestart(model.container_id)} disabled={isRestarting}>
                        {actionLabel === 'Restart' ? <RefreshCw size={14} /> : <Play size={14} />}
                        {isRestarting ? 'Restarting...' : actionLabel}
                      </button>
                      <button className="btn btn-danger" type="button" onClick={() => requestRemove(model.container_id)} disabled={removing || active} title={active ? 'Stop this container before removing it.' : ''}>
                        <Trash2 size={14} /> Remove
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {activeModels.length === 0 && (
              <tr>
                <td className="active-containers-empty" colSpan={columnCount}>
                  No existing configurations found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
