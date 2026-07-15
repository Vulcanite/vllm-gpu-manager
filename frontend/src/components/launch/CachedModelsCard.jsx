import { ExternalLink, HardDrive, Trash2 } from 'lucide-react';

const huggingFaceModelUrl = (modelName) => {
  const parts = String(modelName || '')
    .split('/')
    .map((part) => encodeURIComponent(part));
  return `https://huggingface.co/${parts.join('/')}`;
};

export const CachedModelsCard = ({ deletingCache, downloadedModels, requestDeleteDownloadedModel }) => {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">
          <HardDrive size={20} /> Downloaded Models
        </h2>
      </div>
      <div className="model-list compact-list">
        {downloadedModels.map((model) => (
          <div key={model.name} className="model-item compact-model-item cache-model-item">
            <a
              className="model-name model-name-small cache-model-name cache-model-link"
              href={huggingFaceModelUrl(model.name)}
              target="_blank"
              rel="noreferrer"
              aria-label={`Open ${model.name} on Hugging Face`}
            >
              <span>{model.name}</span>
              <ExternalLink className="model-link-icon" size={13} aria-hidden="true" />
            </a>
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
  );
};
