import { useEffect, useMemo, useState } from 'react';
import { Download, Loader2, Play, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { getErrorDetail, getRecommendations } from '../lib/api.js';

const USE_CASES = ['general', 'coding', 'reasoning', 'chat', 'vision'];

const fitBadgeClass = (fit) => {
  if (fit === 'perfect' || fit === 'good') return 'badge-ready';
  if (fit === 'marginal') return 'badge-starting';
  return 'badge-active';
};

export const RecommendedModels = ({ downloadedModels = [] }) => {
  const navigate = useNavigate();
  const [useCase, setUseCase] = useState('general');
  const [fitOnly, setFitOnly] = useState(false);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const downloaded = useMemo(() => new Set(downloadedModels.map((model) => model.name)), [downloadedModels]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    getRecommendations({ useCase: 'general', fitOnly: false, limit: 200 })
      .then((res) => {
        if (!cancelled) setModels(res.data?.models || []);
      })
      .catch((err) => {
        if (!cancelled) setError(getErrorDetail(err, 'Unable to load model recommendations'));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleModels = useMemo(() => {
    return models.filter((model) => {
      const matchesUseCase = useCase === 'general' || (model.use_cases || []).includes(useCase);
      const matchesFit = !fitOnly || model.fit_level !== 'too_tight';
      return matchesUseCase && matchesFit;
    });
  }, [fitOnly, models, useCase]);

  const openLaunch = (model) => {
    const params = new URLSearchParams({
      model: model.name,
      gpus: String(model.recommended_gpus || 1),
      gpu_ids: model.recommended_gpu_ids || '',
      mem_util: model.recommended_mem_util || '0.9',
      max_len: model.recommended_max_len ? String(model.recommended_max_len) : '',
      kv_cache_dtype: model.recommended_kv_cache_dtype || 'auto',
    });
    navigate(`/launch?${params.toString()}`);
  };

  const openDownload = (model) => {
    navigate(`/launch?download=${encodeURIComponent(model.name)}`);
  };

  return (
    <div className="card full-span">
      <div className="card-header recommendation-header">
        <div>
          <h2 className="card-title">
            <Sparkles size={18} /> Recommended Models
          </h2>
          <p className="text-secondary text-sm mt-1">Ranked for the GPUs detected on this server. Download a model before launching it.</p>
        </div>
        <div className="recommendation-toolbar">
          <div className="recommendation-controls" aria-label="Recommendation filters">
            <div className="pill-container recommendation-pill-group" role="group" aria-label="Use case">
              {USE_CASES.map((item) => (
                <button key={item} className={`pill recommendation-pill ${useCase === item ? 'selected' : ''}`} type="button" onClick={() => setUseCase(item)}>
                  {item}
                </button>
              ))}
            </div>
            <button className={`pill recommendation-pill ${fitOnly ? 'selected' : ''}`} type="button" onClick={() => setFitOnly((value) => !value)} aria-pressed={fitOnly}>
              Fit only
            </button>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}
      {loading ? (
        <div className="empty-state"><Loader2 className="loader-inline" size={24} /> Loading recommendations...</div>
      ) : (
        <div className="recommendation-table-wrap">
          <div className="recommendation-table" role="table" aria-label="Recommended models">
            <div className="recommendation-row recommendation-table-head" role="row">
              <span>Model</span>
              <span>Fit</span>
              <span>Params</span>
              <span>Quant</span>
              <span>Rec</span>
              <span>Quality</span>
              <span>Tok/s</span>
              <span>GPUs</span>
              <span>Need/GPU</span>
              <span>Ctx</span>
              <span>Action</span>
            </div>
          {visibleModels.map((model) => {
            const isDownloaded = downloaded.has(model.name);
            return (
              <div key={model.name} className="recommendation-row" role="row">
                <div className="recommendation-model-cell" role="cell">
                  <span className="model-name">{model.name}</span>
                  <span className="recommendation-reason">{model.reason}</span>
                </div>
                <span role="cell"><span className={`badge ${fitBadgeClass(model.fit_level)}`}>{model.fit_level}</span></span>
                <span role="cell">{model.parameter_count}</span>
                <span role="cell">{model.quantization}</span>
                <span role="cell" className="metric-value">{model.score}</span>
                <span role="cell" className="metric-value">{model.model_quality}</span>
                <span role="cell" className="metric-value">~{model.estimated_tokens_sec}</span>
                <span role="cell">{model.recommended_gpus || '-'} <span className="text-secondary">({model.recommended_gpu_ids || '-'})</span></span>
                <span role="cell">{model.required_gb_per_gpu || 0} GiB</span>
                <span role="cell">{model.recommended_max_len || model.context_length}</span>
                <div className="recommendation-actions" role="cell">
                  {isDownloaded ? (
                    <button className="btn btn-primary" type="button" onClick={() => openLaunch(model)} disabled={model.fit_level === 'too_tight'}>
                      <Play size={14} /> Launch
                    </button>
                  ) : (
                    <button className="btn btn-outline" type="button" onClick={() => openDownload(model)}>
                      <Download size={14} /> Download
                    </button>
                  )}
                </div>
              </div>
            );
          })}
          </div>
          {visibleModels.length === 0 && <div className="empty-state">No recommendations match this filter.</div>}
        </div>
      )}
    </div>
  );
};
