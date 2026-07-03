import { Play, RefreshCw } from 'lucide-react';

import { generateVLLMToken } from '../../lib/tokens.js';

export const DeployServerCard = ({
  downloadedModels,
  gpuList,
  gpuUnavailableMessage,
  handleGpuCountChange,
  handleStart,
  hasGlobalVllmAPIKey,
  kvCacheDTypeOptions,
  launchError,
  launching,
  maxGpus,
  noGpuDetected,
  onClearGpuPinning,
  proxyEnabled = false,
  scaleMaxLen,
  selectedGpuIds,
  serverApiKey,
  setServerApiKey,
  setUseServerApiKey,
  setUseCustomServerApiKey,
  setStartKvCacheDType,
  setStartMaxLen,
  setStartMaxNumSeqs,
  setStartMem,
  setStartModelName,
  setStartParser,
  setStartPort,
  startGpus,
  startKvCacheDType,
  startMaxLen,
  startMaxNumSeqs,
  startMem,
  startModelName,
  startParser,
  startPort,
  useServerApiKey,
  useCustomServerApiKey,
  toggleGpu,
  toolParserOptions,
  usesAllGpus,
}) => {
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

  return (
    <div className="card launch-deploy-card">
      <div className="card-header">
        <h2 className="card-title">
          <Play size={18} /> Deploy Server
        </h2>
      </div>
      {launchError && <div className="alert alert-danger">{launchError}</div>}
      {noGpuDetected && !launchError && <div className="alert alert-danger">{gpuUnavailableMessage}</div>}
      <form onSubmit={handleStart}>
        <div className="form-group">
          <label className="form-label" htmlFor="startModelName">Downloaded Model</label>
          <select id="startModelName" className="form-control" value={startModelName} onChange={(event) => setStartModelName(event.target.value)} required>
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
            <label className="form-label" htmlFor="startPort">Port</label>
            <input id="startPort" type="number" className="form-control" value={startPort} onChange={(event) => setStartPort(event.target.value)} required />
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="startGpus">GPUs</label>
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
          </div>
        </div>

        <div className={usesAllGpus || maxGpus <= 1 ? 'form-group' : 'form-grid'}>
          {!usesAllGpus && maxGpus > 1 && (
            <div className="form-group">
              <label className="form-label">GPU Pinning</label>
              <div className="pill-container">
                {noGpuDetected ? (
                  <span className="text-secondary text-sm">No GPUs detected</span>
                ) : usesAllGpus ? (
                  <button className="pill selected" type="button" onClick={onClearGpuPinning}>
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
          )}
          <div className="form-group">
            <label className="form-label" htmlFor="startMem">Memory Utilization</label>
            <input id="startMem" type="text" className="form-control" value={startMem} onChange={(event) => setStartMem(event.target.value)} required />
          </div>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label" htmlFor="startMaxLen">Max Context Length</label>
            <div className="inline-control max-len-control">
              <button className="btn btn-outline btn-compact" type="button" onClick={() => scaleMaxLen(0.5)}>
                /2
              </button>
              <input id="startMaxLen" type="number" className="form-control" value={startMaxLen} onChange={(event) => setStartMaxLen(event.target.value)} min="1" />
              <button className="btn btn-outline btn-compact" type="button" onClick={() => scaleMaxLen(2)}>
                x2
              </button>
            </div>
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="startKvCacheDType">KV Cache DType</label>
            <select id="startKvCacheDType" className="form-control" value={startKvCacheDType} onChange={(event) => setStartKvCacheDType(event.target.value)}>
              {kvCacheDTypeOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label" htmlFor="startMaxNumSeqs">Max Num Seqs</label>
            <input
              id="startMaxNumSeqs"
              type="number"
              className="form-control"
              value={startMaxNumSeqs}
              onChange={(event) => setStartMaxNumSeqs(event.target.value)}
              min="1"
              placeholder="vLLM default"
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="startParser">Tool Call Parser</label>
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
        </div>

        {hasGlobalVllmAPIKey && !proxyEnabled && (
          <div className="form-group">
            <div className="setting-row launch-setting-row">
              <div>
                <label className="form-label">API Key Authentication</label>
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
                    <label className="form-label">Custom API Key</label>
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
                    <div className="inline-control token-generator-control mt-2">
                      <input
                        id="startServerApiKey"
                        type="text"
                        className="form-control"
                        value={serverApiKey || ''}
                        onChange={(event) => setServerApiKey(event.target.value)}
                        placeholder=""
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
          {launching ? 'Deploying...' : 'Deploy Server'}
        </button>
      </form>
    </div>
  );
};
