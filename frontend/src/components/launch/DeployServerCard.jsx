import { Play } from 'lucide-react';

export const DeployServerCard = ({
  downloadedModels,
  gpuList,
  gpuUnavailableMessage,
  handleGpuCountChange,
  handleStart,
  kvCacheDTypeOptions,
  launchError,
  launching,
  maxGpus,
  noGpuDetected,
  onClearGpuPinning,
  scaleMaxLen,
  selectedGpuIds,
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
  toggleGpu,
  usesAllGpus,
}) => {
  return (
    <div className="card">
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

        <div className="form-grid">
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
            <option value="hermes">hermes</option>
            <option value="llama3_json">llama3_json</option>
            <option value="mistral">mistral</option>
            <option value="qwen3_coder">qwen3_coder</option>
          </select>
        </div>

        <button type="submit" className="btn btn-primary mt-2 full-width" disabled={launching || noGpuDetected}>
          {launching ? 'Deploying...' : 'Deploy Server'}
        </button>
      </form>
    </div>
  );
};
