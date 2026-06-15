import { Download } from 'lucide-react';

import { DownloadProgressPanel } from '../DownloadProgressPanel.jsx';

export const DownloadModelCard = ({
  activeDownloads,
  checkingInfo,
  downloadError,
  downloading,
  downloadMessage,
  downloadName,
  handleDownload,
  setActiveDownloads,
  setDownloadName,
}) => {
  return (
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
          <label className="form-label" htmlFor="downloadName">HuggingFace Model ID</label>
          <div className="inline-control">
            <input
              id="downloadName"
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
      {activeDownloads.map((activeDownload) => (
        <DownloadProgressPanel
          key={activeDownload.name}
          modelName={activeDownload.name}
          expectedSize={activeDownload.expectedSize}
          onDismiss={() => setActiveDownloads((prev) => prev.filter((item) => item.name !== activeDownload.name))}
        />
      ))}
    </div>
  );
};
