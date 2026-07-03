import { CheckCircle2, Download } from 'lucide-react';

import { DownloadProgressPanel } from '../DownloadProgressPanel.jsx';

const findCachedMatch = (downloadName, downloadedModels) => {
  if (!downloadName || !Array.isArray(downloadedModels)) return null;
  const target = downloadName.trim().toLowerCase();
  if (!target) return null;
  return downloadedModels.find((model) => model?.name?.toLowerCase() === target) || null;
};

export const DownloadModelCard = ({
  activeDownloads,
  checkingInfo,
  downloadError,
  downloading,
  downloadMessage,
  downloadName,
  downloadedModels = [],
  handleDownload,
  setActiveDownloads,
  setDownloadName,
}) => {
  const cachedMatch = findCachedMatch(downloadName, downloadedModels);
  const downloadDisabled = checkingInfo || downloading || Boolean(cachedMatch);

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
      {cachedMatch && (
        <div className="alert alert-info">
          <CheckCircle2 size={18} aria-hidden="true" />
          <span>
            <strong>{cachedMatch.name}</strong> is already downloaded ({cachedMatch.size}).
          </span>
        </div>
      )}
      <form onSubmit={handleDownload}>
        <div className="form-group">
          <label className="form-label" htmlFor="downloadName">Hugging Face Model ID</label>
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
            <button type="submit" className="btn btn-primary" disabled={downloadDisabled}>
              <Download size={18} />
              {checkingInfo ? 'Checking...' : cachedMatch ? 'Already Downloaded' : 'Download'}
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
