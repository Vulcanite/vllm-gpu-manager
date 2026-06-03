import { useEffect, useState } from 'react';

import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { getDownloadProgress } from '../lib/api.js';
import { formatBytes } from '../lib/format.js';

export const DownloadProgressPanel = ({ modelName, expectedSize, onDismiss }) => {
  const [downloadedBytes, setDownloadedBytes] = useState(0);
  const [progressError, setProgressError] = useState('');
  const [showCompletion, setShowCompletion] = useState(true);

  useEffect(() => {
    if (!modelName) return undefined;

    const fetchProgress = async () => {
      try {
        const response = await getDownloadProgress(modelName);
        setDownloadedBytes(response.data?.downloaded_bytes || 0);
        setProgressError('');
      } catch (error) {
        setProgressError(error.response?.data?.detail || error.message || 'Failed to fetch download progress');
      }
    };

    fetchProgress();
    const interval = setInterval(fetchProgress, 2000);
    return () => clearInterval(interval);
  }, [modelName]);

  const hasExpectedSize = expectedSize && expectedSize > 0;
  const percent = hasExpectedSize ? Math.min(100, Math.round((downloadedBytes / expectedSize) * 100)) : 0;
  const isComplete = hasExpectedSize && downloadedBytes >= expectedSize;

  useEffect(() => {
    if (!isComplete || !showCompletion) return undefined;

    const timeout = setTimeout(() => {
      setShowCompletion(false);
    }, 5000);

    return () => clearTimeout(timeout);
  }, [isComplete, showCompletion]);

  useAutoDismiss(progressError, setProgressError);

  return (
    <div className="download-progress-panel" aria-live="polite">
      <div className="download-progress-header">
        <div>
          <div className="model-name model-name-small">{modelName}</div>
          <div className="text-secondary text-sm">
            {hasExpectedSize ? `${formatBytes(downloadedBytes)} / ${formatBytes(expectedSize)} (${percent}%)` : `${formatBytes(downloadedBytes)} downloaded`}
          </div>
        </div>
        <button className="btn btn-outline btn-compact" type="button" onClick={onDismiss}>
          Dismiss
        </button>
      </div>
      {hasExpectedSize && (
        <div className="progress-bar-bg">
          <div className="progress-bar-fill fill-green" style={{ width: `${percent}%` }} />
        </div>
      )}
      {isComplete && showCompletion && <div className="alert alert-success compact-alert">Download completed</div>}
      {progressError && <div className="alert alert-danger compact-alert">{progressError}</div>}
    </div>
  );
};
