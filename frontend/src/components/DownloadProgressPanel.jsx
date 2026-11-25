import { useEffect, useState } from 'react';

import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { getDownloadProgress } from '../lib/api.js';
import { formatBytes } from '../lib/format.js';

export const DownloadProgressPanel = ({ modelName, expectedSize, onDismiss }) => {
  const [downloadedBytes, setDownloadedBytes] = useState(0);
  const [state, setState] = useState('running');
  const [failure, setFailure] = useState('');
  const [progressError, setProgressError] = useState('');
  const [showCompletion, setShowCompletion] = useState(true);

  useEffect(() => {
    if (!modelName) return undefined;

    let interval;
    const fetchProgress = async () => {
      try {
        const response = await getDownloadProgress(modelName);
        const data = response.data || {};
        setDownloadedBytes(data.downloaded_bytes || 0);
        setProgressError('');
        if (data.state && data.state !== 'running') {
          // The server removes a finished download container once it has
          // reported it, so stop polling and keep the final state.
          setState(data.state);
          if (data.state === 'failed') {
            setFailure(`Download failed (exit ${data.exit_code})${data.error ? `:\n${data.error}` : ''}`);
          }
          clearInterval(interval);
        }
      } catch (error) {
        setProgressError(error.response?.data?.detail || error.message || 'Failed to fetch download progress');
      }
    };

    fetchProgress();
    interval = setInterval(fetchProgress, 2000);
    return () => clearInterval(interval);
  }, [modelName, expectedSize]);

  const hasExpectedSize = expectedSize && expectedSize > 0;
  const isComplete = state === 'succeeded';
  // Bytes on disk include partial files, so only the server's state says 100%.
  const percent = isComplete ? 100 : hasExpectedSize ? Math.min(99, Math.round((downloadedBytes / expectedSize) * 100)) : 0;

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
      {state === 'failed' && <pre className="alert alert-danger compact-alert" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{failure}</pre>}
      {state === 'stopped' && <div className="alert alert-danger compact-alert">Download stopped before it finished. Start it again to resume.</div>}
      {progressError && <div className="alert alert-danger compact-alert">{progressError}</div>}
    </div>
  );
};
