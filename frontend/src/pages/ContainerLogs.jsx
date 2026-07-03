import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { ArrowLeft, RefreshCw, Copy, Check, Play, Square, Container } from 'lucide-react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';

import { getContainerLogs, streamContainerLogs } from '../lib/api.js';
import { getErrorDetail } from '../lib/api.js';

export const ContainerLogs = ({ activeModels }) => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const containerId = searchParams.get('container_id') || '';

  const [selectedContainer, setSelectedContainer] = useState(containerId);
  const [logs, setLogs] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState('');
  const logRef = useRef(null);
  const stopStreamRef = useRef(null);

  const availableModels = useMemo(() => activeModels || [], [activeModels]);
  const runningModels = useMemo(() => availableModels.filter((m) => m.state === 'running'), [availableModels]);
  const isSelectedActive = selectedContainer && runningModels.some((m) => m.container_id === selectedContainer);

  const stopStreaming = useCallback(() => {
    if (stopStreamRef.current) {
      stopStreamRef.current();
      stopStreamRef.current = null;
    }
    setStreaming(false);
    setStreamError('');
  }, []);

  const startStreaming = useCallback(() => {
    if (stopStreamRef.current || !selectedContainer) return;
    setStreaming(true);
    setStreamError('');
    // Keep the fetched snapshot, then append live chunks below it.
    stopStreamRef.current = streamContainerLogs(
      selectedContainer,
      (data, isStdout) => {
        let toAdd = isStdout ? data : `[stderr] ${data}`;
        if (toAdd && !/[\n\r]$/.test(toAdd)) {
          toAdd += '\n';
        }
        setLogs((prev) => {
          const separator = prev && !/[\n\r]$/.test(prev) ? '\n' : '';
          return `${prev}${separator}${toAdd}`;
        });
      },
      (err) => {
        setStreamError(err);
        setStreaming(false);
        stopStreamRef.current = null;
      },
      () => {
        setStreaming(false);
        stopStreamRef.current = null;
      }
    );
  }, [selectedContainer]);

  const fetchLogs = useCallback(async () => {
    if (!selectedContainer) {
      setLogs('');
      setError('');
      setLoading(false);
      return;
    }
    try {
      stopStreaming();
      setLoading(true);
      const res = await getContainerLogs(selectedContainer);
      setLogs(res.data.logs || '');
      setError('');
    } catch (err) {
      setError(getErrorDetail(err, 'Failed to load container logs'));
    } finally {
      setLoading(false);
    }
  }, [selectedContainer, stopStreaming]);

  const handleContainerChange = (e) => {
    const newContainerId = e.target.value;
    setSelectedContainer(newContainerId);
    if (newContainerId) {
      navigate(`/logs?container_id=${encodeURIComponent(newContainerId)}`);
    } else {
      navigate('/logs');
    }
  };

  useEffect(() => {
    let cancelled = false;
    const loadAndMaybeStream = async () => {
      await fetchLogs();
      if (!cancelled && selectedContainer && isSelectedActive) {
        startStreaming();
      }
    };
    loadAndMaybeStream();
    return () => {
      cancelled = true;
      stopStreaming();
    };
  }, [fetchLogs, selectedContainer, isSelectedActive, startStreaming, stopStreaming]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  const handleRefresh = () => {
    fetchLogs();
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(logs);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  return (
    <div className="logs-page">
      <div className="logs-toolbar">
        <div className="logs-toolbar-left">
          <Link to="/" className="btn btn-icon btn-ghost">
            <ArrowLeft size={18} />
          </Link>
          <h2>Container Logs</h2>
        </div>
        <div className="logs-toolbar-right">
          <div className="container-select-wrapper">
            <Container size={16} />
            <select
              value={selectedContainer}
              onChange={handleContainerChange}
              className="container-select"
            >
              <option value="">Select container...</option>
              {availableModels.map((model) => (
                <option key={model.container_id} value={model.container_id}>
                  {model.model} (port {model.port}) {model.state !== 'running' ? `[${model.state}]` : ''}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={streaming ? stopStreaming : startStreaming}
            className={`btn btn-icon ${streaming ? 'btn-danger' : 'btn-success'}`}
            disabled={!selectedContainer || (!isSelectedActive && !streaming)}
            title={streaming ? 'Stop live stream' : (isSelectedActive ? 'Start live stream' : 'Live streaming only available for active containers')}
          >
            {streaming ? <Square size={16} /> : <Play size={16} />}
          </button>
          <button
            onClick={handleRefresh}
            className="btn btn-icon btn-ghost"
            disabled={!selectedContainer}
            title="Refresh logs"
          >
            <RefreshCw size={16} />
          </button>
          <button
            onClick={handleCopy}
            className="btn btn-icon btn-ghost"
            disabled={!logs}
            title="Copy logs"
          >
            {copied ? <Check size={16} /> : <Copy size={16} />}
          </button>
          <label className="auto-scroll-label">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              disabled={!logs}
            />
            Auto-scroll
          </label>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}
      {streamError && <div className="alert alert-warning">{streamError}</div>}
      {streaming && (
        <div className="alert alert-live">
          <span className="live-dot" /> Streaming live
        </div>
      )}

      <div
        ref={logRef}
        className="logs-container"
      >
        {loading ? (
          <div className="logs-loading">Loading logs...</div>
        ) : !selectedContainer ? (
          <div className="logs-empty">
            <Container size={48} />
            <p>Select a running container from the dropdown above to view its logs.</p>
          </div>
        ) : logs ? (
          <pre className="logs-content">{logs}</pre>
        ) : (
          <div className="logs-empty">
            <p>No logs available for this container.</p>
          </div>
        )}
      </div>
    </div>
  );
};
