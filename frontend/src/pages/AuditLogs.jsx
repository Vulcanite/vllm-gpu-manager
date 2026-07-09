import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw, ShieldCheck } from 'lucide-react';

import { getAuditLogs, getErrorDetail } from '../lib/api.js';

const ACTION_LABELS = {
  model_deploy: 'Deploy',
  model_stop: 'Stop',
  model_remove: 'Remove',
  model_restart: 'Start / Restart',
  proxy_api_key_created: 'Proxy key created',
  proxy_api_key_deleted: 'Proxy key deleted',
  server_api_key_created: 'Server key created',
  server_api_key_deleted: 'Server key deleted',
};

const formatTimestamp = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
};

const eventTarget = (event) => {
  if (event.model) return event.port ? `${event.model} :${event.port}` : event.model;
  if (event.username) return event.username;
  if (event.container_name) return event.container_name;
  if (event.container_id) return event.container_id.slice(0, 12);
  return '-';
};

const paramsPreview = (params) => {
  if (!params || typeof params !== 'object') return '';
  const entries = Object.entries(params).filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return value !== '' && value !== null && value !== undefined;
  });
  if (entries.length === 0) return '';
  return entries
    .map(([key, value]) => {
      const rendered = Array.isArray(value) ? value.join(' ') : String(value);
      return `${key}=${rendered}`;
    })
    .join('  ');
};

export const AuditLogs = () => {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadEvents = useCallback(async () => {
    setError('');
    try {
      const response = await getAuditLogs(200);
      setEvents(Array.isArray(response.data) ? response.data : []);
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Failed to load audit logs'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeout = setTimeout(() => {
      loadEvents();
    }, 0);
    return () => clearTimeout(timeout);
  }, [loadEvents]);

  const renderedEvents = useMemo(() => events.map((event, index) => ({
    ...event,
    id: `${event.timestamp || index}-${event.action || 'event'}-${index}`,
    actionLabel: ACTION_LABELS[event.action] || event.action || 'Event',
    target: eventTarget(event),
    paramsText: paramsPreview(event.params),
  })), [events]);

  return (
    <div className="stack audit-page">
      <div className="card audit-card">
        <div className="card-header audit-header">
          <h2 className="card-title">
            <ShieldCheck size={18} /> Audit Logs
          </h2>
          <button className="btn btn-outline" type="button" onClick={loadEvents} disabled={loading}>
            <RefreshCw size={16} /> Refresh
          </button>
        </div>

        {error && <div className="alert alert-danger">{error}</div>}
        {loading && <p className="text-secondary text-sm">Loading audit events...</p>}
        {!loading && renderedEvents.length === 0 && (
          <div className="empty-state">No audit events have been recorded.</div>
        )}
        {!loading && renderedEvents.length > 0 && (
          <div className="audit-table-wrap">
            <table className="audit-table">
              <thead>
                <tr>
                  <th scope="col">Time</th>
                  <th scope="col">Action</th>
                  <th scope="col">Result</th>
                  <th scope="col">Target</th>
                  <th scope="col">Summary</th>
                </tr>
              </thead>
              <tbody>
                {renderedEvents.map((event) => (
                  <tr key={event.id}>
                    <td className="audit-time">{formatTimestamp(event.timestamp)}</td>
                    <td>{event.actionLabel}</td>
                    <td>
                      <span className={`audit-result audit-result-${event.result || 'unknown'}`}>
                        {event.result || 'unknown'}
                      </span>
                    </td>
                    <td className="audit-target">{event.target}</td>
                    <td>
                      <div className="audit-summary">{event.summary || '-'}</div>
                      {event.error && <div className="audit-error">{event.error}</div>}
                      {event.paramsText && <div className="audit-params">{event.paramsText}</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
