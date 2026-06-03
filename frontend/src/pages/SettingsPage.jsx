import { useEffect, useState } from 'react';
import { Activity, Settings } from 'lucide-react';

import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { getErrorDetail, getSettings, saveSettings } from '../lib/api.js';
import { formatBytes } from '../lib/format.js';

const getStats = (data, prefix) => {
  const total = data[`${prefix}_total`];
  if (!total) return null;

  const used = data[`${prefix}_used`];
  return {
    total,
    used,
    free: data[`${prefix}_free`],
    percent: (used / total) * 100,
  };
};

const ResourceMeter = ({ label, stats, freeLabel }) => {
  if (!stats) return null;

  const fillClass = stats.percent > 90 ? 'danger' : stats.percent > 75 ? 'warning' : 'success';

  return (
    <div>
      <div className="metric-row">
        <span>{label}</span>
        <span className="metric-value">
          {formatBytes(stats.used)} / {formatBytes(stats.total)} ({stats.percent.toFixed(1)}%)
        </span>
      </div>
      <div className="progress-bar-bg">
        <div className={`progress-bar-fill ${fillClass}`} style={{ width: `${stats.percent}%` }} />
      </div>
      <div className="text-right">
        <span className="text-secondary text-sm">
          {formatBytes(stats.free)} {freeLabel}
        </span>
      </div>
    </div>
  );
};

export const SettingsPage = ({ fetchData }) => {
  const [hfCache, setHfCache] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [updateHfToken, setUpdateHfToken] = useState(false);
  const [diskStats, setDiskStats] = useState(null);
  const [ramStats, setRamStats] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useAutoDismiss(message, setMessage);
  useAutoDismiss(error, setError);

  const applySettingsData = (data) => {
    setHfCache(data.hf_cache);
    setHfToken('');
    setHfTokenSet(Boolean(data.hf_token_set));
    setUpdateHfToken(false);
    setDiskStats(getStats(data, 'disk'));
    setRamStats(getStats(data, 'ram'));
  };

  useEffect(() => {
    let mounted = true;
    getSettings()
      .then((response) => {
        if (mounted) {
          applySettingsData(response.data);
          setError('');
        }
      })
      .catch((requestError) => {
        if (mounted) {
          setError(getErrorDetail(requestError, 'Failed to load settings'));
        }
      })
      .finally(() => {
        if (mounted) {
          setLoading(false);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  const handleSave = async (event) => {
    event.preventDefault();
    const requestedHFCache = hfCache.trim();
    setMessage('');
    setError('');

    if (updateHfToken && !hfToken.trim()) {
      setError('Enter a Hugging Face token or disable the token update switch.');
      return;
    }

    setSaving(true);

    try {
      await saveSettings({
        hf_cache: requestedHFCache,
        hf_token: updateHfToken ? hfToken.trim() : '',
        clear_hf_token: false,
      });

      const response = await getSettings();
      applySettingsData(response.data);
      if (requestedHFCache && response.data?.hf_cache && requestedHFCache !== response.data.hf_cache) {
        setMessage(`Settings saved. Hugging Face cache path normalized to ${response.data.hf_cache}.`);
      } else {
        setMessage('Settings saved successfully.');
      }
      fetchData();
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Error saving settings'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-layout">
      <div className="card settings-card">
        <div className="card-header">
          <h2 className="card-title">
            <Settings size={18} /> Configuration
          </h2>
        </div>
        {message && <div className="alert alert-success">{message}</div>}
        {error && <div className="alert alert-danger">{error}</div>}
        <form onSubmit={handleSave}>
          <div className="form-group">
            <label className="form-label">Hugging Face Home Directory</label>
            <input type="text" className="form-control" value={hfCache} onChange={(event) => setHfCache(event.target.value)} required disabled={saving || loading} />
            <p className="text-secondary text-sm mt-1">Absolute path containing the Hugging Face hub directory.</p>
          </div>

          <div className="form-group">
            <div className="setting-row">
              <div>
                <label className="form-label">Hugging Face Token</label>
                <p className="text-secondary text-sm">
                  {hfTokenSet ? 'A token is stored. Enable the switch to replace it.' : 'No token stored. Enable the switch to add one.'}
                </p>
              </div>
              <label className="toggle-control" aria-label="Enable Hugging Face token update">
                <input
                  type="checkbox"
                  checked={updateHfToken}
                  disabled={saving || loading}
                  onChange={(event) => {
                    setUpdateHfToken(event.target.checked);
                    if (!event.target.checked) {
                      setHfToken('');
                    }
                  }}
                />
                <span className="toggle-track" aria-hidden="true">
                  <span className="toggle-thumb" />
                </span>
              </label>
            </div>

            {updateHfToken && (
              <div className="token-update-field">
                <label className="form-label">New Hugging Face Token</label>
                <input
                  type="password"
                  className="form-control"
                  value={hfToken}
                  onChange={(event) => setHfToken(event.target.value)}
                  placeholder="hf_..."
                  autoComplete="off"
                  disabled={saving || loading}
                />
                <p className="text-secondary text-sm mt-1">Used to access gated models and avoid Hugging Face rate limits.</p>
              </div>
            )}
          </div>

          <button type="submit" className="btn btn-primary mt-4" disabled={saving || loading}>
            {saving ? 'Saving...' : 'Save changes'}
          </button>
        </form>
      </div>

      <div className="card resource-card">
        <div className="card-header">
          <h2 className="card-title">
            <Activity size={18} /> Resource Usage
          </h2>
        </div>
        {loading && <p className="text-secondary text-sm">Loading settings...</p>}
        {!loading && (
          <div className="resource-grid resource-grid-stacked">
            <ResourceMeter label="Cache Drive Usage" stats={diskStats} freeLabel="remaining" />
            <ResourceMeter label="System RAM Usage" stats={ramStats} freeLabel="available" />
          </div>
        )}
      </div>
    </div>
  );
};
