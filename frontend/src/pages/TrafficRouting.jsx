import { useEffect, useState } from 'react';
import { Loader2, Network, Save, Trash2 } from 'lucide-react';
import { getRoutingTable, saveRoutingTable } from '../lib/api';
import { ProxyUsersPage } from './ProxyUsersPage.jsx';

export const TrafficRouting = ({ notify }) => {
  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getRoutingTable()
      .then((res) => {
        if (cancelled) return;
        const table = res.data || {};
        const routesList = Object.keys(table).map((model) => ({ model, port: table[model] }));
        setRoutes(routesList);
      })
      .catch(() => {
        if (!cancelled) setError('Failed to load routing table.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      const table = {};
      routes.forEach((route) => {
        if (route.model.trim() && route.port) {
          table[route.model.trim()] = Number(route.port);
        }
      });

      await saveRoutingTable(table);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch {
      setError('Failed to save routing table.');
    } finally {
      setSaving(false);
    }
  };

  const removeRoute = (index) => {
    setRoutes(routes.filter((_, i) => i !== index));
  };

  const updateRoute = (index, field, value) => {
    setRoutes(routes.map((route, i) => (i === index ? { ...route, [field]: value } : route)));
  };

  if (loading) {
    return (
      <div className="flex" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <div className="loader"></div>
      </div>
    );
  }

  return (
    <div className="stack proxy-admin-page">
      <div className="card settings-card">
        <div className="card-header routing-card-header">
          <div>
            <h2 className="card-title">
              <Network size={18} /> Model Routing
            </h2>
            <p className="text-secondary text-sm routing-description">
              Route exact model names from `/v1/models` to managed vLLM ports.
            </p>
          </div>
        </div>

        {error && <div className="alert alert-danger">{error}</div>}
        {success && <div className="alert alert-success">Routing table saved successfully!</div>}

        {routes.length === 0 && (
          <div className="routing-empty-state">
            <p className="text-secondary text-sm">No routes configured.</p>
          </div>
        )}

        {routes.length > 0 && (
          <div className="model-list routing-list">
            {routes.map((route, i) => (
              <div key={i} className="model-item cache-model-item routing-row">
                <div className="routing-model-field">
                  <input
                    type="text"
                    value={route.model}
                    onChange={(e) => updateRoute(i, 'model', e.target.value)}
                    placeholder="Model String (e.g. Qwen/Qwen2.5-VL-72B-Instruct)"
                    className="form-control"
                  />
                </div>
                <div className="routing-port-field">
                  <input
                    type="number"
                    value={route.port}
                    onChange={(e) => updateRoute(i, 'port', e.target.value)}
                    placeholder="Target Port (e.g. 8082)"
                    className="form-control"
                  />
                </div>
                <div className="cache-model-actions">
                  <button type="button" onClick={() => removeRoute(i)} className="icon-button text-red" aria-label="Remove route">
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="routing-actions">
          <button type="button" onClick={handleSave} disabled={saving} className="btn btn-primary">
            {saving ? (
              <>
                <Loader2 size={16} className="loader-inline" /> Saving...
              </>
            ) : (
              <>
                <Save size={16} /> Save Routes
              </>
            )}
          </button>
        </div>
      </div>

      <ProxyUsersPage notify={notify} />
    </div>
  );
};
