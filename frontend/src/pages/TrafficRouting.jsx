import React, { useState, useEffect } from 'react';
import { Network, Plus, Trash2, Save, Loader2 } from 'lucide-react';
import { getRoutingTable, saveRoutingTable } from '../lib/api';

export const TrafficRouting = () => {
  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    fetchRoutes();
  }, []);

  const fetchRoutes = async () => {
    try {
      setLoading(true);
      const res = await getRoutingTable();
      const table = res.data || {};
      const routesList = Object.keys(table).map(model => ({ model, port: table[model] }));
      setRoutes(routesList);
    } catch (err) {
      setError('Failed to load routing table.');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      const table = {};
      routes.forEach(route => {
        if (route.model.trim() && route.port) {
          table[route.model.trim()] = Number(route.port);
        }
      });

      await saveRoutingTable(table);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError('Failed to save routing table.');
    } finally {
      setSaving(false);
    }
  };

  const addRoute = () => {
    setRoutes([...routes, { model: '', port: '' }]);
  };

  const removeRoute = (index) => {
    const newRoutes = [...routes];
    newRoutes.splice(index, 1);
    setRoutes(newRoutes);
  };

  const updateRoute = (index, field, value) => {
    const newRoutes = [...routes];
    newRoutes[index][field] = value;
    setRoutes(newRoutes);
  };

  if (loading) {
    return (
      <div className="flex" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <div className="loader"></div>
      </div>
    );
  }

  return (
    <div className="settings-layout" style={{ gridTemplateColumns: '1fr', maxWidth: '900px' }}>
      <div className="card settings-card">
        <div className="card-header" style={{ marginBottom: '0.5rem' }}>
          <h2 className="card-title">
            <Network size={18} /> Traffic Routing Manager
          </h2>
        </div>
        
        <p className="text-secondary text-sm mb-4">
          Map incoming model requests to backend vLLM ports. The reverse proxy will inspect the "model" field of incoming inference requests and route them to the corresponding port.
        </p>

        {error && <div className="alert alert-danger">{error}</div>}
        {success && <div className="alert alert-success">Routing table saved successfully!</div>}

        <div className="model-list">
          {routes.length === 0 ? (
            <div className="empty-state">
              <p className="text-sm">No routes configured.</p>
            </div>
          ) : (
            routes.map((route, i) => (
              <div key={i} className="model-item cache-model-item">
                <div style={{ flex: 2 }}>
                  <input
                    type="text"
                    value={route.model}
                    onChange={(e) => updateRoute(i, 'model', e.target.value)}
                    placeholder="Model String (e.g. Qwen/Qwen2.5-VL-72B-Instruct)"
                    className="form-control"
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <input
                    type="number"
                    value={route.port}
                    onChange={(e) => updateRoute(i, 'port', e.target.value)}
                    placeholder="Target Port (e.g. 8082)"
                    className="form-control"
                  />
                </div>
                <div className="cache-model-actions">
                  <button onClick={() => removeRoute(i)} className="icon-button text-red">
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-subtle)' }}>
          <button
            onClick={addRoute}
            className="btn btn-outline"
          >
            <Plus size={16} /> Add Route
          </button>

          <button
            onClick={handleSave}
            disabled={saving}
            className="btn btn-primary"
          >
            {saving ? (
              <>
                <Loader2 size={16} className="loader-inline" /> Saving...
              </>
            ) : (
              <>
                <Save size={16} /> Save Routing Table
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
