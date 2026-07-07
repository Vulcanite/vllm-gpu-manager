import { useEffect, useState } from 'react';
import { Network } from 'lucide-react';
import { getRoutingTable } from '../lib/api';
import { ProxyUsersPage } from './ProxyUsersPage.jsx';

export const TrafficRouting = ({ notify }) => {
  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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
                  <span className="routing-field-label">Model</span>
                  <span className="routing-field-value" title={route.model}>{route.model}</span>
                </div>
                <div className="routing-port-field">
                  <span className="routing-field-label">Port</span>
                  <span className="routing-port-value">{route.port}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <ProxyUsersPage notify={notify} />
    </div>
  );
};
