import { useEffect, useState } from 'react';
import { Copy, Eye, KeyRound, Plus, Trash2 } from 'lucide-react';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { createProxyUser, getErrorDetail, getProxyUserKey, getProxyUsers, revokeProxyUser } from '../lib/api.js';

export const ProxyUsersPage = ({ notify }) => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [revealedKeys, setRevealedKeys] = useState({});
  const [revealingUser, setRevealingUser] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [username, setUsername] = useState('');
  const [creating, setCreating] = useState(false);
  const [createdKey, setCreatedKey] = useState(null);
  const [revokeTarget, setRevokeTarget] = useState(null);
  const [revoking, setRevoking] = useState(false);

  const loadUsers = async () => {
    setError('');
    try {
      const response = await getProxyUsers();
      setUsers(Array.isArray(response.data) ? response.data : []);
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Failed to load proxy users'));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;
    getProxyUsers()
      .then((response) => {
        if (mounted) setUsers(Array.isArray(response.data) ? response.data : []);
      })
      .catch((requestError) => {
        if (mounted) setError(getErrorDetail(requestError, 'Failed to load proxy users'));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const handleReveal = async (targetUsername) => {
    if (revealedKeys[targetUsername]) return;
    setRevealingUser(targetUsername);
    setError('');
    try {
      const response = await getProxyUserKey(targetUsername);
      setRevealedKeys((prev) => ({ ...prev, [targetUsername]: response.data?.api_key || '' }));
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Failed to reveal API key'));
    } finally {
      setRevealingUser('');
    }
  };

  const handleCreate = async (event) => {
    event.preventDefault();
    setCreating(true);
    setError('');
    try {
      const response = await createProxyUser(username);
      const key = response.data?.api_key || '';
      setCreatedKey({ username: response.data?.username || username.trim(), apiKey: key });
      setUsername('');
      setShowAddModal(false);
      await loadUsers();
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Failed to create proxy user'));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async () => {
    if (!revokeTarget) return;
    setRevoking(true);
    setError('');
    try {
      await revokeProxyUser(revokeTarget.username);
      setRevealedKeys((prev) => {
        const next = { ...prev };
        delete next[revokeTarget.username];
        return next;
      });
      setRevokeTarget(null);
      notify?.(`Revoked ${revokeTarget.username}.`);
      await loadUsers();
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Failed to revoke proxy user'));
    } finally {
      setRevoking(false);
    }
  };

  const copyKey = (key) => {
    navigator.clipboard?.writeText(key);
  };

  const closeCreatedModal = () => {
    setCreatedKey(null);
  };

  return (
    <div className="stack">
      <div className="card proxy-users-card">
        <div className="card-header proxy-users-header">
          <h2 className="card-title">
            <KeyRound size={18} /> Proxy API Keys
          </h2>
          <button className="btn btn-primary" type="button" onClick={() => setShowAddModal(true)}>
            <Plus size={16} /> Add User
          </button>
        </div>

        {error && <div className="alert alert-danger">{error}</div>}
        {loading && <p className="text-secondary text-sm">Loading users...</p>}
        {!loading && (
          <div className="proxy-users-table-wrap">
            <div className="proxy-users-table">
              <div className="proxy-user-row proxy-user-table-head">
                <span>Sr. No</span>
                <span>Username</span>
                <span>API Key</span>
                <span>Req.</span>
                <span>Action</span>
              </div>
              {users.map((user, index) => {
                const fullKey = revealedKeys[user.username] || '';
                return (
                  <div className="proxy-user-row" key={user.username}>
                    <span>{index + 1}</span>
                    <span className="proxy-user-name">{user.username}</span>
                    <span className="proxy-key-cell">
                      <span className="proxy-key-value">{fullKey || user.api_key_preview}</span>
                      <button
                        className="icon-button"
                        type="button"
                        onClick={() => handleReveal(user.username)}
                        disabled={revealingUser === user.username}
                        aria-label={`Reveal API key for ${user.username}`}
                        title="Reveal API key"
                      >
                        <Eye size={15} />
                      </button>
                      <button
                        className="icon-button"
                        type="button"
                        onClick={() => copyKey(fullKey)}
                        disabled={!fullKey}
                        aria-label={`Copy API key for ${user.username}`}
                        title="Copy API key"
                      >
                        <Copy size={15} />
                      </button>
                    </span>
                    <span>{Number(user.request_count) || 0}</span>
                    <span>
                      <button className="icon-button proxy-revoke-button" type="button" onClick={() => setRevokeTarget(user)} aria-label={`Revoke ${user.username}`} title="Revoke">
                        <Trash2 size={14} />
                      </button>
                    </span>
                  </div>
                );
              })}
              {users.length === 0 && (
                <div className="proxy-user-empty">
                  <p className="text-secondary text-sm">No proxy users have been created.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {showAddModal && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="add-proxy-user-title">
          <div className="modal-content api-key-modal-content">
            <h3 id="add-proxy-user-title">Add Proxy User</h3>
            <form onSubmit={handleCreate}>
              <div className="form-group">
                <label className="form-label" htmlFor="proxyUsername">Username</label>
                <input
                  id="proxyUsername"
                  type="text"
                  className="form-control"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="off"
                  disabled={creating}
                  required
                />
              </div>
              <div className="modal-actions">
                <button className="btn btn-outline" type="button" onClick={() => setShowAddModal(false)} disabled={creating}>
                  Cancel
                </button>
                <button className="btn btn-primary" type="submit" disabled={creating}>
                  {creating ? 'Creating...' : 'Create User'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {createdKey && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="created-proxy-key-title">
          <div className="modal-content api-key-modal-content">
            <h3 id="created-proxy-key-title">Generated API Key</h3>
            <div className="modal-body">
              <p className="text-secondary text-sm">{createdKey.username}</p>
              <div className="inline-control generated-token-row">
                <div className="generated-token-box">{createdKey.apiKey}</div>
                <button className="btn btn-outline btn-compact" type="button" onClick={() => copyKey(createdKey.apiKey)} aria-label="Copy generated proxy API key">
                  <Copy size={14} />
                </button>
              </div>
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" type="button" onClick={closeCreatedModal}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {revokeTarget && (
        <ConfirmModal title="Revoke Proxy User?" confirmLabel="Revoke" danger loading={revoking} onCancel={() => setRevokeTarget(null)} onConfirm={handleRevoke}>
          <p>
            Revoke API access for <strong>{revokeTarget.username}</strong>?
          </p>
        </ConfirmModal>
      )}
    </div>
  );
};
