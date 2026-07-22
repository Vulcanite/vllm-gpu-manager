import { useEffect, useState } from 'react';
import { Activity, Copy, Cpu, ExternalLink, Eye, KeyRound, RefreshCw, Settings } from 'lucide-react';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { getErrorDetail, getSettings, getVllmAPIKey, saveSettings } from '../lib/api.js';
import { generateVLLMToken } from '../lib/tokens.js';
import { getVLLMDocsUrl } from '../lib/vllmDocs.js';
import { ResourceMeter } from '../components/settings/ResourceMeter.jsx';

const credentialStatus = (isSet, suffix) => {
  if (!isSet) return 'Not Configured';
  return suffix ? `Configured · ends in …${suffix}` : 'Configured';
};

export const SettingsPage = ({ fetchData, notify, proxyEnabled = false }) => {
  const [hfCache, setHfCache] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [hfTokenSuffix, setHfTokenSuffix] = useState('');
  const [editingHfToken, setEditingHfToken] = useState(false);
  const [vllmAPIKey, setVllmAPIKey] = useState('');
  const [vllmAPIKeySet, setVllmAPIKeySet] = useState(false);
  const [vllmAPIKeySuffix, setVllmAPIKeySuffix] = useState('');
  const [vllmAPIKeyContainerCount, setVllmAPIKeyContainerCount] = useState(0);
  const [editingVllmAPIKey, setEditingVllmAPIKey] = useState(false);
  const [viewingVllmAPIKey, setViewingVllmAPIKey] = useState(false);
  const [revealedVllmAPIKey, setRevealedVllmAPIKey] = useState('');
  const [revealVllmAPIKeyError, setRevealVllmAPIKeyError] = useState('');
  const [revealVllmAPIKeyLoading, setRevealVllmAPIKeyLoading] = useState(false);
  const [enableCUDACompatibility, setEnableCUDACompatibility] = useState(false);
  const [hfHubOffline, setHFHubOffline] = useState(false);
  const [vllmImage, setVllmImage] = useState('');
  const [serverBindIP, setServerBindIP] = useState('0.0.0.0');
  const [diskData, setDiskData] = useState(null);
  const [ramData, setRamData] = useState(null);
  const [generatedVllmToken, setGeneratedVllmToken] = useState('');
  const [pendingCredentialRemoval, setPendingCredentialRemoval] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const applySettingsData = (data) => {
    setHfCache(data.hf_cache);
    setHfToken('');
    setHfTokenSet(Boolean(data.hf_token_set));
    setHfTokenSuffix(data.hf_token_suffix || '');
    setEditingHfToken(false);
    setVllmAPIKey('');
    setVllmAPIKeySet(Boolean(data.vllm_api_key_set));
    setVllmAPIKeySuffix(data.vllm_api_key_suffix || '');
    setVllmAPIKeyContainerCount(Number(data.vllm_api_key_container_count) || 0);
    setEditingVllmAPIKey(false);
    setGeneratedVllmToken('');
    setEnableCUDACompatibility(Boolean(data.enable_cuda_compatibility));
    setHFHubOffline(Boolean(data.hf_hub_offline));
    setVllmImage(data.vllm_image || '');
    setServerBindIP(data.server_bind_ip || '0.0.0.0');
    setDiskData(data);
    setRamData(data);
  };

  useEffect(() => {
    let mounted = true;
    getSettings()
      .then((response) => {
        if (mounted) {
          applySettingsData(response.data);
        }
      })
      .catch((requestError) => {
        if (mounted) {
          notify(getErrorDetail(requestError, 'Failed to load settings'), 'danger');
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
  }, [notify]);

  const baseSettingsPayload = () => ({
    hf_cache: hfCache.trim(),
    hf_token: '',
    clear_hf_token: false,
    vllm_api_key: '',
    clear_vllm_api_key: false,
    inference_api_key: '',
    clear_inference_api_key: false,
    enable_cuda_compatibility: enableCUDACompatibility,
    hf_hub_offline: hfHubOffline,
    vllm_image: vllmImage.trim(),
    server_bind_ip: serverBindIP.trim(),
  });

  const saveAndRefresh = async (payload, successMessage) => {
    setSaving(true);
    try {
      await saveSettings(payload);
      const response = await getSettings();
      applySettingsData(response.data);
      notify(successMessage);
      fetchData();
    } catch (requestError) {
      notify(getErrorDetail(requestError, 'Error saving settings'), 'danger');
    } finally {
      setSaving(false);
    }
  };

  const handleGeneralSave = async (event) => {
    event.preventDefault();
    const requestedHFCache = hfCache.trim();

    const payload = baseSettingsPayload();

    await saveAndRefresh(
      payload,
      requestedHFCache && requestedHFCache !== hfCache ? `Settings saved. Hugging Face cache path normalized to ${hfCache}.` : 'Settings saved successfully.',
    );
  };

  const handleSaveHfToken = async (event) => {
    event.preventDefault();

    if (!hfToken.trim()) {
      notify('Enter a Hugging Face token or cancel replacement.', 'danger');
      return;
    }

    await saveAndRefresh({ ...baseSettingsPayload(), hf_token: hfToken.trim() }, 'Hugging Face token saved.');
  };

  const handleSaveVllmAPIKey = async (event) => {
    event.preventDefault();

    if (!vllmAPIKey.trim()) {
      notify('Enter a vLLM API key or cancel replacement.', 'danger');
      return;
    }

    if (/\s/.test(vllmAPIKey.trim())) {
      notify('vLLM API key cannot contain whitespace.', 'danger');
      return;
    }

    await saveAndRefresh({ ...baseSettingsPayload(), vllm_api_key: vllmAPIKey.trim() }, 'vLLM API key saved.');
  };

  const handleGenerateVllmAPIKey = () => {
    const token = generateVLLMToken();
    setVllmAPIKey(token);
    setGeneratedVllmToken(token);
  };

  const handleCopyGeneratedVllmToken = () => {
    navigator.clipboard?.writeText(generatedVllmToken);
  };

  const handleViewVllmAPIKey = async () => {
    setViewingVllmAPIKey(true);
    setRevealVllmAPIKeyError('');
    setRevealVllmAPIKeyLoading(true);
    try {
      const response = await getVllmAPIKey();
      setRevealedVllmAPIKey(response.data?.vllm_api_key || '');
    } catch (requestError) {
      setRevealVllmAPIKeyError(getErrorDetail(requestError, 'Failed to load vLLM API key'));
      setRevealedVllmAPIKey('');
    } finally {
      setRevealVllmAPIKeyLoading(false);
    }
  };

  const handleCopyRevealedVllmAPIKey = () => {
    navigator.clipboard?.writeText(revealedVllmAPIKey);
  };

  const closeVllmAPIKeyModal = () => {
    setViewingVllmAPIKey(false);
    setRevealedVllmAPIKey('');
    setRevealVllmAPIKeyError('');
    setRevealVllmAPIKeyLoading(false);
  };

  const requestCredentialRemoval = (type) => {
    setPendingCredentialRemoval(type);
  };

  const confirmCredentialRemoval = async () => {
    if (!pendingCredentialRemoval) return;

    const payload = baseSettingsPayload();
    if (pendingCredentialRemoval === 'hf') {
      payload.clear_hf_token = true;
    } else {
      payload.clear_vllm_api_key = true;
    }

    const successMessage = pendingCredentialRemoval === 'hf' ? 'Hugging Face token removed.' : 'vLLM API key removed.';
    setPendingCredentialRemoval(null);
    await saveAndRefresh(payload, successMessage);
  };

  const vllmAPIKeyInUse = vllmAPIKeyContainerCount > 0;
  const vllmDocsUrl = getVLLMDocsUrl(vllmImage);
  const vllmRemoveKeyTitle = vllmAPIKeyInUse
    ? `${vllmAPIKeyContainerCount} existing vLLM container${vllmAPIKeyContainerCount === 1 ? ' is' : 's are'} using this API key. Remove those containers before removing the key.`
    : 'Remove the stored vLLM API key.';

  return (
    <div className="settings-layout">
      <div className="stack">
        <div className="card settings-card">
          <div className="card-header">
            <h2 className="card-title">
              <Settings size={18} /> Model Storage Settings
            </h2>
          </div>
          <form className="serving-environment-form" onSubmit={handleGeneralSave}>
            <div className="form-group">
              <label className="form-label" htmlFor="hfCache">Hugging Face Model Directory</label>
              <input id="hfCache" type="text" className="form-control" value={hfCache} onChange={(event) => setHfCache(event.target.value)} required disabled={saving || loading} />
              <p className="text-secondary text-sm mt-1">Absolute path containing the Hugging Face Model hub directory.</p>
            </div>
            <button type="submit" className="btn btn-primary mt-2" disabled={saving || loading}>
              {saving ? 'Saving...' : 'Save directory'}
            </button>
          </form>
        </div>

        <div className="card settings-card">
          <div className="card-header">
            <h2 className="card-title">
              <KeyRound size={18} /> Hugging Face Token
            </h2>
          </div>

          <div className="credential-panel no-top-border">
            <div className="credential-header-row">
              <div>
                <label className="form-label">Hugging Face Token</label>
                <div className={`credential-status ${hfTokenSet ? 'active' : ''}`}>{credentialStatus(hfTokenSet, hfTokenSuffix)}</div>
                <p className="text-secondary text-sm mt-1">Used to access gated models and avoid Hugging Face rate limits.</p>
              </div>
              <div className="credential-actions">
                <button className="btn btn-outline" type="button" onClick={() => setEditingHfToken(true)} disabled={saving || loading || editingHfToken}>
                  {hfTokenSet ? 'Replace' : 'Configure'}
                </button>
                {hfTokenSet && (
                  <button className="btn btn-danger" type="button" onClick={() => requestCredentialRemoval('hf')} disabled={saving || loading}>
                    Remove
                  </button>
                )}
              </div>
            </div>

            {editingHfToken && (
              <form className="credential-edit-form" onSubmit={handleSaveHfToken}>
                <input id="hfToken" type="password" className="form-control" value={hfToken} onChange={(event) => setHfToken(event.target.value)} placeholder="hf_..." autoComplete="off" disabled={saving || loading} />
                <div className="credential-edit-actions">
                  <button className="btn btn-primary" type="submit" disabled={saving || loading}>{saving ? 'Saving...' : 'Save'}</button>
                  <button className="btn btn-outline" type="button" onClick={() => { setEditingHfToken(false); setHfToken(''); }} disabled={saving || loading}>Cancel</button>
                </div>
                <p className="text-secondary text-sm">Used for gated model access; save to replace the stored token.</p>
              </form>
            )}
          </div>
        </div>

        {!proxyEnabled && (
          <div className="card settings-card">
            <div className="card-header">
              <h2 className="card-title">
                <KeyRound size={18} /> vLLM API Key
              </h2>
            </div>

            <div className="credential-panel no-top-border">
              <div className="credential-header-row">
                <div>
                  <label className="form-label">vLLM API Key</label>
                  <div className={`credential-status ${vllmAPIKeySet ? 'active' : ''}`}>{credentialStatus(vllmAPIKeySet, vllmAPIKeySuffix)}</div>
                  <p className="text-secondary text-sm mt-1">Authenticates requests to all new serving containers.</p>
                </div>
                <div className="credential-actions">
                  {vllmAPIKeySet && (
                    <button className="btn btn-outline" type="button" onClick={handleViewVllmAPIKey} disabled={saving || loading || revealVllmAPIKeyLoading}>
                      <Eye size={14} /> View
                    </button>
                  )}
                  {!vllmAPIKeyInUse && (
                    <button className="btn btn-outline" type="button" onClick={() => setEditingVllmAPIKey(true)} disabled={saving || loading || editingVllmAPIKey}>
                      {vllmAPIKeySet ? 'Replace' : 'Configure'}
                    </button>
                  )}
                </div>
              </div>

              {editingVllmAPIKey && (
                <form className="credential-edit-form" onSubmit={handleSaveVllmAPIKey}>
                  <div className="inline-control token-generator-control">
                    <input id="vllmAPIKey" type="password" className="form-control" value={vllmAPIKey} onChange={(event) => setVllmAPIKey(event.target.value)} placeholder="vllm_..." autoComplete="off" disabled={saving || loading} />
                    <button className="btn btn-outline" type="button" onClick={handleGenerateVllmAPIKey} disabled={saving || loading}>
                      <RefreshCw size={14} /> Regenerate
                    </button>
                  </div>
                  <div className="credential-edit-actions">
                    <button className="btn btn-primary" type="submit" disabled={saving || loading}>{saving ? 'Saving...' : 'Save'}</button>
                    <button className="btn btn-outline" type="button" onClick={() => { setEditingVllmAPIKey(false); setVllmAPIKey(''); }} disabled={saving || loading}>Cancel</button>
                  </div>
                  <p className="text-secondary text-sm">Takes effect on next container start.</p>
                </form>
              )}

              {vllmAPIKeySet && (
                <div className="credential-danger-zone">
                  <p className="text-secondary text-sm">
                    {vllmAPIKeyInUse
                      ? `${vllmAPIKeyContainerCount} existing container${vllmAPIKeyContainerCount === 1 ? ' is' : 's are'} using this key. Remove those containers before deleting the stored key.`
                      : 'Removes authentication from all new containers. Existing running containers are unaffected.'}
                  </p>
                  <span className="disabled-tooltip-wrap" title={vllmRemoveKeyTitle}>
                    <button className="btn btn-danger btn-nowrap" type="button" onClick={() => requestCredentialRemoval('vllm')} disabled={saving || loading || vllmAPIKeyInUse}>
                      Remove Key
                    </button>
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="stack">
        <div className="card resource-card">
          <div className="card-header">
            <h2 className="card-title">
              <Activity size={18} /> Resource Usage
            </h2>
          </div>
          {loading && <p className="text-secondary text-sm">Loading settings...</p>}
          {!loading && (
            <div className="resource-grid resource-grid-stacked">
              <ResourceMeter label="Cache Drive Usage" data={diskData} prefix="disk" freeLabel="remaining" />
              <ResourceMeter label="System RAM Usage" data={ramData} prefix="ram" freeLabel="available" />
            </div>
          )}
        </div>

        <div className="card settings-card">
          <div className="card-header">
            <h2 className="card-title">
              <Cpu size={18} /> Serving Environment
            </h2>
          </div>
          <form onSubmit={handleGeneralSave}>
            <div className="setting-row">
              <div>
                <label className="form-label">CUDA Compatibility</label>
                <p className="text-secondary text-sm">Enable if containers fail to start due to a CUDA version mismatch.</p>
              </div>
              <label className="toggle-control" aria-label="Enable CUDA compatibility for vLLM containers">
                <input type="checkbox" checked={enableCUDACompatibility} disabled={saving || loading} onChange={(event) => setEnableCUDACompatibility(event.target.checked)} />
                <span className="toggle-track" aria-hidden="true">
                  <span className="toggle-thumb" />
                </span>
              </label>
            </div>

            <div className="setting-row mt-4">
              <div>
                <label className="form-label">Hugging Face Offline Mode</label>
                <p className="text-secondary text-sm">Blocks outbound Hugging Face requests. Only enable once the model is fully cached locally.</p>
              </div>
              <label className="toggle-control" aria-label="Enable Hugging Face offline mode for vLLM containers">
                <input type="checkbox" checked={hfHubOffline} disabled={saving || loading} onChange={(event) => setHFHubOffline(event.target.checked)} />
                <span className="toggle-track" aria-hidden="true">
                  <span className="toggle-thumb" />
                </span>
              </label>
            </div>

            <div className="form-group serving-vllm-image-field">
              <div className="form-label-row">
                <label className="form-label" htmlFor="vllmImage">vLLM Docker Image</label>
                <a className="inline-doc-link" href={vllmDocsUrl} target="_blank" rel="noreferrer">
                  Docs <ExternalLink size={13} aria-hidden="true" />
                </a>
              </div>
              <input id="vllmImage" type="text" className="form-control" value={vllmImage} onChange={(event) => setVllmImage(event.target.value)} placeholder="vllm/vllm-openai:v0.23.0" disabled={saving || loading} />
              <p className="text-secondary text-sm mt-1">Docker image used to launch serving containers and perform downloads. Defaults to <code>vllm/vllm-openai:v0.23.0</code>.</p>
            </div>

            {!proxyEnabled && (
              <div className="form-group mt-4">
                <label className="form-label" htmlFor="serverBindIP">Default Server Bind IP</label>
                <input id="serverBindIP" type="text" className="form-control" value={serverBindIP} onChange={(event) => setServerBindIP(event.target.value)} placeholder="0.0.0.0" disabled={saving || loading} />
                <p className="text-secondary text-sm mt-1">Applies to new direct model servers. Proxy mode keeps model servers on loopback.</p>
              </div>
            )}

            <button type="submit" className="btn btn-primary serving-environment-save" disabled={saving || loading}>
              {saving ? 'Saving...' : 'Save environment'}
            </button>
          </form>
        </div>
      </div>

      {generatedVllmToken && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="generated-vllm-key-title">
          <div className="modal-content api-key-modal-content">
            <h3 id="generated-vllm-key-title">Generated vLLM API Key</h3>
            <div className="modal-body">
              <p>This token has been placed in the vLLM API key field. Save the key below to apply it to new serving containers.</p>
              <div className="inline-control generated-token-row">
                <div className="generated-token-box">{generatedVllmToken}</div>
                <button className="btn btn-outline btn-compact" type="button" onClick={handleCopyGeneratedVllmToken} aria-label="Copy generated vLLM API key">
                  <Copy size={14} />
                </button>
              </div>
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" type="button" onClick={() => setGeneratedVllmToken('')}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {viewingVllmAPIKey && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="stored-vllm-key-title">
          <div className="modal-content api-key-modal-content">
            <h3 id="stored-vllm-key-title">Stored vLLM API Key</h3>
            <div className="modal-body">
              <p>Use this key for requests to containers that were launched with the global vLLM API key.</p>
              {revealVllmAPIKeyLoading && <p className="text-secondary text-sm">Loading key...</p>}
              {revealVllmAPIKeyError && <div className="alert alert-danger compact-alert">{revealVllmAPIKeyError}</div>}
              {!revealVllmAPIKeyLoading && !revealVllmAPIKeyError && (
                <div className="inline-control generated-token-row">
                  <div className="generated-token-box">{revealedVllmAPIKey || 'No key configured'}</div>
                  <button className="btn btn-outline btn-compact" type="button" onClick={handleCopyRevealedVllmAPIKey} disabled={!revealedVllmAPIKey} aria-label="Copy stored vLLM API key">
                    <Copy size={14} />
                  </button>
                </div>
              )}
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" type="button" onClick={closeVllmAPIKeyModal}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {pendingCredentialRemoval && (
        <ConfirmModal
          title={pendingCredentialRemoval === 'hf' ? 'Remove Hugging Face Token?' : 'Remove vLLM API Key?'}
          confirmLabel={pendingCredentialRemoval === 'hf' ? 'Remove token' : 'Remove key'}
          danger
          loading={saving}
          onCancel={() => setPendingCredentialRemoval(null)}
          onConfirm={confirmCredentialRemoval}
        >
          <p>
            {pendingCredentialRemoval === 'hf'
              ? 'Model downloads will no longer use a stored Hugging Face token.'
              : 'New serving containers will start without vLLM API authentication.'}
          </p>
        </ConfirmModal>
      )}
    </div>
  );
};
