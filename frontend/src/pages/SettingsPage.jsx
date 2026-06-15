import { useEffect, useState } from 'react';
import { Activity, Copy, Cpu, KeyRound, RefreshCw, Settings } from 'lucide-react';

import { ConfirmModal } from '../components/ConfirmModal.jsx';
import { useAutoDismiss } from '../hooks/useAutoDismiss.js';
import { getErrorDetail, getSettings, saveSettings } from '../lib/api.js';
import { ApiKeyField } from '../components/settings/ApiKeyField.jsx';
import { ResourceMeter } from '../components/settings/ResourceMeter.jsx';

const VLLM_TOKEN_PREFIX = 'vllm_';
const VLLM_TOKEN_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

const generateVLLMToken = () => {
  const bytes = new Uint8Array(40);
  crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (byte) => VLLM_TOKEN_CHARS[byte % VLLM_TOKEN_CHARS.length]).join('');
  return `${VLLM_TOKEN_PREFIX}${suffix}`;
};

const credentialStatus = (isSet, suffix) => {
  if (!isSet) return 'Not Configured';
  return suffix ? `Configured · ends in …${suffix}` : 'Configured';
};

export const SettingsPage = ({ fetchData, proxyEnabled = false }) => {
  const [hfCache, setHfCache] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [hfTokenSet, setHfTokenSet] = useState(false);
  const [hfTokenSuffix, setHfTokenSuffix] = useState('');
  const [editingHfToken, setEditingHfToken] = useState(false);
  const [vllmAPIKey, setVllmAPIKey] = useState('');
  const [vllmAPIKeySet, setVllmAPIKeySet] = useState(false);
  const [vllmAPIKeySuffix, setVllmAPIKeySuffix] = useState('');
  const [editingVllmAPIKey, setEditingVllmAPIKey] = useState(false);
  const [inferenceAPIKey, setInferenceAPIKey] = useState('');
  const [inferenceAPIKeySet, setInferenceAPIKeySet] = useState(false);
  const [updateInferenceAPIKey, setUpdateInferenceAPIKey] = useState(false);
  const [clearInferenceAPIKey, setClearInferenceAPIKey] = useState(false);
  const [enableCUDACompatibility, setEnableCUDACompatibility] = useState(false);
  const [hfHubOffline, setHFHubOffline] = useState(false);
  const [diskData, setDiskData] = useState(null);
  const [ramData, setRamData] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [generatedVllmToken, setGeneratedVllmToken] = useState('');
  const [pendingCredentialRemoval, setPendingCredentialRemoval] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useAutoDismiss(message, setMessage);
  useAutoDismiss(error, setError);

  const applySettingsData = (data) => {
    setHfCache(data.hf_cache);
    setHfToken('');
    setHfTokenSet(Boolean(data.hf_token_set));
    setHfTokenSuffix(data.hf_token_suffix || '');
    setEditingHfToken(false);
    setVllmAPIKey('');
    setVllmAPIKeySet(Boolean(data.vllm_api_key_set));
    setVllmAPIKeySuffix(data.vllm_api_key_suffix || '');
    setEditingVllmAPIKey(false);
    setGeneratedVllmToken('');
    setInferenceAPIKey('');
    setInferenceAPIKeySet(Boolean(data.inference_api_key_set));
    setUpdateInferenceAPIKey(false);
    setClearInferenceAPIKey(false);
    setEnableCUDACompatibility(Boolean(data.enable_cuda_compatibility));
    setHFHubOffline(Boolean(data.hf_hub_offline));
    setDiskData(data);
    setRamData(data);
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
  });

  const saveAndRefresh = async (payload, successMessage) => {
    setSaving(true);
    try {
      await saveSettings(payload);
      const response = await getSettings();
      applySettingsData(response.data);
      setMessage(successMessage);
      fetchData();
    } catch (requestError) {
      setError(getErrorDetail(requestError, 'Error saving settings'));
    } finally {
      setSaving(false);
    }
  };

  const handleGeneralSave = async (event) => {
    event.preventDefault();
    const requestedHFCache = hfCache.trim();
    setMessage('');
    setError('');

    if (proxyEnabled && updateInferenceAPIKey && !inferenceAPIKey.trim()) {
      setError('Enter an Inference API key or disable the API key update switch.');
      return;
    }

    if (proxyEnabled && updateInferenceAPIKey && /\s/.test(inferenceAPIKey.trim())) {
      setError('Inference API key cannot contain whitespace.');
      return;
    }

    const payload = baseSettingsPayload();
    if (proxyEnabled) {
      payload.inference_api_key = updateInferenceAPIKey ? inferenceAPIKey.trim() : '';
      payload.clear_inference_api_key = clearInferenceAPIKey;
    }

    await saveAndRefresh(
      payload,
      requestedHFCache && requestedHFCache !== hfCache ? `Settings saved. Hugging Face cache path normalized to ${hfCache}.` : 'Settings saved successfully.',
    );
  };

  const handleSaveHfToken = async (event) => {
    event.preventDefault();
    setMessage('');
    setError('');

    if (!hfToken.trim()) {
      setError('Enter a Hugging Face token or cancel replacement.');
      return;
    }

    await saveAndRefresh({ ...baseSettingsPayload(), hf_token: hfToken.trim() }, 'Hugging Face token saved.');
  };

  const handleSaveVllmAPIKey = async (event) => {
    event.preventDefault();
    setMessage('');
    setError('');

    if (!vllmAPIKey.trim()) {
      setError('Enter a vLLM API key or cancel replacement.');
      return;
    }

    if (/\s/.test(vllmAPIKey.trim())) {
      setError('vLLM API key cannot contain whitespace.');
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

  const requestCredentialRemoval = (type) => {
    setPendingCredentialRemoval(type);
  };

  const confirmCredentialRemoval = async () => {
    if (!pendingCredentialRemoval) return;
    setMessage('');
    setError('');

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

  return (
    <div className="settings-layout">
      <div className="stack">
        <div className="card settings-card">
          <div className="card-header">
            <h2 className="card-title">
              <Settings size={18} /> Configuration
            </h2>
          </div>
          {message && <div className="alert alert-success">{message}</div>}
          {error && <div className="alert alert-danger">{error}</div>}
          <form onSubmit={handleGeneralSave}>
            <div className="form-group">
              <label className="form-label" htmlFor="hfCache">Hugging Face Home Directory</label>
              <input id="hfCache" type="text" className="form-control" value={hfCache} onChange={(event) => setHfCache(event.target.value)} required disabled={saving || loading} />
              <p className="text-secondary text-sm mt-1">Absolute path containing the Hugging Face hub directory.</p>
            </div>

            {proxyEnabled && (
              <ApiKeyField
                label="Inference Reverse Proxy API Key"
                description={inferenceAPIKeySet ? 'A key is stored. Enable the switch to replace it.' : 'No key stored. Enable the switch to add one.'}
                placeholder="token-..."
                htmlFor="inferenceAPIKey"
                value={inferenceAPIKey}
                setValue={setInferenceAPIKey}
                enabled={updateInferenceAPIKey}
                setEnabled={setUpdateInferenceAPIKey}
                keySet={inferenceAPIKeySet}
                clearChecked={clearInferenceAPIKey}
                setClearChecked={setClearInferenceAPIKey}
                saving={saving}
                loading={loading}
                helpText="Used to protect the public /v1/ inference endpoints."
              />
            )}

            <button type="submit" className="btn btn-primary mt-4" disabled={saving || loading}>
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
                <button className="btn btn-outline" type="button" onClick={() => setEditingVllmAPIKey(true)} disabled={saving || loading || editingVllmAPIKey}>
                  {vllmAPIKeySet ? 'Replace' : 'Configure'}
                </button>
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
                <p className="text-secondary text-sm">Removes authentication from all new containers. Existing running containers are unaffected.</p>
                <button className="btn btn-danger" type="button" onClick={() => requestCredentialRemoval('vllm')} disabled={saving || loading}>
                  Remove key
                </button>
              </div>
            )}
          </div>
        </div>
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

            <button type="submit" className="btn btn-primary mt-4" disabled={saving || loading}>
              {saving ? 'Saving...' : 'Save environment'}
            </button>
          </form>
        </div>
      </div>

      {generatedVllmToken && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="generated-vllm-key-title">
          <div className="modal-content">
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
