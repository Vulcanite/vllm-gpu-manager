import axios from 'axios';

const API_BASE = '/api';

axios.defaults.withCredentials = true;
axios.defaults.headers.common['X-Requested-With'] = 'XMLHttpRequest';

export const getErrorDetail = (error, fallback = 'Request failed') => {
  return error.response?.data?.detail || error.message || fallback;
};

export const getConfig = () => axios.get(`${API_BASE}/config`);

export const getRecommendations = ({ useCase = 'general', fitOnly = false, limit = 50 } = {}) => {
  const params = new URLSearchParams({ use_case: useCase, fit_only: fitOnly ? 'true' : 'false', limit: String(limit) });
  return axios.get(`${API_BASE}/recommendations?${params.toString()}`);
};

export const getAuthStatus = () => axios.get(`${API_BASE}/auth/status`);

export const login = (password) => axios.post(`${API_BASE}/auth/login`, { password });

export const logout = () => axios.post(`${API_BASE}/auth/logout`);

export const fetchOverview = async () => {
  const [gpuRes, activeRes, downloadedRes] = await Promise.allSettled([
    axios.get(`${API_BASE}/gpus`),
    axios.get(`${API_BASE}/models/active`),
    axios.get(`${API_BASE}/models/downloaded`),
  ]);

  const results = [gpuRes, activeRes, downloadedRes];
  for (const res of results) {
    if (res.status === 'rejected' && res.reason?.response?.status === 401) {
      throw res.reason;
    }
  }

  if (activeRes.status === 'rejected') throw activeRes.reason;
  if (downloadedRes.status === 'rejected') throw downloadedRes.reason;

  return {
    gpus: gpuRes.status === 'fulfilled' ? gpuRes.value.data || [] : [],
    gpuError: gpuRes.status === 'rejected' ? getErrorDetail(gpuRes.reason, 'GPU detection failed') : '',
    activeModels: activeRes.value.data || [],
    downloadedModels: downloadedRes.value.data || [],
  };
};

export const startModel = (payload) => axios.post(`${API_BASE}/models/start`, payload);

export const importModelCommand = (command) => axios.post(`${API_BASE}/models/import-command`, { command });

export const stopModel = (port) => axios.post(`${API_BASE}/models/stop`, { port });

export const restartModel = (containerId) => {
  return axios.post(`${API_BASE}/models/restart`, { container_id: containerId });
};

export const getModelInfo = (model) => {
  return axios.get(`${API_BASE}/models/info?model=${encodeURIComponent(model)}`);
};

export const downloadModel = (model) => axios.post(`${API_BASE}/models/download`, { model });

export const getDownloadProgress = (model) => axios.get(`${API_BASE}/models/download/progress?model=${encodeURIComponent(model)}`);

export const removeModel = (containerId) => axios.post(`${API_BASE}/models/remove`, { container_id: containerId });

export const deleteDownloadedModel = (model) => axios.post(`${API_BASE}/models/downloaded/delete`, { model });

export const getSettings = () => axios.get(`${API_BASE}/settings`);

export const getVllmAPIKey = () => axios.get(`${API_BASE}/settings/vllm-api-key`);

export const getServerAPIKey = (containerName) =>
    axios.get(`${API_BASE}/models/server-api-key?container_name=${encodeURIComponent(containerName)}`);

export const getProxyUsers = () => axios.get(`${API_BASE}/proxy-users`);

export const createProxyUser = (username) => axios.post(`${API_BASE}/proxy-users`, { username });

export const getProxyUserKey = (username) =>
    axios.get(`${API_BASE}/proxy-users/key?username=${encodeURIComponent(username)}`);

export const revokeProxyUser = (username) => axios.post(`${API_BASE}/proxy-users/revoke`, { username });

export const saveSettings = (payload) => axios.post(`${API_BASE}/settings`, payload);

export const getRoutingTable = () => axios.get(`${API_BASE}/routing`);

export const saveRoutingTable = (payload) => axios.post(`${API_BASE}/routing`, payload);

export const getContainerLogs = (containerId) =>
    axios.get(`${API_BASE}/models/logs?container_id=${encodeURIComponent(containerId)}`);

export const streamContainerLogs = (containerId, onChunk, onError, onDone) => {
  const url = `${API_BASE}/models/logs/stream?container_id=${encodeURIComponent(containerId)}`;
  const headers = {};
  if (axios.defaults.headers.common['X-Requested-With']) {
    headers['X-Requested-With'] = axios.defaults.headers.common['X-Requested-With'];
  }
  const controller = new AbortController();

  fetch(url, {
    credentials: 'include',
    headers,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const body = await response.text();
        let detail = body;
        try {
          const json = JSON.parse(body);
          detail = json.detail || body;
        } catch {
          // already have body text
        }
        onError?.(detail);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const read = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              onDone?.();
              return;
            }
            buffer += decoder.decode(value, { stream: true });

            // Split on double-newline to get SSE event blocks
            const blocks = buffer.split(/\r?\n\r?\n/);
            buffer = blocks.pop() || '';

            for (const block of blocks) {
              const trimmed = block.trim();
              if (!trimmed) continue;

              let eventType = 'log';
              const dataLines = [];

              // Parse each line within the block
              const blockLines = trimmed.split('\n');
              for (const line of blockLines) {
                if (line.startsWith('event:')) {
                  eventType = line.slice(6).trim();
                } else if (line.startsWith('data:')) {
                  dataLines.push(line.slice(5));
                }
              }

              // Join multiple data: lines with newlines (standard SSE)
              const data = dataLines.join('\n');

              if (eventType === 'error') {
                onError?.(data);
                return;
              }
              if (eventType === 'close') {
                onDone?.();
                return;
              }
              onChunk?.(data, eventType === 'stdout');
            }
          }
        } catch (err) {
          if (err.name === 'AbortError') return;
          onError?.(err.message || 'Stream error');
        }
      };

      read();
    })
    .catch((err) => {
      if (err.name === 'AbortError') return;
      onError?.(err.message || 'Failed to connect to log stream');
    });

  // Return cleanup function for stop streaming
  return () => controller.abort();
};
