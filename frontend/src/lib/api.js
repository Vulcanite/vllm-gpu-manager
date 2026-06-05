import axios from 'axios';

const API_BASE = '/api';

axios.defaults.withCredentials = true;

export const getErrorDetail = (error, fallback = 'Request failed') => {
  return error.response?.data?.detail || error.message || fallback;
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

export const saveSettings = (payload) => axios.post(`${API_BASE}/settings`, payload);
