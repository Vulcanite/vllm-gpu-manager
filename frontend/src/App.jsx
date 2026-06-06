import { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

import { Layout } from './components/Layout.jsx';
import { Dashboard } from './pages/Dashboard.jsx';
import { LaunchModel } from './pages/LaunchModel.jsx';
import { LoginPage } from './pages/LoginPage.jsx';
import { SettingsPage } from './pages/SettingsPage.jsx';
import { useAutoDismiss } from './hooks/useAutoDismiss.js';
import { fetchOverview, getAuthStatus, getConfig, getErrorDetail, login, logout, stopModel } from './lib/api.js';

const App = () => {
  const [gpus, setGpus] = useState([]);
  const [activeModels, setActiveModels] = useState([]);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [launchConfig, setLaunchConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState('');
  const [gpuError, setGpuError] = useState('');
  const [notice, setNotice] = useState(null);
  const [stopConfirmPort, setStopConfirmPort] = useState(null);
  const [stopError, setStopError] = useState('');
  const [stopping, setStopping] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const notify = useCallback((message, type = 'success') => {
    setNotice({ message, type });
  }, []);

  useAutoDismiss(stopError, setStopError);

  const clearDashboardData = useCallback(() => {
    setGpus([]);
    setActiveModels([]);
    setDownloadedModels([]);
    setGpuError('');
    setFetchError('');
  }, []);

  const checkAuth = useCallback(async () => {
    try {
      const response = await getAuthStatus();
      const required = Boolean(response.data?.required);
      const isAuthenticated = Boolean(response.data?.authenticated);
      setAuthRequired(required);
      setAuthenticated(!required || isAuthenticated);
      setAuthError('');
      if (required && !isAuthenticated) {
        setLoading(false);
      }
    } catch (error) {
      setAuthRequired(true);
      setAuthenticated(false);
      setAuthError(getErrorDetail(error, 'Unable to check authentication'));
      setLoading(false);
    } finally {
      setAuthChecked(true);
    }
  }, []);

  const fetchData = useCallback(async () => {
    try {
      const overview = await fetchOverview();
      setGpus(overview.gpus);
      setActiveModels(overview.activeModels);
      setDownloadedModels(overview.downloadedModels);
      setGpuError(overview.gpuError || '');
      setFetchError(overview.gpuError || '');
    } catch (error) {
      if (error.response?.status === 401) {
        clearDashboardData();
        setAuthRequired(true);
        setAuthenticated(false);
        setAuthError('Session expired. Sign in again.');
        return;
      }
      setGpuError('');
      setFetchError(getErrorDetail(error, 'Unable to refresh vLLM manager data'));
    } finally {
      setLoading(false);
    }
  }, [clearDashboardData]);

  useEffect(() => {
    const initialAuthCheck = setTimeout(() => {
      checkAuth();
    }, 0);

    return () => {
      clearTimeout(initialAuthCheck);
    };
  }, [checkAuth]);

  useEffect(() => {
    if (!authChecked || (authRequired && !authenticated)) {
      return undefined;
    }

    const initialLoad = setTimeout(() => {
      fetchData();
    }, 0);
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') {
        fetchData();
      }
    }, 5000);

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        fetchData();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearTimeout(initialLoad);
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [authChecked, authRequired, authenticated, fetchData]);

  useEffect(() => {
    if (!authChecked || (authRequired && !authenticated)) {
      return undefined;
    }

    let cancelled = false;
    getConfig()
      .then((res) => {
        if (!cancelled && res.data) setLaunchConfig(res.data);
      })
      .catch(() => {
        // Static tuning config; LaunchModel falls back to built-in defaults.
      });

    return () => {
      cancelled = true;
    };
  }, [authChecked, authRequired, authenticated]);

  useEffect(() => {
    if (!notice) return undefined;

    const timeout = setTimeout(() => {
      setNotice(null);
    }, 5000);

    return () => clearTimeout(timeout);
  }, [notice]);

  const requestStop = (port) => {
    setStopError('');
    setStopConfirmPort(port);
  };

  const cancelStop = () => {
    if (stopping) return;
    setStopConfirmPort(null);
  };

  const confirmStop = async () => {
    if (!stopConfirmPort) return;

    setStopping(true);
    setStopError('');
    try {
      await stopModel(Number(stopConfirmPort));
      setStopConfirmPort(null);
      notify(`Stopped server on port ${stopConfirmPort}.`);
      fetchData();
    } catch (error) {
      setStopError(getErrorDetail(error, 'Failed to stop model'));
    } finally {
      setStopping(false);
    }
  };

  const handleLogin = async (password) => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const response = await login(password);
      const required = Boolean(response.data?.required);
      setAuthRequired(required);
      setAuthenticated(true);
      setLoading(true);
      await fetchData();
    } catch (error) {
      setAuthError(getErrorDetail(error, 'Sign in failed'));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } finally {
      clearDashboardData();
      setAuthenticated(false);
      setAuthRequired(true);
      setLoading(false);
      setNotice(null);
    }
  };

  const renderContent = () => {
    if (!authChecked) {
      return (
        <div className="loading-screen">
          <Loader2 className="loader" size={48} />
        </div>
      );
    }

    if (authRequired && !authenticated) {
      return <LoginPage error={authError} loading={authLoading} onLogin={handleLogin} />;
    }

    if (loading && gpus.length === 0 && !fetchError) {
      return (
        <div className="loading-screen">
          <Loader2 className="loader" size={48} />
        </div>
      );
    }

    return (
      <Layout authRequired={authRequired} notice={notice} systemError={fetchError} onDismissNotice={() => setNotice(null)} onLogout={handleLogout} onRetry={fetchData}>
        <Routes>
          <Route
            path="/"
            element={
              <Dashboard
                gpus={gpus}
                activeModels={activeModels}
                stopConfirmPort={stopConfirmPort}
                stopError={stopError}
                stopping={stopping}
                onCancelStop={cancelStop}
                onConfirmStop={confirmStop}
                onRequestStop={requestStop}
              />
            }
          />
          <Route
            path="/launch"
            element={<LaunchModel gpus={gpus} gpuError={gpuError} downloadedModels={downloadedModels} activeModels={activeModels} launchConfig={launchConfig} fetchData={fetchData} notify={notify} />}
          />
          <Route path="/settings" element={<SettingsPage fetchData={fetchData} />} />
        </Routes>
      </Layout>
    );
  };

  return <BrowserRouter>{renderContent()}</BrowserRouter>;
};

export default App;
