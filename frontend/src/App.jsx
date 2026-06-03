import { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

import { Layout } from './components/Layout.jsx';
import { Dashboard } from './pages/Dashboard.jsx';
import { LaunchModel } from './pages/LaunchModel.jsx';
import { SettingsPage } from './pages/SettingsPage.jsx';
import { useAutoDismiss } from './hooks/useAutoDismiss.js';
import { fetchOverview, getErrorDetail, stopModel } from './lib/api.js';

const App = () => {
  const [gpus, setGpus] = useState([]);
  const [activeModels, setActiveModels] = useState([]);
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState('');
  const [notice, setNotice] = useState(null);
  const [stopConfirmPort, setStopConfirmPort] = useState(null);
  const [stopError, setStopError] = useState('');
  const [stopping, setStopping] = useState(false);

  const notify = useCallback((message, type = 'success') => {
    setNotice({ message, type });
  }, []);

  useAutoDismiss(stopError, setStopError);

  const fetchData = useCallback(async () => {
    try {
      const overview = await fetchOverview();
      setGpus(overview.gpus);
      setActiveModels(overview.activeModels);
      setDownloadedModels(overview.downloadedModels);
      setFetchError('');
    } catch (error) {
      setFetchError(getErrorDetail(error, 'Unable to refresh vLLM manager data'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = setTimeout(() => {
      fetchData();
    }, 0);
    const interval = setInterval(() => {
      fetchData();
    }, 5000);

    return () => {
      clearTimeout(initialLoad);
      clearInterval(interval);
    };
  }, [fetchData]);

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

  if (loading && gpus.length === 0 && !fetchError) {
    return (
      <div className="loading-screen">
        <Loader2 className="loader" size={48} />
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Layout notice={notice} systemError={fetchError} onDismissNotice={() => setNotice(null)} onRetry={fetchData}>
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
            element={<LaunchModel gpus={gpus} downloadedModels={downloadedModels} activeModels={activeModels} fetchData={fetchData} notify={notify} />}
          />
          <Route path="/settings" element={<SettingsPage fetchData={fetchData} />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
};

export default App;
