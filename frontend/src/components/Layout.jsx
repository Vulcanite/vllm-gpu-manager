import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, LogOut, Moon, Play, Settings, Sparkles, Sun, X, Network, FileText, ShieldCheck } from 'lucide-react';

const NavItem = ({ to, icon, label }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link to={to} className={`nav-item ${isActive ? 'active' : ''}`}>
      {icon}
      <span>{label}</span>
    </Link>
  );
};

export const Layout = ({ authRequired = false, children, notice, systemError, onDismissNotice, onLogout, onRetry, proxyEnabled = false }) => {
  const [theme, setTheme] = useState(localStorage.getItem('vllm-theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('vllm-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div>
          <div className="sidebar-header">
            <Activity className="header-icon" size={28} />
            <h2>vLLM Manager</h2>
          </div>
          <nav className="nav-menu">
            <NavItem to="/" icon={<Activity size={20} />} label="Dashboard" />
            <NavItem to="/recommendations" icon={<Sparkles size={20} />} label="Recommended" />
            <NavItem to="/launch" icon={<Play size={20} />} label="Models" />
            {proxyEnabled && <NavItem to="/routing" icon={<Network size={20} />} label="Proxy Gateway" />}
            <NavItem to="/logs" icon={<FileText size={20} />} label="Logs" />
            <NavItem to="/audit" icon={<ShieldCheck size={20} />} label="Audit" />
            <NavItem to="/settings" icon={<Settings size={20} />} label="Settings" />
          </nav>
        </div>

        <div className="sidebar-footer">
          {authRequired && (
            <button onClick={onLogout} className="btn btn-outline theme-toggle" type="button">
              <LogOut size={18} />
              <span>Sign Out</span>
            </button>
          )}
          <button onClick={toggleTheme} className="btn btn-outline theme-toggle" type="button">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>
      </aside>
      <main className="main-content">
        {systemError && (
          <div className="alert alert-danger alert-row">
            <span>{systemError}</span>
            {onRetry && (
              <button className="btn btn-outline" type="button" onClick={onRetry}>
                Retry
              </button>
            )}
          </div>
        )}

        {children}
      </main>

      {notice && (
        <div className={`toast toast-popup toast-${notice.type || 'success'}`} role="status" aria-live="polite">
          <span>{notice.message}</span>
          <button className="icon-button" type="button" onClick={onDismissNotice} aria-label="Dismiss notification">
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
};
