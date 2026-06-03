import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Moon, Play, Settings, Sun, X } from 'lucide-react';

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

export const Layout = ({ children, notice, systemError, onDismissNotice, onRetry }) => {
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
            <NavItem to="/launch" icon={<Play size={20} />} label="Models" />
            <NavItem to="/settings" icon={<Settings size={20} />} label="Settings" />
          </nav>
        </div>

        <div className="sidebar-footer">
          <button onClick={toggleTheme} className="btn btn-outline theme-toggle">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          </button>
        </div>
      </aside>
      <main className="main-content">
        {notice && (
          <div className={`toast toast-${notice.type || 'success'}`}>
            <span>{notice.message}</span>
            <button className="icon-button" type="button" onClick={onDismissNotice} aria-label="Dismiss notification">
              <X size={16} />
            </button>
          </div>
        )}

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
    </div>
  );
};
