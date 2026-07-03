import { useState } from 'react';
import { Loader2, Lock } from 'lucide-react';

export const LoginPage = ({ error, loading, onLogin }) => {
  const [password, setPassword] = useState('');

  const submit = (event) => {
    event.preventDefault();
    onLogin(password);
  };

  return (
    <main className="login-screen">
      <form className="login-panel" onSubmit={submit}>
        <div className="login-header">
          <div className="login-icon">
            <Lock size={22} />
          </div>
          <div>
            <h1>vLLM Manager</h1>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="dashboard-password">
            Password
          </label>
          <input
            id="dashboard-password"
            className="form-control"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            autoFocus
          />
        </div>

        {error && <div className="alert alert-danger compact-alert">{error}</div>}

        <button className="btn btn-primary full-width" type="submit" disabled={loading || password.trim() === ''}>
          {loading ? <Loader2 className="loader-inline" size={16} /> : null}
          Sign in
        </button>
      </form>
    </main>
  );
};
