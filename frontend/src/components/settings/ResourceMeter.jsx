import { formatBytes } from '../../lib/format.js';

const getStats = (data, prefix) => {
  const total = data[`${prefix}_total`];
  if (!total) return null;

  const used = data[`${prefix}_used`];
  return {
    total,
    used,
    free: data[`${prefix}_free`],
    percent: (used / total) * 100,
  };
};

export const ResourceMeter = ({ label, data, prefix, freeLabel }) => {
  const stats = getStats(data, prefix);
  if (!stats) return null;

  const fillClass = stats.percent > 90 ? 'danger' : stats.percent > 75 ? 'warning' : 'success';

  return (
    <div>
      <div className="metric-row">
        <span>{label}</span>
        <span className="metric-value">
          {formatBytes(stats.used)} / {formatBytes(stats.total)} ({stats.percent.toFixed(1)}%)
        </span>
      </div>
      <div className="progress-bar-bg">
        <div className={`progress-bar-fill ${fillClass}`} style={{ width: `${stats.percent}%` }} />
      </div>
      <div className="text-right">
        <span className="text-secondary text-sm">
          {formatBytes(stats.free)} {freeLabel}
        </span>
      </div>
    </div>
  );
};
