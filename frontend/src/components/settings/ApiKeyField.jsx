export const ApiKeyField = ({
  label,
  description,
  placeholder,
  htmlFor,
  value,
  setValue,
  enabled,
  setEnabled,
  keySet,
  clearChecked,
  setClearChecked,
  saving,
  loading,
  helpText,
}) => (
  <div className="form-group">
    <div className="setting-row">
      <div>
        <label className="form-label">{label}</label>
        <p className="text-secondary text-sm">{description}</p>
      </div>
      <label className="toggle-control" aria-label={`Enable ${label} update`}>
        <input
          type="checkbox"
          checked={enabled}
          disabled={saving || loading || (!!clearChecked && !!setClearChecked)}
          onChange={(event) => {
            setEnabled(event.target.checked);
            if (event.target.checked) {
              setClearChecked?.(false);
            } else {
              setValue('');
            }
          }}
        />
        <span className="toggle-track" aria-hidden="true">
          <span className="toggle-thumb" />
        </span>
      </label>
    </div>

    {enabled && (
      <div className="token-update-field">
        <label className="form-label" htmlFor={htmlFor}>New {label}</label>
        <input
          id={htmlFor}
          type="password"
          className="form-control"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={placeholder}
          autoComplete="off"
          disabled={saving || loading}
        />
        {helpText && <p className="text-secondary text-sm mt-1">{helpText}</p>}
      </div>
    )}

    {keySet && setClearChecked && (
      <label className="checkbox-control" htmlFor={`clear${htmlFor}`}>
        <input
          id={`clear${htmlFor}`}
          type="checkbox"
          checked={clearChecked}
          disabled={saving || loading}
          onChange={(event) => {
            setClearChecked(event.target.checked);
            if (event.target.checked) {
              setEnabled(false);
              setValue('');
            }
          }}
        />
        Clear stored {label.toLowerCase()}
      </label>
    )}
  </div>
);
