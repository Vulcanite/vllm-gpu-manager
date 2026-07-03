import { useCallback, useEffect, useRef } from 'react';

export const ConfirmModal = ({
  title,
  children,
  confirmLabel,
  cancelLabel = 'Cancel',
  danger = false,
  loading = false,
  onCancel,
  onConfirm,
}) => {
  const overlayRef = useRef(null);
  const contentRef = useRef(null);

  // Escape key dismisses the modal
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && !loading) {
        onCancel();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onCancel, loading]);

  // Focus trap: cycle focus within the modal
  useEffect(() => {
    const content = contentRef.current;
    if (!content) return;

    const focusable = content.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    first.focus();

    const handleTab = (event) => {
      if (event.key !== 'Tab') return;

      if (event.shiftKey) {
        if (document.activeElement === first) {
          event.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };

    content.addEventListener('keydown', handleTab);
    return () => content.removeEventListener('keydown', handleTab);
  }, [loading]);

  // Click on overlay backdrop dismisses
  const handleOverlayClick = useCallback(
    (event) => {
      if (event.target === overlayRef.current && !loading) {
        onCancel();
      }
    },
    [onCancel, loading]
  );

  return (
    <div className="modal-overlay" ref={overlayRef} onClick={handleOverlayClick} role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="modal-content" ref={contentRef}>
        <h3 id="modal-title">{title}</h3>
        <div className="modal-body">{children}</div>
        <div className="modal-actions">
          <button className="btn btn-outline" type="button" onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </button>
          <button className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`} type="button" onClick={onConfirm} disabled={loading}>
            {loading ? 'Working...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
};
