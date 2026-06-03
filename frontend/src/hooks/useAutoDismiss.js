import { useEffect } from 'react';

export const useAutoDismiss = (value, clearValue, delay = 5000) => {
  useEffect(() => {
    if (!value) return undefined;

    const timeout = setTimeout(() => {
      clearValue('');
    }, delay);

    return () => clearTimeout(timeout);
  }, [value, clearValue, delay]);
};
