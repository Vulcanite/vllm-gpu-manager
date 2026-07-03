const VLLM_TOKEN_PREFIX = 'vllm_';
const VLLM_TOKEN_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

export const generateVLLMToken = () => {
  const bytes = new Uint8Array(40);
  crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, (byte) => VLLM_TOKEN_CHARS[byte % VLLM_TOKEN_CHARS.length]).join('');
  return `${VLLM_TOKEN_PREFIX}${suffix}`;
};
