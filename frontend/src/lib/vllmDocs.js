export const getVLLMDocsUrl = (image, path = '') => {
  const text = String(image || '').trim();
  const lastSlash = text.lastIndexOf('/');
  const lastColon = text.lastIndexOf(':');
  const tag = lastColon > lastSlash ? text.slice(lastColon + 1).split('@')[0] : '';
  const version = tag.match(/^(v\d+\.\d+\.\d+)/)?.[1] || 'latest';
  const cleanPath = String(path || '').replace(/^\/+|\/+$/g, '');
  return `https://docs.vllm.ai/en/${version}/${cleanPath ? `${cleanPath}/` : ''}`;
};