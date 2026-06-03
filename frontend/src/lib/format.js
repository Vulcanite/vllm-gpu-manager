export const formatBytes = (bytes) => {
  if (bytes === 0 || !bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const validIndex = Math.min(i, sizes.length - 1);
  return `${parseFloat((bytes / Math.pow(k, validIndex)).toFixed(2))} ${sizes[validIndex]}`;
};

export const getMemoryColor = (percent) => {
  if (percent > 85) return 'fill-red';
  if (percent > 60) return 'fill-yellow';
  return 'fill-green';
};
