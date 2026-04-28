import { useEffect, useState } from 'react';
import type { Attachment } from '../useAgent';

const formatBytes = (bytes: number): string => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
};

const isImage = (file: File) => file.type.startsWith('image/');
const isText = (file: File) =>
  file.type.startsWith('text/') ||
  /\.(md|txt|json|csv|tsv|yaml|yml|toml|log|py|ts|tsx|js|jsx|html|css|sh)$/i.test(
    file.name,
  );

export function PendingFileChip({
  attachment,
  onRemove,
  onOpen,
}: {
  attachment: Attachment;
  onRemove: () => void;
  onOpen: () => void;
}) {
  const { file, dataURL } = attachment;
  return (
    <span
      className='shovs-chip shovs-attach-chip'
      title={`${file.name} · ${formatBytes(file.size)}`}
    >
      <button type='button' className='shovs-attach-chip-body' onClick={onOpen}>
        {dataURL ? (
          <img src={dataURL} alt='' className='shovs-attach-thumb' />
        ) : (
          <span className='shovs-attach-icon'>📄</span>
        )}
        <span className='shovs-attach-meta'>
          <span className='shovs-attach-name'>{file.name}</span>
          <span className='shovs-attach-size'>{formatBytes(file.size)}</span>
        </span>
      </button>
      <button type='button' onClick={onRemove} aria-label='Remove attachment'>
        ✕
      </button>
    </span>
  );
}

export function MessageFileTile({
  attachment,
  onOpen,
}: {
  attachment: Attachment;
  onOpen: () => void;
}) {
  const { file, dataURL } = attachment;
  return (
    <button
      type='button'
      className={`shovs-msg-file-tile ${dataURL ? 'image' : 'doc'}`}
      onClick={onOpen}
      title={`${file.name} · ${formatBytes(file.size)}`}
    >
      {dataURL ? (
        <img src={dataURL} alt={file.name} />
      ) : (
        <div className='shovs-msg-file-tile-icon'>📄</div>
      )}
      <div className='shovs-msg-file-tile-meta'>
        <div className='shovs-msg-file-tile-name'>{file.name}</div>
        <div className='shovs-msg-file-tile-size'>{formatBytes(file.size)}</div>
      </div>
    </button>
  );
}

export function FileViewerModal({
  attachment,
  onClose,
}: {
  attachment: Attachment | null;
  onClose: () => void;
}) {
  const [textPreview, setTextPreview] = useState<string | null>(null);
  const [textError, setTextError] = useState<string | null>(null);

  useEffect(() => {
    if (!attachment) return;
    const { file } = attachment;
    if (!isText(file) || isImage(file)) {
      setTextPreview(null);
      setTextError(null);
      return;
    }
    let cancelled = false;
    const reader = new FileReader();
    reader.onload = (e) => {
      if (cancelled) return;
      const raw = String(e.target?.result || '');
      setTextPreview(
        raw.length > 200_000 ? `${raw.slice(0, 200_000)}\n\n…(truncated)` : raw,
      );
    };
    reader.onerror = () => {
      if (cancelled) return;
      setTextError('Could not read file as text.');
    };
    reader.readAsText(file);
    return () => {
      cancelled = true;
    };
  }, [attachment]);

  useEffect(() => {
    if (!attachment) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [attachment, onClose]);

  if (!attachment) return null;
  const { file, dataURL } = attachment;
  const objectURL =
    !dataURL && !isText(file) ? URL.createObjectURL(file) : null;

  return (
    <div className='shovs-file-modal-backdrop' onClick={onClose}>
      <div className='shovs-file-modal' onClick={(e) => e.stopPropagation()}>
        <div className='shovs-file-modal-head'>
          <div className='shovs-file-modal-title'>
            <span className='shovs-file-modal-name'>{file.name}</span>
            <span className='shovs-file-modal-meta'>
              {file.type || 'file'} · {formatBytes(file.size)}
            </span>
          </div>
          <button
            type='button'
            className='shovs-file-modal-close'
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        <div className='shovs-file-modal-body'>
          {dataURL ? (
            <img
              src={dataURL}
              alt={file.name}
              className='shovs-file-modal-image'
            />
          ) : isText(file) ? (
            textError ? (
              <div className='shovs-file-modal-error'>{textError}</div>
            ) : (
              <pre className='shovs-file-modal-text'>
                {textPreview ?? 'Loading…'}
              </pre>
            )
          ) : objectURL ? (
            <iframe
              title={file.name}
              src={objectURL}
              className='shovs-file-modal-iframe'
            />
          ) : (
            <div className='shovs-file-modal-empty'>No preview available.</div>
          )}
        </div>
      </div>
    </div>
  );
}
