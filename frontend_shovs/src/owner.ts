const OWNER_STORAGE_KEY = 'shovs_owner_id';

function createOwnerId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `owner_${Date.now().toString(16)}_${Math.random().toString(16).slice(2, 10)}`;
}

export function getOwnerId(): string {
  if (typeof window === 'undefined') return 'local-owner';
  let ownerId = window.localStorage.getItem(OWNER_STORAGE_KEY);
  if (!ownerId) {
    ownerId = createOwnerId();
    window.localStorage.setItem(OWNER_STORAGE_KEY, ownerId);
  }
  return ownerId;
}

export function withOwnerQuery(path: string): string {
  if (typeof window === 'undefined') return path;
  const url = new URL(path, window.location.origin);
  url.searchParams.set('owner_id', getOwnerId());
  return `${url.pathname}${url.search}`;
}

export function withOwnerPayload<T extends Record<string, unknown>>(payload: T): T & { owner_id: string } {
  return {
    ...payload,
    owner_id: getOwnerId(),
  };
}

export function appendOwnerId(formData: FormData): FormData {
  formData.set('owner_id', getOwnerId());
  return formData;
}
