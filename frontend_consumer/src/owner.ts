const OWNER_STORAGE_KEY = 'shovs_consumer_owner_id';

function createOwnerId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `consumer_${Date.now().toString(16)}_${Math.random().toString(16).slice(2, 10)}`;
}

export function getOwnerId(): string {
  if (typeof window === 'undefined') return 'consumer-local-owner';
  let ownerId = window.localStorage.getItem(OWNER_STORAGE_KEY);
  if (!ownerId) {
    ownerId = createOwnerId();
    window.localStorage.setItem(OWNER_STORAGE_KEY, ownerId);
  }
  return ownerId;
}

export function appendOwnerId(formData: FormData): FormData {
  formData.set('owner_id', getOwnerId());
  return formData;
}
