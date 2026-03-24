"""Encrypted local credential vault. Falls back to os.getenv transparently."""
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet

_VAULT = Path(os.getenv("VAULT_PATH", ".vault.enc"))
_KEY = Path(os.getenv("VAULT_KEY_PATH", ".vault.key"))


def _f() -> Fernet:
    if not _KEY.exists():
        key = Fernet.generate_key()
        _KEY.write_bytes(key)
        try:
            _KEY.chmod(0o600)
        except Exception as e:
            print(f"[vault] Warning: could not tighten key permissions: {e}")
    return Fernet(_KEY.read_bytes())


def vault_set(name: str, value: str):
    f = _f()
    data = {}
    if _VAULT.exists():
        data = json.loads(f.decrypt(_VAULT.read_bytes()))
    data[name] = value
    _VAULT.write_bytes(f.encrypt(json.dumps(data).encode()))


def vault_get(name: str) -> str:
    try:
        f = _f()
        if _VAULT.exists():
            data = json.loads(f.decrypt(_VAULT.read_bytes()))
            if name in data:
                return data[name]
    except Exception:
        pass
    return os.getenv(name, "")
