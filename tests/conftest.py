import sys
import types


def _install_slowapi_stub() -> None:
    if "slowapi" in sys.modules:
        return

    slowapi_mod = types.ModuleType("slowapi")
    util_mod = types.ModuleType("slowapi.util")
    errors_mod = types.ModuleType("slowapi.errors")

    class RateLimitExceeded(Exception):
        pass

    class Limiter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def limit(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    async def _rate_limit_exceeded_handler(*args, **kwargs):
        return None

    def get_remote_address(*args, **kwargs):
        return "test"

    slowapi_mod.Limiter = Limiter
    slowapi_mod._rate_limit_exceeded_handler = _rate_limit_exceeded_handler
    util_mod.get_remote_address = get_remote_address
    errors_mod.RateLimitExceeded = RateLimitExceeded

    sys.modules.setdefault("slowapi", slowapi_mod)
    sys.modules.setdefault("slowapi.util", util_mod)
    sys.modules.setdefault("slowapi.errors", errors_mod)


try:
    import slowapi  # type: ignore  # pragma: no cover
except Exception:  # pragma: no cover
    _install_slowapi_stub()
