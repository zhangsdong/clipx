__version__ = '0.4.2'

class LazyLoader:
    def __init__(self, lib_name):
        self.lib_name = lib_name
        self._lib = None

    def __getattr__(self, name):
        if self._lib is None:
            import importlib
            self._lib = importlib.import_module(self.lib_name)
        return getattr(self._lib, name)


core = LazyLoader('.core')


def remove(*args, **kwargs):
    return core.remove(*args, **kwargs)