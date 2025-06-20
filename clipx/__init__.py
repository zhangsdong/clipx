__version__ = '0.5.0'

class LazyLoader:
    def __init__(self, module_name, package):
        self.module_name = module_name
        self.package = package
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name, self.package)
        return getattr(self._module, name)

core = LazyLoader('.core', 'clipx')

def remove(*args, **kwargs):
    return core.remove(*args, **kwargs)