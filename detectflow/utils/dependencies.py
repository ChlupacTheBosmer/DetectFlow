import inspect
import importlib


def get_import_path(func):
    """
    Get the full import path of a function.

    Args:
        func (callable): The function to get the import path for.

    Returns:
        str: Full import path of the function.
    """
    try:
        module = inspect.getmodule(func)
        if module is None:
            raise ValueError("Could not determine the module for the function.")

        module_name = module.__name__
        func_name = func.__qualname__

        return f"{module_name}.{func_name}"
    except Exception as e:
        raise ValueError(f"Could not get import path for function '{func}'. Error: {e}")


def get_callable(import_path: str):
    """
    Get a callable object from its import path.

    Args:
        import_path (str): The import path of the callable.

    Returns:
        callable: The callable object.
    """
    try:
        module_name, function_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{import_path}'. Error: {e}")