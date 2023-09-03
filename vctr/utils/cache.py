import pandas as pd
import pickle
import hashlib
import os


CACHE_DIR = '/Users/vince/vctr/data/cache'


def cache_plz(filepath=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            no_cache = kwargs.pop('no_cache', False)

            if no_cache:
                return func(*args, **kwargs)

            # Build cache key from function name, args, and kwargs
            cache_key = func.__name__
            cache_key += '_' + '_'.join(str(arg) for arg in args)
            sorted_kwargs = sorted(kwargs.items(), key=lambda x: x[0])
            for key, value in sorted_kwargs:
                cache_key += f'_{key}={value}'

            # Incorporate filepath hash into cache key
            if filepath is not None:
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                cache_key += '_' + file_hash

            # Check cache for existing result
            cache_file = f'{CACHE_DIR}/{cache_key}.pkl'
            if not no_cache and os.path.isfile(cache_file):
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                return result

            # Invoke the function and cache the result
            result = func(*args, **kwargs)
            if not no_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return result

        return wrapper

    return decorator
