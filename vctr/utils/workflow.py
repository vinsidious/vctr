from functools import wraps
from tqdm import tqdm

WORKFLOW_FUNCTIONS = []


def workflow(name):
    def register_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        WORKFLOW_FUNCTIONS.append((name, wrapper))
        return wrapper

    return register_function


def run_workflow(workflow_name=None):
    workflow_functions = WORKFLOW_FUNCTIONS
    if workflow_name:
        workflow_functions = [(name, func) for name, func in workflow_functions if name == workflow_name]
    for name, func in tqdm(workflow_functions, desc='Workflow'):
        args_str = ', '.join([repr(arg) for arg in func.args])
        kwargs_str = ', '.join([f'{k}={repr(v)}' for k, v in func.kwargs.items()])
        if args_str and kwargs_str:
            print(f'Running function {name} with args: {args_str} and kwargs: {kwargs_str}')
        elif args_str:
            print(f'Running function {name} with args: {args_str}')
        elif kwargs_str:
            print(f'Running function {name} with kwargs: {kwargs_str}')
        else:
            print(f'Running function {name}')
        func()
