import time
from functools import wraps
from contextlib import contextmanager
from typing import Optional

def timeit(description: Optional[str] = None):
    """
    A decorator to time the execution of a function and print the duration.
    
    Args:
        description: Optional custom description to display instead of function name
    """
    def decorator(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Use custom description or function name
            display_name = description or func.__name__
            print(f"⏱️  {display_name:<35} | Time: {total_time:.4f} seconds")
            return result
        return timeit_wrapper
    
    # Handle case where decorator is used without parentheses
    if callable(description):
        return decorator(description)
    return decorator

@contextmanager
def timer(description: str):
    """
    A context manager to time code blocks.
    
    Args:
        description: Description of what is being timed
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"⏱️  {description:<35} | Time: {total_time:.4f} seconds")

def time_function(func, *args, **kwargs):
    """
    Time a function call and return both the result and execution time.
    
    Args:
        func: Function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        tuple: (result, execution_time)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time 