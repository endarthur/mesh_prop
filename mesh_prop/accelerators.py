"""
Optional acceleration utilities using Numba JIT compilation and parallel processing.

This module provides optional performance enhancements that gracefully fall back
to standard implementations if the required dependencies are not available.
"""

import warnings

# Try to import numba for JIT compilation
try:
    from numba import jit as numba_jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def numba_jit(*args, **kwargs):
        """Fallback decorator when numba is not installed."""
        def decorator(func):
            return func
        # Handle both @jit and @jit() syntax
        if args and callable(args[0]):
            return args[0]
        return decorator


# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Parallel = None
    delayed = None


def get_optimal_n_jobs(n_jobs='auto'):
    """
    Determine the optimal number of parallel jobs to use.
    
    Parameters
    ----------
    n_jobs : int or 'auto'
        Number of parallel jobs. If 'auto', uses number of CPU cores.
        If 1 or less, disables parallelization.
    
    Returns
    -------
    int or None
        Number of jobs to use, or None if parallelization is disabled.
    """
    if n_jobs == 'auto':
        import os
        return max(1, os.cpu_count() or 1)
    elif isinstance(n_jobs, int):
        return n_jobs if n_jobs > 1 else None
    return None


def parallel_map(func, iterable, n_jobs='auto', backend='multiprocessing'):
    """
    Apply function to iterable in parallel if joblib is available.
    
    Falls back to sequential map if joblib is not installed or n_jobs=1.
    
    Parameters
    ----------
    func : callable
        Function to apply to each element
    iterable : iterable
        Elements to process
    n_jobs : int or 'auto', optional
        Number of parallel jobs. If 'auto', uses CPU count.
        If 1 or less, uses sequential processing.
    backend : str, optional
        Joblib backend to use ('loky', 'multiprocessing', 'threading').
        Default is 'multiprocessing'.
    
    Returns
    -------
    list
        Results from applying func to each element
    """
    n_jobs_actual = get_optimal_n_jobs(n_jobs)
    
    # Use sequential processing if parallelization is disabled or unavailable
    if n_jobs_actual is None or n_jobs_actual == 1 or not HAS_JOBLIB:
        return list(map(func, iterable))
    
    # Use parallel processing
    try:
        return Parallel(n_jobs=n_jobs_actual, backend=backend)(
            delayed(func)(item) for item in iterable
        )
    except Exception as e:
        warnings.warn(
            f"Parallel processing failed ({e}), falling back to sequential processing",
            RuntimeWarning
        )
        return list(map(func, iterable))


def check_numba_available():
    """Check if Numba is available for JIT compilation."""
    return HAS_NUMBA


def check_joblib_available():
    """Check if joblib is available for parallel processing."""
    return HAS_JOBLIB


def warn_if_accelerator_unavailable(accelerator='numba'):
    """
    Warn user if requested accelerator is not available.
    
    Parameters
    ----------
    accelerator : str
        Name of accelerator ('numba' or 'joblib')
    """
    if accelerator == 'numba' and not HAS_NUMBA:
        warnings.warn(
            "Numba not installed. Install with 'pip install numba' or "
            "'pip install mesh_prop[speedups]' for ~2-5× speedup on ray-casting operations.",
            RuntimeWarning
        )
    elif accelerator == 'joblib' and not HAS_JOBLIB:
        warnings.warn(
            "Joblib not installed. Install with 'pip install joblib' or "
            "'pip install mesh_prop[speedups]' for parallel processing speedups.",
            RuntimeWarning
        )
