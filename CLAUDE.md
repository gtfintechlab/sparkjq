# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparkJQ is a Python library that provides a wrapper for starting Apache Spark clusters on SLURM (high-performance computing job scheduler). It simplifies launching distributed Spark clusters in SLURM environments by automatically handling node allocation, configuration, and cluster lifecycle management.

## Development Commands

### Installation
```bash
# Install from git repository
uv pip install git+https://github.com/roloza7/sparkjq

# Install in development mode with uv
uv venv
source .venv/bin/activate  # On Linux/Mac
# or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

### Code Quality
```bash
# Run linting
ruff check sparkjq/

# Auto-fix linting issues
ruff check sparkjq/ --fix

# Format code
ruff format sparkjq/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=sparkjq --cov-report=html

# Run specific test file
uv run pytest tests/test_slurm.py

# Run tests in verbose mode
uv run pytest -v
```

### Running the Code
The main usage pattern is through the `SLURMCluster` context manager:
```python
from sparkjq import SLURMCluster

with SLURMCluster() as (sc, spark, master_url):
    # Use sc (SparkContext), spark (SparkSession), and master_url
    pass
```

Note: There is currently no test suite, linting configuration, or CI/CD setup in this project.

## Architecture Overview

### Core Components

1. **`sparkjq/lib.py`** - Main orchestrator
   - `SLURMCluster` class: Implements context manager protocol for cluster lifecycle
   - Handles creation/cleanup of worker configuration files
   - Manages start/stop scripts for Spark

2. **`sparkjq/slurm.py`** - SLURM integration layer
   - Extracts SLURM environment variables (SLURM_JOB_ID, SLURM_NODELIST, etc.)
   - Determines master host from allocated nodes
   - Manages scratch directories and logging paths

3. **`sparkjq/spark.py`** - Spark management
   - Downloads and caches Spark distributions in `~/.cache/spark-jobqueue/`
   - Configures Spark environment variables
   - Creates `spark_worker` script for worker nodes

### Key Architectural Patterns

- **Context Manager Pattern**: `SLURMCluster` uses `__enter__`/`__exit__` for automatic resource cleanup
- **Master-Worker Architecture**: First node in SLURM allocation becomes master, others are workers
- **Lazy Installation**: Spark distributions are downloaded on-demand if not present locally
- **File-based Configuration**: Uses temporary files to coordinate worker nodes across the cluster

### Important Implementation Details

- The project expects to run within a SLURM job allocation (requires SLURM environment variables)
- Spark distributions are cached to avoid repeated downloads
- Worker processes are managed through shell scripts and a workers file
- The master URL follows the pattern: `spark://{master_host}:7077`

## Known Issues

1. **Missing Dependencies**: The `pyproject.toml` doesn't declare required dependencies (pyspark, requests, tqdm)
2. **Export Inconsistency**: `__init__.py` exports `SlurmCluster` but README uses `SLURMCluster`
3. **No Error Handling**: Limited error handling for SLURM/Spark failures
4. **Early Development**: Version 0.1.0 with minimal documentation and no tests

## Improvement Plan

### Fix Critical Errors
1. Remove duplicate `sys` import in lib.py
2. Add missing `sys` import in spark.py  
3. Fix README example to use correct class name `SLURMCluster`
4. Add missing dependencies to pyproject.toml

### Improve Error Handling
5. Add try-except blocks around subprocess calls
6. Validate Spark download with checksums
7. Fix exception re-raising syntax in __exit__
8. Add proper cleanup in case of startup failures

### Fix Resource Management
9. Ensure workers file cleanup happens even on exceptions
10. Add cleanup for temporary directories
11. Add timeouts to subprocess.wait() calls
12. Check port availability before starting

### Remove Dead Code & Fix API
13. Remove unused `is_port_open()` function
14. Remove unused `master()` and `port()` methods
15. Fix context manager to return proper values
16. Extract hardcoded values to constants

### Fix Type Safety
17. Change type annotations to be Python 3.8 compatible
18. Add missing type annotations
19. Add validation for environment variables

### Fix Async Issues
20. Remove unnecessary async from get_spark_context()
21. Fix blocking asyncio call in __init__

### Update Documentation
22. Fix README example to show proper context manager usage
23. Add informative error messages
24. Add docstrings to all public functions

### Development Environment Setup
25. Initialize repository with `uv` package manager
26. Set up `ruff` for linting and formatting
27. Run `ruff` to identify and fix code style issues
28. Update development commands to use `uv` instead of pip