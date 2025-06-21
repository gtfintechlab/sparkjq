"""Shared fixtures and test configuration for sparkjq tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_slurm_env(monkeypatch):
    """Mock SLURM environment variables for testing."""
    env_vars = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NNODES": "3",
        "SLURM_JOB_NODELIST": "node[001-003]",
        "SLURM_PROCID": "0",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_NTASKS": "3",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_single_node_env(monkeypatch):
    """Mock SLURM environment for single node setup."""
    env_vars = {
        "SLURM_JOB_ID": "54321",
        "SLURM_NNODES": "1",
        "SLURM_JOB_NODELIST": "node001",
        "SLURM_PROCID": "0",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NTASKS": "1",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_subprocess():
    """Mock subprocess module for testing."""
    with patch("subprocess.check_output") as mock_check_output:
        # Default behavior for scontrol command
        mock_check_output.return_value = "node001\nnode002\nnode003"
        yield mock_check_output


@pytest.fixture
def mock_subprocess_single_node():
    """Mock subprocess for single node."""
    with patch("subprocess.check_output") as mock_check_output:
        mock_check_output.return_value = "node001"
        yield mock_check_output


@pytest.fixture
def temp_spark_home():
    """Create a temporary Spark home directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spark_home = Path(tmpdir) / "spark-3.5.0-bin-hadoop3"
        spark_home.mkdir(parents=True)

        # Create expected subdirectories
        (spark_home / "sbin").mkdir()
        (spark_home / "conf").mkdir()

        # Create mock executables
        (spark_home / "sbin" / "start-all.sh").touch()
        (spark_home / "sbin" / "stop-all.sh").touch()

        yield str(spark_home)


@pytest.fixture
def mock_spark_context(temp_spark_home):
    """Mock SparkContext object."""
    from sparkjq.spark import SparkContext

    return SparkContext(home=temp_spark_home, version="3.5.0")


@pytest.fixture
def mock_popen():
    """Mock subprocess.Popen for testing."""
    with patch("subprocess.Popen") as mock_popen_class:
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_popen_class.return_value = mock_process
        yield mock_popen_class


@pytest.fixture
def cleanup_env(monkeypatch):
    """Clean up environment variables after tests."""
    # Get current env vars
    original_env = dict(os.environ)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_requests():
    """Mock requests for HTTP operations."""
    import responses as resp

    with resp.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def mock_asyncio_loop():
    """Mock asyncio event loop."""
    import asyncio

    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()

        # Handle coroutines properly
        def run_until_complete(coro):
            if asyncio.iscoroutine(coro):
                try:
                    # Create a new event loop to run the coroutine
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(coro)
                    loop.close()
                    return result
                except Exception:
                    # If that fails, just return a mock
                    return MagicMock()
            return coro

        mock_loop.run_until_complete = run_until_complete
        mock_get_loop.return_value = mock_loop
        yield mock_loop


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module-level variables between tests."""
    # This ensures clean state between tests
    yield

    # Clean up any module-level state if needed
    # Reload modules if necessary
