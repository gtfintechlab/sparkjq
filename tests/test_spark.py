"""Tests for the Spark module."""

import os
import socket
import sys
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests
import responses
from sparkjq.spark import (
    SPARKJQ_HOME,
    SparkContext,
    get_spark_context,
    is_port_open,
    make_workers_file,
    setup_spark_environment_variables,
)


class TestSparkContext:
    """Tests for SparkContext NamedTuple."""

    def test_spark_context_creation(self):
        """Test creating a SparkContext object."""
        ctx = SparkContext(home="/opt/spark", version="3.5.0")
        assert ctx.home == "/opt/spark"
        assert ctx.version == "3.5.0"


class TestGetSparkContext:
    """Tests for get_spark_context function."""

    @pytest.mark.asyncio
    async def test_get_spark_context_already_exists(self, monkeypatch):
        """Test when Spark is already installed."""
        # Mock pyspark version
        with patch("pyspark.__version__", "3.5.0"):
            spark_home = os.path.join(SPARKJQ_HOME, "spark-3.5.0-bin-hadoop3")

            # Mock that directory exists
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True

                ctx = await get_spark_context()

                assert ctx.home == spark_home
                assert ctx.version == "3.5.0"
                mock_exists.assert_called_once_with(spark_home)

    @pytest.mark.asyncio
    async def test_get_spark_context_download(self, mock_requests):
        """Test downloading Spark when not installed."""
        with patch("pyspark.__version__", "3.5.0"):
            spark_home = os.path.join(SPARKJQ_HOME, "spark-3.5.0-bin-hadoop3")

            # Mock that directory doesn't exist
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False

                # Mock makedirs
                with patch("os.makedirs") as mock_makedirs:
                    # Mock HTTP response
                    mock_requests.add(
                        responses.GET,
                        "https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz",
                        body=b"fake tarball content",
                        headers={"content-length": "20"},
                    )

                    # Mock file operations
                    with patch("builtins.open", mock_open()):
                        # Mock tarfile operations
                        with patch("tarfile.open") as mock_tarfile:
                            mock_tar = MagicMock()
                            mock_tar.getmembers.return_value = [MagicMock()]
                            mock_tarfile.return_value.__enter__.return_value = mock_tar

                            # Mock os.remove
                            with patch("os.remove") as mock_remove:
                                # Mock tqdm
                                with patch("tqdm.tqdm") as mock_tqdm:
                                    mock_tqdm.return_value.__enter__.return_value = MagicMock()

                                    ctx = await get_spark_context()

                                    assert ctx.home == spark_home
                                    assert ctx.version == "3.5.0"
                                    mock_makedirs.assert_called()
                                    mock_remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_spark_context_download_error(self, mock_requests):
        """Test error handling during Spark download."""
        with patch("pyspark.__version__", "3.5.0"):
            with patch("os.path.exists", return_value=False):
                with patch("os.makedirs"):
                    # Mock HTTP error response
                    mock_requests.add(
                        responses.GET,
                        "https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz",
                        status=404,
                    )

                    with pytest.raises(requests.HTTPError):
                        await get_spark_context()


class TestIsPortOpen:
    """Tests for is_port_open function."""

    def test_is_port_open_success(self):
        """Test when port is open."""
        with patch("socket.create_connection") as mock_socket:
            mock_socket.return_value.__enter__.return_value = MagicMock()

            result = is_port_open("localhost", 7077)

            assert result is True
            mock_socket.assert_called_once_with(("localhost", 7077), timeout=2)

    def test_is_port_open_failure(self):
        """Test when port is closed or connection fails."""
        with patch("socket.create_connection") as mock_socket:
            mock_socket.side_effect = OSError("Connection refused")

            result = is_port_open("localhost", 7077)

            assert result is False

    def test_is_port_open_timeout(self):
        """Test when connection times out."""
        with patch("socket.create_connection") as mock_socket:
            mock_socket.side_effect = socket.timeout("Connection timed out")

            result = is_port_open("localhost", 7077)

            assert result is False


class TestSetupSparkEnvironmentVariables:
    """Tests for setup_spark_environment_variables function."""

    def test_setup_spark_env_basic(self, monkeypatch):
        """Test basic environment variable setup."""
        monkeypatch.delenv("PYSPARK_PYTHON", raising=False)

        setup_spark_environment_variables(
            log_dir="/var/log/spark",
            spark_home="/opt/spark",
        )

        assert os.environ["SPARK_HOME"] == "/opt/spark"
        assert os.environ["SPARK_LOG_DIR"] == "/var/log/spark"
        assert os.environ["PYSPARK_PYTHON"] == sys.executable

    def test_setup_spark_env_with_python_executable(self, monkeypatch):
        """Test setup with custom Python executable."""
        setup_spark_environment_variables(
            log_dir="/var/log/spark",
            spark_home="/opt/spark",
            python_executable="/usr/bin/python3.9",
        )

        assert os.environ["PYSPARK_PYTHON"] == "/usr/bin/python3.9"

    def test_setup_spark_env_existing_pyspark_python(self, monkeypatch):
        """Test that existing PYSPARK_PYTHON is preserved when no executable provided."""
        monkeypatch.setenv("PYSPARK_PYTHON", "/custom/python")

        setup_spark_environment_variables(
            log_dir="/var/log/spark",
            spark_home="/opt/spark",
        )

        assert os.environ["PYSPARK_PYTHON"] == "/custom/python"

    def test_setup_spark_env_with_scratch(self, monkeypatch):
        """Test setup with scratch directory."""
        setup_spark_environment_variables(
            log_dir="/var/log/spark",
            spark_home="/opt/spark",
            scratch_dir="/scratch/spark",
        )

        assert os.environ["SPARK_LOCAL_DIRS"] == "/scratch/spark"

    def test_setup_spark_env_no_scratch(self, monkeypatch):
        """Test that SPARK_LOCAL_DIRS is not set when no scratch provided."""
        monkeypatch.delenv("SPARK_LOCAL_DIRS", raising=False)

        setup_spark_environment_variables(
            log_dir="/var/log/spark",
            spark_home="/opt/spark",
        )

        assert "SPARK_LOCAL_DIRS" not in os.environ


class TestMakeWorkersFile:
    """Tests for make_workers_file function."""

    def test_make_workers_file_multiple_hosts(self):
        """Test creating workers file with multiple hostnames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spark_home = tmpdir
            hostnames = ["node002", "node003", "node004"]

            result = make_workers_file(spark_home, hostnames)

            expected_path = os.path.join(spark_home, "conf", "workers")
            assert result == expected_path

            # Verify file contents
            with open(expected_path) as f:
                contents = f.read()
                assert contents == "node002\nnode003\nnode004\n"

    def test_make_workers_file_empty_list(self):
        """Test creating workers file with no workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spark_home = tmpdir
            hostnames = []

            result = make_workers_file(spark_home, hostnames)

            expected_path = os.path.join(spark_home, "conf", "workers")
            assert result == expected_path

            # Verify empty file
            with open(expected_path) as f:
                contents = f.read()
                assert contents == ""

    def test_make_workers_file_creates_conf_dir(self):
        """Test that conf directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spark_home = tmpdir
            conf_dir = os.path.join(spark_home, "conf")

            # Ensure conf doesn't exist
            assert not os.path.exists(conf_dir)

            make_workers_file(spark_home, ["node001"])

            # Verify conf was created
            assert os.path.exists(conf_dir)
            assert os.path.isdir(conf_dir)

    def test_make_workers_file_overwrites_existing(self):
        """Test that existing workers file is overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spark_home = tmpdir
            conf_dir = os.path.join(spark_home, "conf")
            os.makedirs(conf_dir)

            # Create existing workers file
            workers_path = os.path.join(conf_dir, "workers")
            with open(workers_path, "w") as f:
                f.write("old_node\n")

            # Create new workers file
            make_workers_file(spark_home, ["new_node1", "new_node2"])

            # Verify new contents
            with open(workers_path) as f:
                contents = f.read()
                assert contents == "new_node1\nnew_node2\n"
                assert "old_node" not in contents
