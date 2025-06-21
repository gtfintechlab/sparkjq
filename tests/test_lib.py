"""Tests for the main lib module."""

import asyncio
import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from sparkjq.lib import SLURMCluster
from sparkjq.slurm import SlurmContext
from sparkjq.spark import SparkContext


class TestSLURMCluster:
    """Tests for SLURMCluster class."""

    @pytest.fixture
    def mock_dependencies(self, mock_slurm_env, mock_subprocess):
        """Mock all dependencies for SLURMCluster."""
        # Mock get_slurm_context
        mock_slurm_ctx = SlurmContext(
            nnodes=3,
            rank=0,
            ncpus=4,
            world_size=3,
            hostname="node001",
            port=7077,
            scratch="/tmp/spark-jobqueue-123",
        )
        
        # Mock get_spark_context
        mock_spark_ctx = SparkContext(
            home="/opt/spark-3.5.0",
            version="3.5.0"
        )
        
        # Mock asyncio
        async def mock_get_spark_context():
            return mock_spark_ctx
        
        with patch("sparkjq.lib.get_slurm_context") as mock_get_slurm:
            mock_get_slurm.return_value = mock_slurm_ctx
            with patch("sparkjq.lib.get_spark_context") as mock_get_spark:
                mock_get_spark.return_value = mock_get_spark_context()
                with patch("sparkjq.lib.build_log_dir") as mock_build_log:
                    mock_build_log.return_value = "/var/log/spark/12345-log"
                    with patch("asyncio.get_event_loop") as mock_loop:
                        mock_loop.return_value.run_until_complete.return_value = mock_spark_ctx
                        yield {
                            "slurm_context": mock_slurm_ctx,
                            "spark_context": mock_spark_ctx,
                            "get_slurm": mock_get_slurm,
                            "get_spark": mock_get_spark,
                            "build_log": mock_build_log,
                        }

    def test_slurm_cluster_init_default(self, mock_dependencies):
        """Test SLURMCluster initialization with default parameters."""
        cluster = SLURMCluster()
        
        assert cluster.slurm_context == mock_dependencies["slurm_context"]
        assert cluster.spark_context == mock_dependencies["spark_context"]
        assert cluster.log_dir == "/var/log/spark/12345-log"
        assert cluster.handle is None
        assert cluster.workers_file is None
        
        # Verify calls
        mock_dependencies["get_slurm"].assert_called_once_with(port=None, scratch=None)
        mock_dependencies["build_log"].assert_called_once_with(None)

    def test_slurm_cluster_init_custom_params(self, mock_dependencies):
        """Test SLURMCluster initialization with custom parameters."""
        cluster = SLURMCluster(
            port=8080,
            scratch_dir="/custom/scratch",
            log_dir="/custom/logs"
        )
        
        # Verify calls with custom params
        mock_dependencies["get_slurm"].assert_called_once_with(
            port=8080, 
            scratch="/custom/scratch"
        )
        mock_dependencies["build_log"].assert_called_once_with("/custom/logs")

    def test_slurm_cluster_enter(self, mock_dependencies, mock_popen):
        """Test context manager __enter__ method."""
        with patch("sparkjq.lib.setup_spark_environment_variables") as mock_setup_env:
            with patch("sparkjq.lib.make_workers_file") as mock_make_workers:
                with patch("sparkjq.lib.get_worker_list") as mock_get_workers:
                    mock_get_workers.return_value = ["node002", "node003"]
                    mock_make_workers.return_value = "/opt/spark-3.5.0/conf/workers"
                    
                    cluster = SLURMCluster()
                    result = cluster.__enter__()
                    
                    # Should return self
                    assert result is cluster
                    
                    # Verify environment setup
                    mock_setup_env.assert_called_once_with(
                        log_dir="/var/log/spark/12345-log",
                        spark_home="/opt/spark-3.5.0",
                        python_executable=sys.executable,
                        scratch_dir="/tmp/spark-jobqueue-123",
                    )
                    
                    # Verify workers file creation
                    mock_make_workers.assert_called_once_with(
                        spark_home="/opt/spark-3.5.0",
                        hostnames=["node002", "node003"],
                    )
                    
                    # Verify start-all.sh was called
                    mock_popen.assert_called_once()
                    call_args = mock_popen.call_args[0][0]
                    assert call_args[0].endswith("start-all.sh")
                    
                    # Verify cluster attributes
                    assert cluster.workers_file == "/opt/spark-3.5.0/conf/workers"
                    assert cluster.handle is not None

    def test_slurm_cluster_exit_normal(self, mock_dependencies, mock_popen):
        """Test context manager __exit__ with normal exit."""
        cluster = SLURMCluster()
        cluster.workers_file = "/opt/spark-3.5.0/conf/workers"
        cluster.handle = MagicMock()
        
        with patch("os.remove") as mock_remove:
            result = cluster.__exit__(None, None, None)
            
            # Should return False (don't suppress exceptions)
            assert result is False
            
            # Verify stop-all.sh was called
            assert mock_popen.call_count == 1
            call_args = mock_popen.call_args[0][0]
            assert call_args[0].endswith("stop-all.sh")
            
            # Verify workers file was removed
            mock_remove.assert_called_once_with("/opt/spark-3.5.0/conf/workers")

    def test_slurm_cluster_exit_with_exception(self, mock_dependencies, mock_popen):
        """Test context manager __exit__ with exception."""
        cluster = SLURMCluster()
        cluster.workers_file = "/opt/spark-3.5.0/conf/workers"
        cluster.handle = MagicMock()
        
        with patch("os.remove") as mock_remove:
            # Test with ValueError exception
            with pytest.raises(ValueError, match="Test error"):
                cluster.__exit__(ValueError, ValueError("Test error"), None)
            
            # Cleanup should still happen
            assert mock_popen.call_count == 1
            mock_remove.assert_called_once()

    def test_slurm_cluster_exit_no_workers_file(self, mock_dependencies, mock_popen):
        """Test __exit__ when workers_file is None."""
        cluster = SLURMCluster()
        cluster.workers_file = None
        cluster.handle = MagicMock()
        
        with patch("os.remove") as mock_remove:
            cluster.__exit__(None, None, None)
            
            # Should not try to remove non-existent file
            mock_remove.assert_not_called()

    def test_slurm_cluster_master_method(self, mock_dependencies):
        """Test master() method returns hostname."""
        cluster = SLURMCluster()
        assert cluster.master() == "node001"

    def test_slurm_cluster_port_method(self, mock_dependencies):
        """Test port() method returns port."""
        cluster = SLURMCluster()
        assert cluster.port() == 7077

    def test_slurm_cluster_full_context_manager(self, mock_dependencies, mock_popen):
        """Test full context manager usage."""
        with patch("sparkjq.lib.setup_spark_environment_variables"):
            with patch("sparkjq.lib.make_workers_file") as mock_make_workers:
                with patch("sparkjq.lib.get_worker_list") as mock_get_workers:
                    with patch("os.remove") as mock_remove:
                        mock_get_workers.return_value = ["node002"]
                        mock_make_workers.return_value = "/opt/spark/conf/workers"
                        
                        with SLURMCluster() as cluster:
                            # Inside context
                            assert cluster.handle is not None
                            assert cluster.workers_file == "/opt/spark/conf/workers"
                        
                        # After context exit
                        # Verify both start and stop were called
                        assert mock_popen.call_count == 2
                        mock_remove.assert_called_once()

    def test_slurm_cluster_exception_in_enter(self, mock_dependencies):
        """Test exception handling in __enter__."""
        with patch("sparkjq.lib.setup_spark_environment_variables") as mock_setup:
            mock_setup.side_effect = Exception("Setup failed")
            
            cluster = SLURMCluster()
            with pytest.raises(Exception, match="Setup failed"):
                cluster.__enter__()

    def test_slurm_cluster_popen_wait_called(self, mock_dependencies, mock_popen):
        """Test that Popen.wait() is called for both start and stop."""
        mock_process = mock_popen.return_value
        
        with patch("sparkjq.lib.setup_spark_environment_variables"):
            with patch("sparkjq.lib.make_workers_file"):
                with patch("sparkjq.lib.get_worker_list", return_value=[]):
                    with patch("os.remove"):
                        with SLURMCluster():
                            pass
                        
                        # wait() should be called twice (start and stop)
                        assert mock_process.wait.call_count == 2