"""Integration tests for sparkjq."""

import os
from unittest.mock import MagicMock, patch

import pytest

import sparkjq
from sparkjq import SLURMCluster


class TestModuleImports:
    """Test module imports and structure."""

    def test_main_import(self):
        """Test that main module can be imported."""
        assert hasattr(sparkjq, "SLURMCluster")

    def test_slurm_cluster_import(self):
        """Test that SLURMCluster is properly exported."""
        from sparkjq import SLURMCluster as ImportedCluster
        from sparkjq.lib import SLURMCluster as DirectCluster
        
        assert ImportedCluster is DirectCluster

    def test_submodule_imports(self):
        """Test that all submodules can be imported."""
        import sparkjq.lib
        import sparkjq.slurm
        import sparkjq.spark
        
        # Verify key functions exist
        assert hasattr(sparkjq.slurm, "get_slurm_context")
        assert hasattr(sparkjq.spark, "get_spark_context")
        assert hasattr(sparkjq.lib, "SLURMCluster")


class TestEndToEndFlow:
    """Test end-to-end flow with all components mocked."""

    @pytest.fixture
    def full_mock_environment(
        self, 
        mock_slurm_env, 
        mock_subprocess, 
        mock_popen,
        mock_requests,
        temp_spark_home
    ):
        """Set up full mock environment for integration tests."""
        # Ensure subprocess mock returns proper format
        mock_subprocess.return_value = "node001\nnode002\nnode003"
        
        # Mock get_worker_list to return workers
        with patch("sparkjq.lib.get_worker_list") as mock_workers:
            mock_workers.return_value = ["node002", "node003"]
            
            # Mock spark context
            from sparkjq.spark import SparkContext
            mock_spark_ctx = SparkContext(home=temp_spark_home, version="3.5.0")
            
            # Mock async get_spark_context
            async def mock_get_spark():
                return mock_spark_ctx
            
            with patch("sparkjq.lib.get_spark_context", return_value=mock_get_spark()):
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_until_complete.return_value = mock_spark_ctx
                    
                    yield {
                        "spark_home": temp_spark_home,
                        "mock_workers": mock_workers,
                        "mock_subprocess": mock_subprocess,
                        "mock_popen": mock_popen,
                    }

    def test_basic_cluster_lifecycle(self, full_mock_environment):
        """Test basic cluster creation and teardown."""
        workers_file_path = None
        
        # Track workers file creation
        original_make_workers = sparkjq.lib.make_workers_file
        
        def track_workers_file(*args, **kwargs):
            nonlocal workers_file_path
            workers_file_path = original_make_workers(*args, **kwargs)
            return workers_file_path
        
        with patch("sparkjq.lib.make_workers_file", side_effect=track_workers_file):
            with SLURMCluster() as cluster:
                # Verify cluster is initialized
                assert cluster.slurm_context is not None
                assert cluster.spark_context is not None
                assert cluster.log_dir is not None
                
                # Verify workers file was created
                assert workers_file_path is not None
                assert os.path.exists(workers_file_path)
                
                # Verify content of workers file
                with open(workers_file_path, "r") as f:
                    content = f.read()
                    assert "node002" in content
                    assert "node003" in content
                
                # Verify start command was called
                assert full_mock_environment["mock_popen"].call_count == 1
                start_call = full_mock_environment["mock_popen"].call_args_list[0]
                assert "start-all.sh" in start_call[0][0][0]
        
        # After context exit
        # Verify stop command was called
        assert full_mock_environment["mock_popen"].call_count == 2
        stop_call = full_mock_environment["mock_popen"].call_args_list[1]
        assert "stop-all.sh" in stop_call[0][0][0]
        
        # Verify workers file was cleaned up
        assert not os.path.exists(workers_file_path)

    def test_cluster_with_custom_configuration(self, full_mock_environment):
        """Test cluster with custom port and directories."""
        custom_port = 9999
        custom_scratch = "/custom/scratch"
        custom_log = "/custom/logs"
        
        with patch("sparkjq.lib.build_log_dir") as mock_build_log:
            mock_build_log.return_value = f"{custom_log}/12345-log"
            
            with SLURMCluster(
                port=custom_port,
                scratch_dir=custom_scratch,
                log_dir=custom_log
            ) as cluster:
                # Verify custom configurations were used
                assert cluster.slurm_context.port == custom_port
                assert cluster.slurm_context.scratch == custom_scratch
                mock_build_log.assert_called_once_with(custom_log)

    def test_cluster_exception_handling(self, full_mock_environment):
        """Test that cleanup happens even when exception occurs."""
        workers_file_path = None
        
        # Track workers file
        original_make_workers = sparkjq.lib.make_workers_file
        
        def track_workers_file(*args, **kwargs):
            nonlocal workers_file_path
            workers_file_path = original_make_workers(*args, **kwargs)
            return workers_file_path
        
        with patch("sparkjq.lib.make_workers_file", side_effect=track_workers_file):
            try:
                with SLURMCluster() as cluster:
                    # Verify workers file exists
                    assert os.path.exists(workers_file_path)
                    
                    # Simulate an error in user code
                    raise ValueError("User error")
            except ValueError:
                pass
            
            # Verify cleanup still happened
            assert full_mock_environment["mock_popen"].call_count == 2
            assert not os.path.exists(workers_file_path)

    def test_environment_variables_set_correctly(self, full_mock_environment):
        """Test that all environment variables are set correctly."""
        original_env = dict(os.environ)
        
        try:
            with SLURMCluster():
                # Check Spark environment variables
                assert "SPARK_HOME" in os.environ
                assert "SPARK_LOG_DIR" in os.environ
                assert "PYSPARK_PYTHON" in os.environ
                assert os.environ["SPARK_HOME"] == full_mock_environment["spark_home"]
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_no_workers_scenario(self, full_mock_environment):
        """Test cluster with no worker nodes (single node)."""
        # Override mock to return empty worker list
        full_mock_environment["mock_workers"].return_value = []
        
        with SLURMCluster() as cluster:
            # Should still work with no workers
            assert cluster.workers_file is not None
            
            # Verify empty workers file
            with open(cluster.workers_file, "r") as f:
                content = f.read()
                assert content == ""

    def test_cluster_accessors(self, full_mock_environment):
        """Test cluster accessor methods."""
        with SLURMCluster() as cluster:
            # Test master() method
            master_host = cluster.master()
            assert master_host == "node001"
            
            # Test port() method
            port = cluster.port()
            assert port == 7077