"""Tests for the SLURM module."""

import os
import tempfile
from unittest.mock import patch

import pytest

from sparkjq.slurm import (
    SlurmContext,
    build_log_dir,
    determine_master_host,
    get_slurm_context,
    get_worker_list,
)


class TestDetermineMasterHost:
    """Tests for determine_master_host function."""

    def test_determine_master_host_multi_node(self, mock_slurm_env, mock_subprocess):
        """Test master host determination with multiple nodes."""
        mock_subprocess.return_value = "node001\nnode002\nnode003"
        result = determine_master_host()
        assert result == "node001"
        mock_subprocess.assert_called_once_with(
            ["scontrol", "show", "hostnames", mock_slurm_env["SLURM_JOB_NODELIST"]], text=True
        )

    def test_determine_master_host_single_node(
        self, mock_single_node_env, mock_subprocess_single_node
    ):
        """Test master host determination with single node."""
        result = determine_master_host()
        assert result == "node001"

    def test_determine_master_host_no_slurm_env(self, monkeypatch):
        """Test that function raises assertion when SLURM_JOB_ID is missing."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(AssertionError, match="SLURM is not running"):
            determine_master_host()


class TestGetWorkerList:
    """Tests for get_worker_list function."""

    def test_get_worker_list_multi_node(self, mock_slurm_env, mock_subprocess):
        """Test getting worker list with multiple nodes."""
        mock_subprocess.return_value = "node001\nnode002\nnode003"
        result = get_worker_list()
        assert result == ["node002", "node003"]

    def test_get_worker_list_single_node(
        self, mock_single_node_env, mock_subprocess_single_node
    ):
        """Test getting worker list with single node (no workers)."""
        result = get_worker_list()
        assert result == []

    def test_get_worker_list_no_slurm_env(self, monkeypatch):
        """Test that function raises assertion when SLURM_JOB_ID is missing."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(AssertionError, match="SLURM is not running"):
            get_worker_list()


class TestSlurmContext:
    """Tests for SlurmContext NamedTuple."""

    def test_slurm_context_creation(self):
        """Test creating a SlurmContext object."""
        ctx = SlurmContext(
            nnodes=3,
            rank=0,
            ncpus=4,
            world_size=3,
            hostname="node001",
            port=7077,
            scratch="/tmp/spark-jobqueue-123",
        )
        assert ctx.nnodes == 3
        assert ctx.rank == 0
        assert ctx.ncpus == 4
        assert ctx.world_size == 3
        assert ctx.hostname == "node001"
        assert ctx.port == 7077
        assert ctx.scratch == "/tmp/spark-jobqueue-123"

    def test_slurm_context_str(self):
        """Test string representation of SlurmContext."""
        ctx = SlurmContext(
            nnodes=2,
            rank=1,
            ncpus=8,
            world_size=2,
            hostname="node002",
            port=7077,
            scratch="/scratch/temp",
        )
        result = str(ctx)
        assert "nnodes=2" in result
        assert "rank=1" in result
        assert "ncpus=8" in result
        assert "world_size=2" in result
        assert "port=7077" in result
        assert "scratch_path=/scratch/temp" in result


class TestGetSlurmContext:
    """Tests for get_slurm_context function."""

    def test_get_slurm_context_default(self, mock_slurm_env, mock_subprocess):
        """Test getting SLURM context with default parameters."""
        mock_subprocess.return_value = "node001\nnode002\nnode003"
        
        ctx = get_slurm_context()
        
        assert ctx.nnodes == 3
        assert ctx.rank == 0
        assert ctx.ncpus == 4
        assert ctx.world_size == 3
        assert ctx.hostname == "node001"
        assert ctx.port == 7077
        assert ctx.scratch.startswith("/tmp/spark-jobqueue-")

    def test_get_slurm_context_custom_port_scratch(
        self, mock_slurm_env, mock_subprocess
    ):
        """Test getting SLURM context with custom port and scratch."""
        mock_subprocess.return_value = "node001\nnode002\nnode003"
        
        ctx = get_slurm_context(port=8080, scratch="~/myscratch")
        
        assert ctx.port == 8080
        assert ctx.scratch == os.path.expanduser("~/myscratch")

    def test_get_slurm_context_no_cpus_per_task(
        self, mock_slurm_env, mock_subprocess, monkeypatch
    ):
        """Test fallback to os.sched_getaffinity when SLURM_CPUS_PER_TASK not set."""
        mock_subprocess.return_value = "node001"
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        
        with patch("os.sched_getaffinity") as mock_affinity:
            mock_affinity.return_value = {0, 1, 2, 3, 4, 5, 6, 7}
            ctx = get_slurm_context()
            assert ctx.ncpus == 8

    def test_get_slurm_context_no_ntasks(
        self, mock_slurm_env, mock_subprocess, monkeypatch
    ):
        """Test fallback to SLURM_NNODES when SLURM_NTASKS not set."""
        mock_subprocess.return_value = "node001"
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        
        ctx = get_slurm_context()
        assert ctx.world_size == 3  # Should use SLURM_NNODES value

    def test_get_slurm_context_custom_spark_port(
        self, mock_slurm_env, mock_subprocess, monkeypatch
    ):
        """Test using SPARK_MASTER_PORT environment variable."""
        mock_subprocess.return_value = "node001"
        monkeypatch.setenv("SPARK_MASTER_PORT", "9999")
        
        ctx = get_slurm_context()
        assert ctx.port == 9999

    def test_get_slurm_context_no_slurm_env(self, monkeypatch):
        """Test that function raises assertion when SLURM_JOB_ID is missing."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        with pytest.raises(AssertionError, match="SLURM is not running"):
            get_slurm_context()


class TestBuildLogDir:
    """Tests for build_log_dir function."""

    def test_build_log_dir_default(self, mock_slurm_env):
        """Test building log directory with default path."""
        with patch("os.path.realpath") as mock_realpath:
            mock_realpath.return_value = "/current/dir"
            with patch("os.makedirs") as mock_makedirs:
                result = build_log_dir()
                
                assert result == "/current/dir/12345-log"
                mock_makedirs.assert_called_once_with(
                    "/current/dir/12345-log", exist_ok=True
                )

    def test_build_log_dir_custom_path(self, mock_slurm_env):
        """Test building log directory with custom path."""
        with patch("os.makedirs") as mock_makedirs:
            result = build_log_dir("/custom/logs")
            
            assert result == "/custom/logs/12345-log"
            mock_makedirs.assert_called_once_with(
                "/custom/logs/12345-log", exist_ok=True
            )

    def test_build_log_dir_expanduser(self, mock_slurm_env):
        """Test that ~ is expanded in custom path."""
        with patch("os.path.expanduser") as mock_expanduser:
            mock_expanduser.return_value = "/home/user/logs"
            with patch("os.makedirs") as mock_makedirs:
                result = build_log_dir("~/logs")
                
                mock_expanduser.assert_called_once_with("~/logs")
                assert result == "/home/user/logs/12345-log"

    def test_build_log_dir_exists(self, mock_slurm_env):
        """Test that existing directory doesn't cause error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the log directory first
            log_dir = os.path.join(tmpdir, "12345-log")
            os.makedirs(log_dir)
            
            # Should not raise an error
            result = build_log_dir(tmpdir)
            assert result == log_dir
            assert os.path.exists(log_dir)