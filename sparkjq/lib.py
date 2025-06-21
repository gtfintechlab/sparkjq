import asyncio
import os
import subprocess
import sys

from .slurm import SlurmContext, build_log_dir, get_slurm_context, get_worker_list
from .spark import (
    SparkContext,
    get_spark_context,
    make_workers_file,
    setup_spark_environment_variables,
)


class SLURMCluster:
    """Represents a slurm cluster in pyspark.
    Takes care of spinning up the cluster and shutting it down.
    """

    slurm_context: SlurmContext
    spark_context: SparkContext

    def __init__(self, port: int = None, scratch_dir: str = None, log_dir: str = None):
        self.slurm_context = get_slurm_context(port=port, scratch=scratch_dir)

        self.spark_context = asyncio.get_event_loop().run_until_complete(get_spark_context())
        self.log_dir = build_log_dir(log_dir)

        self.handle = None
        self.workers_file = None
        # From here on only the master node is working because the workers blocked

    def __enter__(self):
        """Utilization as context manager."""
        setup_spark_environment_variables(
            log_dir=self.log_dir,
            spark_home=self.spark_context.home,
            python_executable=os.environ.get("PYSPARK_PYTHON", sys.executable),
            scratch_dir=self.slurm_context.scratch,
        )

        # Create conf/workers in spark home
        self.workers_file = make_workers_file(
            spark_home=self.spark_context.home,
            hostnames=get_worker_list(),
        )

        executable_path = os.path.join(self.spark_context.home, "sbin", "start-all.sh")

        self.handle = subprocess.Popen(
            [executable_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the cluster to start
        self.handle.wait()

        # All nodes should be up and running now

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Utilization as context manager."""
        executable_path = os.path.join(self.spark_context.home, "sbin", "stop-all.sh")
        self.handle = subprocess.Popen(
            [executable_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for the cluster to stop
        self.handle.wait()

        # Clean up the workers file
        if self.workers_file:
            os.remove(self.workers_file)

        if exc_type:
            raise exc_type(exc_value)

        return False

    def master(self):
        """Get the master node."""
        return self.slurm_context.hostname

    def port(self):
        """Get the port of the master node."""
        return self.slurm_context.port
