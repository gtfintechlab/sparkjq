import os
import subprocess
import tempfile
from typing import List, NamedTuple, Optional


def determine_master_host() -> str:
    """Determine the master host for the Spark cluster."""
    # Check if SLURM is running
    assert (
        "SLURM_JOB_ID" in os.environ
    ), "SLURM is not running. Please run this script in a SLURM job."

    nodelist = subprocess.check_output(  # nosec B603,B607 - scontrol is a SLURM system command
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    # Get the first node in the list
    master_host = nodelist[0]

    return master_host


def get_worker_list() -> List[str]:
    """Get the list of worker nodes for the SLURM job."""
    assert (
        "SLURM_JOB_ID" in os.environ
    ), "SLURM is not running. Please run this script in a SLURM job."

    nodelist = subprocess.check_output(  # nosec B603,B607 - scontrol is a SLURM system command
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    # Get the list of worker nodes
    worker_list = nodelist[1:]

    return worker_list


class SlurmContext(NamedTuple):
    """Context for SLURM job."""

    nnodes: int
    rank: int
    ncpus: int
    world_size: int
    hostname: str
    port: int
    scratch: str

    def __str__(self):
        return f"SlurmContext(nnodes={self.nnodes}, rank={self.rank}, ncpus={self.ncpus}, world_size={self.world_size}, port={self.port}, scratch_path={self.scratch})"


def get_slurm_context(
    port: Optional[int] = None,
    scratch: Optional[str] = None,
) -> SlurmContext:
    """Get the SLURM context for the current job."""
    assert (
        "SLURM_JOB_ID" in os.environ
    ), "SLURM is not running. Please run this script in a SLURM job."

    nnodes = int(os.environ["SLURM_NNODES"])  # We want this to break if not set
    rank = int(
        os.environ["SLURM_PROCID"]
    )  # We want this to break if not set, since slurmw always populates this
    if "SLURM_CPUS_PER_TASK" in os.environ:
        ncpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        print("SLURM_CPUS_PER_TASK not set, using os scheduler affinity")
        ncpus = len(os.sched_getaffinity(0))

    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        print("SLURM_NTASKS not set, using number of nodes")
        world_size = int(os.environ["SLURM_NNODES"])

    if port is None:
        port = int(os.environ.get("SPARK_MASTER_PORT", 7077))

    if scratch is None:
        scratch = tempfile.mkdtemp(
            prefix="spark-jobqueue-",
            dir=os.environ.get("SLURM_SCRATCH", "/tmp"),  # nosec B108
        )
    else:
        # Expand the scratch directory
        scratch = os.path.expanduser(scratch)

    hostname = determine_master_host()

    return SlurmContext(
        nnodes=nnodes,
        rank=rank,
        ncpus=ncpus,
        world_size=world_size,
        hostname=hostname,
        port=port,
        scratch=scratch,
    )


def build_log_dir(
    path: Optional[str] = None,
):
    """Build the log directory for the SLURM job."""
    if path is None:
        path = os.path.realpath(".")
        # # Use local path
        # path = os.path.join(
        #     ".",
        #     os.environ.get("SLURM_JOB_ID", "slurm-jobqueue-logs"),
        # )
    else:
        path = os.path.expanduser(path)

    # Append the job ID to the path
    log_dir_name = f"{os.environ['SLURM_JOB_ID']}-log"

    path = os.path.join(path, log_dir_name)

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    return path
