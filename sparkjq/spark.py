import os
import socket
import sys
import tarfile
from typing import NamedTuple

import pyspark
import requests
import tqdm

SPARKJQ_HOME = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "spark-jobqueue",
)


class SparkContext(NamedTuple):
    """Context for Spark job."""

    home: str
    version: str


async def get_spark_context() -> SparkContext:
    """Get the aligned Spark versions from the Apache Spark website."""
    version = pyspark.__version__

    spark_home = os.path.join(SPARKJQ_HOME, f"spark-{version}-bin-hadoop3")

    if os.path.exists(spark_home):
        print(f"Spark home already exists: {spark_home}")
        return SparkContext(home=spark_home, version=version)

    os.makedirs(SPARKJQ_HOME, exist_ok=True)
    # Get the latest Spark version from the Apache Spark website
    url = f"https://dlcdn.apache.org/spark/spark-{version}/spark-{version}-bin-hadoop3.tgz"
    temp_destination = os.path.join(SPARKJQ_HOME, f"spark-{version}-bin-hadoop3.tgz")

    print(f"Downloading Spark from {url}")
    # Spark is large, so we use a streaming download and show a progress bar
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with open(temp_destination, "wb") as file, tqdm.tqdm(
            desc=temp_destination,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

    print(f"Unpacking Spark to {spark_home}")
    os.makedirs(os.path.dirname(spark_home), exist_ok=True)
    # Unpack the tarball with a progress bar
    with tarfile.open(temp_destination, "r:gz") as tar:
        total_size = len(tar.getmembers())
        with tqdm.tqdm(total=total_size, desc="Unpacking Spark") as bar:
            for member in tar.getmembers():
                tar.extract(member, SPARKJQ_HOME)
                bar.update(1)

    # Remove the tarball after unpacking
    os.remove(temp_destination)
    print(f"Done. Spark is installed at {spark_home}")
    return SparkContext(home=spark_home, version=version)


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


def setup_spark_environment_variables(
    log_dir: str,
    spark_home: str,
    python_executable: str = None,
    scratch_dir: str = None,
) -> None:
    """Setup the environment variables for Spark."""
    os.environ["SPARK_HOME"] = spark_home
    os.environ["SPARK_LOG_DIR"] = log_dir
    if python_executable is not None:
        os.environ["PYSPARK_PYTHON"] = python_executable
    else:
        # Use current python executable
        if "PYSPARK_PYTHON" not in os.environ:
            os.environ["PYSPARK_PYTHON"] = sys.executable

    if scratch_dir is not None:
        os.environ["SPARK_LOCAL_DIRS"] = scratch_dir


def make_workers_file(
    spark_home: str,
    hostnames: list[str],
) -> str:
    dir_to_store = os.path.join(spark_home, "conf")
    os.makedirs(dir_to_store, exist_ok=True)
    with open(os.path.join(dir_to_store, "workers"), "w") as f:
        for hostname in hostnames:
            f.write(f"{hostname}\n")

    return os.path.join(dir_to_store, "workers")
