# sparkjq
A wrapper to start spark clusters on slurm inspired by dask_jobqueue

See examples.py (more docs soon)

```python
# Example: Start a Spark cluster on Slurm
from sparkjq import SlurmCluster

cluster = SlurmCluster()

# Normal pyspark code here [...]

cluster.shutdown()
```

## Installation

```
pip install git+https://github.com/roloza7/sparkjq
```
