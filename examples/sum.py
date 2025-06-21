from pyspark.sql import SparkSession
from sparkjq import SLURMCluster

if __name__ == "__main__":
    # Create a SLURM cluster

    # From here onwards you can use normal pyspark code
    # Only the master node will proceed here, workers will not
    with SLURMCluster() as cluster:
        # The master node will run this code
        print(f"Master node: {cluster.slurm_context.hostname}")
        print(f"Worker nodes: {cluster.slurm_context.world_size - 1}")

        # Create a Spark session
        spark = SparkSession.builder.appName("Sum Example").getOrCreate()

        # Create a DataFrame with some data
        data = [(1,), (2,), (3,), (4,), (5,)]
        df = spark.createDataFrame(data, ["number"])

        # Calculate the sum of the numbers
        total_sum = df.agg({"number": "sum"}).collect()[0][0]
        print(f"The sum is: {total_sum}")

        # Stop the Spark session
        spark.stop()
