# import pyspark libraries
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# import python libraries
from time import time
from importlib import reload
import os

# import modules
from sectorisation.steps import (
    staging,
    split_owners,
    preprocessing,
    predict_label,
    recommend_label
)
reload(staging)
reload(split_owners)
reload(preprocessing)
reload(predict_label)
reload(recommend_label)

# import user configuration
from sectorisation import config
reload(config)

def labelling(spark, config):

#    # EXTRACT
#    path = os.path.join(config.staged_dir, 'idbr')
#    idbr = spark.read.parquet(path)
#    path = os.path.join(config.staged_dir, 'tr')
#    tr = spark.read.parquet(path)
#    # TRANSFORM
#    idbr, ids_dups, bp, tr, labels = split_owners.main(idbr, tr, config)
#    # LOAD
#    path = '/user/karace/sectorisation/test/idbr'
#    idbr.write.mode("overwrite").parquet(path)
#    path = '/user/karace/sectorisation/test/ids_dups'
#    ids_dups.write.mode("overwrite").parquet(path)
#    path = '/user/karace/sectorisation/test/bp'
#    bp.write.mode("overwrite").parquet(path)
#    path = '/user/karace/sectorisation/test/tr'
#    tr.write.mode("overwrite").parquet(path)
#    path = '/user/karace/sectorisation/test/labels'
#    labels.write.mode("overwrite").parquet(path)

#    # labelled_matching

#    # EXTRACT
#    path = '/user/karace/sectorisation/test/tr'
#    tr = spark.read.parquet(path)
#    path = '/user/karace/sectorisation/test/labels'
#    labels = spark.read.parquet(path)
#    # TRANSFORM
#    df = preprocessing.main(tr, labels, config)
#    # LOAD
#    path = '/user/karace/sectorisation/test/df'
#    df.write.mode("overwrite").parquet(path)

#    # EXTRACT
#    path = '/user/karace/sectorisation/test/df'
#    df = spark.read.parquet(path)
#    # TRANSFORM
#    ml_labels, manual_labels = predict_label.main(spark, df, config)
#    # LOAD
#    path = '/user/karace/sectorisation/test/ml_labels'
#    ml_labels.write.mode("overwrite").parquet(path)
#    path = '/user/karace/sectorisation/test/manual_labels'
#    manual_labels.write.mode("overwrite").parquet(path)

    # EXTRACT
    path = os.path.join(config.staged_dir, 'tr')
    tr = spark.read.parquet(path)
    path = '/user/karace/sectorisation/test/idbr'
    idbr = spark.read.parquet(path)
    path = '/user/karace/sectorisation/test/manual_labels'
    manual_labels = spark.read.parquet(path)
    # TRANSFORM
    mlabels_rm, mlabels_no_rm = recommend_label.main(
        tr, manual_labels, idbr, config)
    # LOAD
    path = '/user/karace/sectorisation/test/mlabels_rm'
    mlabels_rm.write.mode("overwrite").parquet(path)
    path = '/user/karace/sectorisation/test/mlabels_no_rm'
    mlabels_no_rm.write.mode("overwrite").parquet(path)

    # predict_label conflict between pysparkml and sklearn names
    # samples from fuzzy/ml for misclassification error)
    # broadcast sklearn to cluster
    # analysis

    # comments/docstrings
    # run system end-to-end

    return mlabels_rm, mlabels_no_rm

if __name__ == "__main__":

    # start spark session
    spark = (
        SparkSession.builder.appName('sectorisation')
        .config('spark.executor.memory', '10g')
        .config('spark.yarn.executor.memoryOverhead', '1g')
        .config('spark.executor.cores', 6)
        .config('spark.dynamicAllocation.maxExecutors', 5)
        .config('spark.dynamicAllocation.enabled', 'false')
        .config('spark.shuffle.service.enabled', 'true')
        .config('spark.driver.maxResultSize', '5g')
        .config('spark.sql.execution.arrow.enabled', 'false')
        .enableHiveSupport()
        .getOrCreate()
    )
    start = time()
    # idbr, tr = staging.main()
    mlabels_rm, mlabels_no_rm = labelling(spark, config)
    print('\nrun_time:', round((time()-start)/60, 2), 'minutes')
