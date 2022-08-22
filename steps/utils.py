# import spark libraries
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, ArrayType, FloatType,
                               DoubleType, IntegerType, StringType)
from pyspark.ml.feature import (StringIndexer, VectorAssembler, IndexToString,
                                Imputer, OneHotEncoder, OneHotEncoderEstimator,
                                CountVectorizer, RegexTokenizer, Tokenizer,
                                StopWordsRemover, StandardScaler, PCA)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import (DecisionTreeClassifier,
                                       RandomForestClassifier)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.pipeline import PipelineModel

# import python libraries
import pandas as pd
import numpy as np
import subprocess
import string
ps = list(string.punctuation)
import scipy as sp
import sys
import matplotlib.pyplot as plt
from functools import reduce
from itertools import product
from importlib import reload

def nunique(df, column):
    return df.select(column).distinct().count()

def unique(df, column):
    return df.select(column).distinct()

def value_counts(df, column, top_n):
    df.groupBy(column).count().sort("count", ascending=False).show(top_n)

def unionAll(*dfs):
    # df = unionAll(d1, df2, df3)
    return reduce(DataFrame.unionAll, dfs)


def get_folder_filepaths(folderpath, extension):

    """
    Returns list of paths to files inside the folderpath that have
    the specified extension/format (e.g. csv, parquet, etc)
    """

    ls = subprocess.Popen(

        ["hadoop","fs","-ls", folderpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    filepaths = []

    for line in ls.stdout:
        f = line.decode("utf-8")
        path = f.split()[-1]

        if path.split('.')[-1] == extension:
            filepaths.append(path)

    return filepaths


def shape(df):
    print((df.count(), len(df.columns)))


def union_df_dir_csv_files(paths):

    n = 0

    for path in paths:
        df1 = import_csv(path)
        n += 1
        if n == 1:
            df = df1
        else:
            df = df.union(df1)
    return df


def union_dfs_different_cols(df1, df2):

    """
    Append (vertically) dataframes with different columns.
     Fill the columns that do not exist in the other dataframe with Null.
    """

    cols1_missing = [c for c in df2.columns if c not in df1.columns]
    cols2_missing = [c for c in df1.columns if c not in df2.columns]

    for column in cols1_missing:
        df1 = df1.withColumn(column, F.lit(None))

    for column in cols2_missing:
        df2 = df2.withColumn(column, F.lit(None))

    return df1.union(df2)


def jaccard_ratio(s1,s2):

    s1 = set(s1)
    s2 = set(s2)
    uni = len(s1.union(s2))
    inter = len(s1.intersection(s2))

    return (uni-inter)/uni

jaccard_udf = F.udf(jaccard_ratio)


def missing_report(df):

    n = df.count()

    for c in nvs:
        print(c, df.where(df[c].isNull()).count()/n*100)


def metadata_1(df):

    dts = dict(df.dtypes)

    for c in df.columns:
        print(c)
        print(len(c)*'-')
        print('dtype =', dts[c])
        print('n_vals =', nunique(df, c))
        value_counts(df, c, top_n=10)


def metadata_2(df, top_n, normalize):

    n = df.count()

    dts = dict(df.dtypes)

    for c in df.columns:

        fs = df.groupBy(c).count().sort("count", ascending=False)
        fs = fs.withColumn('% total', fs['count']/n*100)

        print(c)
        print(len(c)*'-')
        print('dtype =', dts[c])
        print('n_vals =', fs.count())
        fs.filter(df[c].isNull()).show()

        fs = fs.toPandas()[['count', '% total']].head(top_n)

        if normalize:
            fs['% total'].plot(kind='barh')
        else:
            fs['count'].plot(kind='barh')

        plt.show()
        plt.clf


def hdfs_write(df, out_format, folderpath, write_mode):

    """
    Writes spark dataframe to hdfs folderpath

    Args:
        df         :
        out_format : csv, parquet, etc
        folderpath :
        write_mode : overwrite, append

    """

    (
        df
        .coalesce(1)
        .write
        .save(folderpath, header=True, format=out_format, mode=write_mode)
    )


def frequency_plot(df, column, top_n, normalize):

    fs = (

        df
        .groupBy(column)
        .count()
        .sort("count", ascending=False)
        .toPandas()['count'][:top_n]
    )

    if normalize:
        n = fs.sum()
    else:
        n = 1

    (fs/n).plot(kind='barh')


def create_target_weight_col(df, target):
    ws = df.groupBy(target).count()
    ws = ws.withColumn('1/count', 1/ws['count'])
    ws = ws.crossJoin(ws.select('1/count').groupBy().sum())
    ws = ws.withColumn('count*sum(1/count)', ws['count']*ws['sum(1/count)'])
    ws = ws.withColumn('weight', 1/ws['count*sum(1/count)'])
    return df.join(ws.select([target, 'weight']) , [target], how='left')


def read_txt(filepath, delimiter=','):

    sc = spark.sparkContext

    df = sc.textFile(filepath)

    header = df.first()
    df = df.filter(lambda line: line != header)

    df = df.map(lambda k: k.split(delimiter))
    df = df.toDF(header.split(delimiter))

    return df


def split_unique_duplicates(df, column):

    fs = df.groupBy(column).count()
    df = df.join(fs, [column], how='left')

    df_dups = df.filter(df['count']>1)
    df_unique = df.filter(df['count']==1).drop('count')

    return df_unique, df_dups


def left_join(df1, df2, id1, id2, mode):

    df1 = df1.withColumnRenamed(id1, 'id1')
    df2 = df2.withColumnRenamed(id2, 'id2')

    df1 = df1.join(df2.select('id2'), df1.id1==df2.id2, 'left')
    if mode == 'inner':
        df1 = df1.where(F.col('id2').isNotNull())
    elif mode == 'outer':
        df1 = df1.where(F.col('id2').isNull())

    return df1.drop('id2').withColumnRenamed('id1', id1)


def extract_substring(df, input_col, output_col, values):
    cols = [F.when(df[input_col].contains(k), k) for k in values]
    return df.select("*", F.coalesce(*cols).alias(output_col))


def get_df_bytes_in_driver(df, fraction):
    s = (
        df
        .sample(False, fraction=fraction, seed=0)
        .toPandas()
        .memory_usage(deep=True)
        .sum()/fraction
    )
    print("df size in driver: {0:,.0f} bytes".format(s))
    return s


def get_df_bytes_in_driver_2(df):

    """
    # df.toPandas().info(memory_usage='deep')
    """

    n_total = df.count()
    n_sample = 100

    df_sample = df.limit(n_sample).toPandas()
    sample_size = df_sample.memory_usage(deep=True).sum()
    total_size = sample_size * n_total / n_sample
    print("Total size: {0:,.0f} bytes".format(total_size))

    return total_size


def cat_to_dummies(df, cvs):
    df = df.copy()
    for col in cvs:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df.drop([col],axis=1,inplace=True)
        df = pd.concat([df,dummies],axis=1)
    return df


def group_cat_var_other(df, col, f_min):
    df = df.copy()
    fs = df[col].value_counts()
    if f_min<1:
        f_min = len(df)*f_min
    low_fs = fs[fs<f_min]

    if len(low_fs) > 1:
        g_vals = list(low_fs.index)
        # g_val = '_'.join(g_vals)
        df[col] = df[col].apply(lambda x: 'other' if x in g_vals else x)
    return df
