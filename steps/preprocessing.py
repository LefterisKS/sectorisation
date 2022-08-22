# import pyspark libraries
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame

# import python libraries
from importlib import reload
from functools import reduce
import string


def group_categorical(df, cols_to_group, f_min, concat_other):

    if f_min < 1:
        f_min = df.count()*f_min

    fs = []
    for c in cols_to_group:
        fs.append(
            df
            .dropna(subset=[c])
            .groupBy(c)
            .count()
            .select(c, 'count')
            .coalesce(1)
            .withColumnRenamed(c, 'category')
            .withColumn('column', F.lit(c))
        )
    fs = reduce(DataFrame.unionByName, fs).toPandas()
    for c in cols_to_group:
        other_list = list(fs[(fs['column']==c)&(fs['count']<f_min)]['category'])
        if concat_other:
            other = '_'.join(other_list)
        else:
            other = 'other'
        df = df.withColumn(
            c, F.when(df[c].isin(other_list), other).otherwise(df[c]))

    return df


def get_dummies(df, cvs):

    original_cols = df.columns

    df = df.withColumn('row_id', F.monotonically_increasing_id())
    for cv in cvs:
        df = df.fillna('Null', subset=[cv])
        cats = df.select(cv).distinct().rdd.flatMap(lambda x: x).collect()
        exprs = []
        for cat in cats:
            expr = F.when(F.col(cv)==cat, 1).otherwise(0).alias(cv+'/'+cat)
            exprs.append(expr)
        dummies = df.select('row_id', *exprs)
        df = df.join(dummies, ['row_id'], 'left')
    df = df.drop('row_id')

    dummy_cols = [c for c in df.columns if c not in original_cols]

    return df, dummy_cols


def prepare_dynamic_features(df, dvs, cvs, nvs, tvs):

    df = df.select(dvs + cvs + nvs + tvs + ['Owner_ID'])

    # DATE COLUMNS

    #.withColumn('ISIN2', df['ISIN'].substr(1, 2))

    for c in dvs:
        df = (
            df
            #.withColumn('year', F.year(df[c]))
            .withColumn('month', F.month(df[c]))
            .withColumn('dayofmonth', F.dayofmonth(df[c]))
            .withColumn('dayofweek', F.dayofweek(df[c]))
            .withColumn('quarter', F.quarter(df[c]))
            .drop(c)
        )

    nvs = nvs + ['month', 'dayofmonth', 'dayofweek', 'quarter']

    # CATEGORICAL COLUMNS

    # regroup and get dummmies
    df = group_categorical(df, cvs, f_min=0.05, concat_other=False)
    df, dummy_cols = get_dummies(df, cvs)

    # get grouped version of dynamic part of df
    grouped = df.groupBy("Owner_ID")

    # initialise aggregated dfd with counts
    dfg = grouped.count().coalesce(1)

    for aggfun in ['mean', 'sum']:
        exprs = {x: aggfun for x in dummy_cols}
        dfg = (dfg.join(grouped.agg(exprs).coalesce(1), "Owner_ID"))

    # NUMERICAL COLUMNS

    for aggfun in ['mean', 'sum', 'min', 'max']: # stddev
        exprs = {x: aggfun for x in nvs}
        dfg = (dfg.join(grouped.agg(exprs).coalesce(1), "Owner_ID"))

    for c in dfg.columns:
        if c != "Owner_ID":
            dfg = dfg.withColumn(c, dfg[c].cast(T.DoubleType()))

    return dfg


def prepare_static_features(df, dvs, cvs, nvs, tvs, keywords):

    df = df.select(dvs + cvs + nvs + tvs + ['Owner_ID'])
    df = df.dropDuplicates(subset=['Owner_ID']).coalesce(1)

    # TEXT COLUMNS

    c = tvs[0]
    for w in keywords:
        df = df.withColumn(w, F.when(df[c].contains(w), 1.0).otherwise(0.0))
    df = df.withColumn('ltd', df['ltd'] + df['limited'])

    # CATEGORICAL COLUMNS

    # regroup
    df = group_categorical(df, cvs, f_min=0.05, concat_other=False)

    # NUMERICAL COLUMNS

    for c in nvs:
        df = df.withColumn(c, df[c].cast(T.DoubleType()))

    return df


def main(df, labels, config):

    # dynamic predictors of df
    dvs = config.columns['dynamic']['date']
    cvs = config.columns['dynamic']['categorical']
    nvs = config.columns['dynamic']['numerical']
    tvs = config.columns['dynamic']['text']
    dfd = prepare_dynamic_features(df, dvs, cvs, nvs, tvs)

    # static predictors of df
    dvs = config.columns['static']['date']
    cvs = config.columns['static']['categorical']
    nvs = config.columns['static']['numerical']
    tvs = config.columns['static']['text']
    keywords = config.columns['keywords']
    dfs = prepare_static_features(df, dvs, cvs, nvs, tvs, keywords)

    idCol = config.columns['idCol']
    targetCol = config.columns['targetCol']

    # labels
    labels = group_categorical(labels, [targetCol], f_min=50, concat_other=True)

    # join all predictors and target in one df
    df = dfs.join(dfd, [idCol], 'left')
    df = df.join(labels, [idCol], 'left')

    new_names = [''.join(e for e in c if e.isalnum() or e=='_')
                 for c in df.columns]
    df = df.toDF(*new_names)

    return df
