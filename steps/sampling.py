# import pyspark libraries
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    StandardScaler, VectorAssembler, Imputer, PCA, StringIndexer)
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.classification import DecisionTreeClassifier

# import python libraries
import pandas as pd

def stratified_sample(df, column, p, seed):

    if p > 1:
        n = df.count()
        p = p/df.count()

    fractions = (
        df
        .select(column)
        .distinct()
        .withColumn('fraction', F.lit(p))
        .rdd
        .collectAsMap()
    )
    sampled_df = df.stat.sampleBy(column, fractions, seed)
    return sampled_df


def stratified_train_test_split(df, target, p_test, seed):

    df = df.withColumn('id', F.monotonically_increasing_id())

    # create test data
    test = stratified_sample(df, target, p_test, seed)
    test = test.withColumnRenamed('id', 'id1')

    train = df.join(test.select('id1'), df.id == test.id1, 'left')
    train = train.where(train['id1'].isNull())
    train = train.drop(*['id', 'id1'])

    test = test.drop('id1')

    return train, test


def pca_transform(df, n_components_max, sum_evs_min):

    pca = PCA(k=n_components_max, inputCol="std_features", outputCol="pca")
    evs = pca.fit(df).explainedVariance

    sum_evs = 0
    for k, ev in enumerate(evs):
        sum_evs += ev
        if sum_evs > sum_evs_min:
            n_components = k+1
            print(n_components, sum_evs)
            break

    if sum_evs > sum_evs_min:
        pca = PCA(k=n_components, inputCol="std_features", outputCol="pca")
        df = pca.fit(df).transform(df)
    else:
        print(f'{n_components_max} not enough to reach {sum_evs_min}')
        print('original dataframe returned')

    return df


def stratified_kmeans_sample(
    df,
    features,
    apply_pca,
    n_components_max,
    sum_evs_min,
    n_clusters_max,
    cost_reduction,
    n_sample
):

    # transformers stages of pipeline
    imputer = Imputer(inputCols=features, outputCols=features).setStrategy("median")
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    standardizer = StandardScaler(inputCol='features', outputCol='std_features')

    # pipeline and transform data (preprocessing for kmeans)
    stages = [imputer, assembler, standardizer]
    pipeline = Pipeline(stages=stages)
    df = pipeline.fit(df).transform(df)

    if apply_pca:
        df = pca_transform(df, n_components_max, sum_evs_min)
        FeaturesCol = "pca"
    else:
        FeaturesCol = "std_features"

#    # find best k, i.e. optimal number of clusters to minimise
#    costs = []
#    best_k_found = False
#    for k in range(2, n_clusters_max):
#        print('number of clusters:', k)
#        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol(FeaturesCol)
#        model = kmeans.fit(df) # df.sample(False, 0.3, seed=42)
#        # WSSSE = Within Set Sum of Squared Errors
#        cost = model.computeCost(df)
#        costs.append([k, cost])
#        if k == 2:
#            cost_max = cost
#        if cost < cost_max*(1-cost_reduction) and not best_k_found:
#            best_k = k
#            best_k_found = True
#            print('best_k:', best_k)
#
#    # if n_clusters_max is not enough to reach cost2_frac
#    if not best_k_found:
#        best_k = 5

    # find best k, i.e. optimal number of clusters to minimise cost
    costs = []
    for k in range(2, n_clusters_max):
        print('number of clusters:', k)
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol(FeaturesCol)
        model = kmeans.fit(df) # df.sample(False, 0.3, seed=42)
        # WSSSE = Within Set Sum of Squared Errors
        cost = model.computeCost(df)
        costs.append([k, cost])

    cost_max = costs[0][1]
    cost_min = costs[-1][1]
    cost_crit = cost_max - (cost_max - cost_min)*cost_reduction

    for c in costs:
        if c[1] < cost_crit:
            best_k = c[0]
            print('best_k:', best_k)
            break

    (
        pd
        .DataFrame(costs, columns = ['k', 'cost'])
        .plot(x='k', y='cost', legend=False)
    )

    # predict cluster for each record (label)
    kmeans = KMeans().setK(best_k).setSeed(1).setFeaturesCol(FeaturesCol)
    df = kmeans.fit(df).transform(df)

    # get stratified sample based on clusters and drop cols
    dfs = stratified_sample(df, column='prediction', p=n_sample, seed=42)

    return dfs


def stratified_dtree_sample(
    df,
    features,
    targetCol,
    minInstancesPerNode,
    n_sample
):

    label_indexer = StringIndexer(inputCol=targetCol, outputCol="indexedLabel")
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    dt = DecisionTreeClassifier(
        labelCol="indexedLabel", minInstancesPerNode=minInstancesPerNode)
    pipeline = Pipeline(stages=[label_indexer, assembler, dt])
    df = pipeline.fit(df).transform(df)

    # get stratified sample based on clusters and drop cols
    dfs = stratified_sample(df, column='prediction', p=n_sample, seed=42)

    return dfs
