path = '/user/karace/sectorisation/test/manual_labels'
df = spark.read.parquet(path)

targetCol = 'target_max_prob'
idCol = 'Owner_ID'

# numerical variables as features for clustering
cols = [idCol, 'max_prob', targetCol]
features = [c for c in df.columns if c not in cols]

n_sample = 100

minInstancesPerNode = 1000


dfs = stratified_dtree_sample(
    df,
    features,
    targetCol,
    minInstancesPerNode,
    n_sample
)

dfs = dfs.drop(
    'indexedLabel',
    'features',
    'rawPrediction',
    'probability',
    'prediction'
)

################

path = '/user/karace/sectorisation/test/manual_labels'
df = spark.read.parquet(path)

 # numerical variables as features for clustering
cols = [idCol, 'target_max_prob', 'max_prob']
features = [c for c in df.columns if c not in cols]

apply_pca = True
n_components_max = 100
sum_evs_min = 0.9

cost_reduction = 0.8
n_clusters_max = 100

n_sample = 100

dfs = stratified_kmeans_sample(
    df,
    features,
    apply_pca,
    n_components_max,
    sum_evs_min,
    n_clusters_max,
    cost_reduction,
    n_sample
)

dfs = dfs.drop('features', 'std_features', 'pca', 'prediction')
