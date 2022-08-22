# import pyspark libraries
from pyspark.sql import DataFrame
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    IndexToString,
    Imputer,
    OneHotEncoder,
    OneHotEncoderEstimator
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.pipeline import PipelineModel

# import python libraries
import pandas as pd
import numpy as np
import joblib
from functools import reduce
from importlib import reload
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# import custom libraries
from sectorisation.steps import utils
reload(utils)


def using_spark(df, p_test, idCol, targetCol, weightCol, min_max_prob, model_dir):

    """
    1. split labelled/unlabelled
    2. create weight column (optional)
    3. separate nvs, cvs
    4. create all stages and pipeline
    5. cache df to train
    6a. create and store model by using cross validation for tuning:
        * create cross validator
        * cache labelled
        * fit to all labelled
        * print performance
        * store best model
    6b. create and store model by  default params without tuning:
        * split labelled to train and test
        * cache labelled, train and test
        * fit to train
        * predict on test
        * print performance
        * fit to all labelled
        * store model
    7. load model
    8. predict on unlabelled
    9. reverse transform label
    10. get max probability
    11. split predictions based on whether label probability is above threshold
    """

    # split data in labelled and unlabelled
    labelled = df.filter(df[targetCol].isNotNull())
    unlabelled = df.filter(df[targetCol].isNull()).drop(targetCol)

    # create weight column for unbalanced data
    if weightCol:
        labelled = create_target_weight_col(df=labelled, target=targetCol)

    # split numerical and categorical variables
    dts = [c for c in df.dtypes if c[0] not in [targetCol, weightCol, idCol]]
    cvs = [c[0] for c in dts if c[1]=='string']
    nvs = [c[0] for c in dts if c[1]!='string']

    # prepare stages and pipeline
    target_indexer = StringIndexer(inputCol=targetCol, outputCol='label')
    nvs_imputer = Imputer(inputCols=nvs, outputCols=nvs).setStrategy("median")
    indexers = []
    for c in cvs:
        indexers.append(
            StringIndexer(inputCol=c, outputCol=c+"_indexed", handleInvalid="keep")
        )
    encoder = OneHotEncoderEstimator(
        inputCols=[idx.getOutputCol() for idx in indexers],
        outputCols=["{0}_encoded".format(idx.getOutputCol()) for idx in indexers]
    )
    features = nvs + encoder.getOutputCols()
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    rf = RandomForestClassifier() # lr = LogisticRegression() # weightCol='weight'

    stages = [target_indexer, nvs_imputer] + indexers + [encoder, assembler, rf]
    pipeline = Pipeline(stages=stages)

    if p_test==0:
        grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, [3, 5, 8])
            .addGrid(rf.minInstancesPerNode, [10, 20, 40])
            .addGrid(rf.numTrees, [10, 20, 30])
            .build()
        )
        #grid = (
        #    ParamGridBuilder()
        #    .addGrid(lr.elasticNetParam, [0.5])
        #    .addGrid(lr.aggregationDepth, [2])
        #    .addGrid(lr.maxIter, [5])
        #    .addGrid(lr.regParam, [0.5])
        #    .build()
        #)
        evaluator = (
            MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setMetricName("accuracy")
        )
        cv = (
            CrossValidator()
            .setNumFolds(3)
            .setEstimator(pipeline)
            .setEstimatorParamMaps(grid)
            .setEvaluator(evaluator)
            .setSeed(0)
        )
        #cv = CrossValidator(
        #    estimator=pipeline,
        #    estimatorParamMaps=paramGrid,
        #    evaluator=MulticlassClassificationEvaluator(),
        #    numFolds=3, # use 3+ folds in practice
        #)

        # repartition and cache for faster training (assume small train)
        labelled.repartition(1).cache().count()

        # training
        model = cv.fit(labelled)

        # performance
        scores = model.avgMetrics
        best_score = max(scores)
        #best_params = model.getEstimatorParamMaps()[scores.index(best_score)]
        print('best score:', best_score)

        # store model
        model.bestModel.write().overwrite().save(model_dir)

    else:
        # split labelled in train and test
        train, test = stratified_train_test_split(
            df=labelled, target=targetCol, p_test=p_test, seed=0)

        # repartition and cache for faster training (assume small train)
        labelled.repartition(1).cache().count()
        train.repartition(1).cache().count()
        test.repartition(1).cache().count()

        # train to predict on test for performance
        model = pipeline.fit(train)

        # predict test to get performance
        prediction = model.transform(test)

        # accuracy
        acc_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
        acc = acc_eval.evaluate(prediction)
        print(f"Accuracy: {acc}")

        # f1 score
        f1_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1")
        f1 = f1_eval.evaluate(prediction)
        print(f"F1 Score: {f1}")

        # train on all labelled
        model = pipeline.fit(labelled)

        # store model
        model.write().overwrite().save(model_dir)

    # load stored model
    model = PipelineModel.load(model_dir)

    # predict label for unlabelled
    prediction = model.transform(unlabelled)

    # reverse label from indexed version to original values
    labelReverse = IndexToString(
        inputCol="prediction",
        outputCol="label",
        labels=target_indexer.fit(labelled).labels
    )
    prediction = labelReverse.transform(prediction)

    # get max probability that gave each label
    udf_max_prob = F.udf(lambda v: float(max(v)), T.FloatType())
    prediction = prediction.withColumn('max_prob', udf_max_prob("probability"))

    ml_labels = prediction.filter(prediction['max_prob']>=min_max_prob)
    manual_labels = prediction.filter(prediction['max_prob']<min_max_prob)

    return ml_labels, manual_labels


def using_python(df, targetCol, idCol, cvs, min_max_prob):

    for c in cvs:
        df = group_cat_var_other(df, col=c, f_min=0.05)
    df = cat_to_dummies(df, cvs)

    predictors = [c for c in df.columns if c not in [idCol, targetCol]]
    df[predictors] = df[predictors].apply(pd.to_numeric, downcast='float')

    df[predictors] = (
        Imputer(strategy='median')
        .fit(df[predictors])
        .transform(df[predictors])
    )

    labelled = df[df[targetCol].notnull()]
    unlabelled = df[df[targetCol].isnull()].reset_index(drop=True).drop(targetCol, 1)

    le = LabelEncoder()
    le.fit(labelled[targetCol])

    X_train = labelled.drop([idCol, targetCol], axis=1)
    y_train = le.transform(labelled[targetCol])

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

    rf = RandomForestClassifier(class_weight='balanced')

    param_grid = {
        'n_estimators'      : [30, 40, 50],
        'max_depth'         : [5, 10, 15, 20],
        'min_samples_split' : [2],
        'min_samples_leaf'  : [2, 5, 10, 15],
        'max_features'      : ['sqrt']           # ['sqrt', 'log2', None]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=False, random_state=123)
    model = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv=cv,
        scoring= 'accuracy',
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    print(f'Best score found: {model.best_score_}')
    print(f"Best parameters found:\n{model.best_params_}")
    #print("Grid scores on development set:")
    #print()
    #means = model.cv_results_['mean_test_score']
    #stds = model.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, model.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_pred = model.predict_proba(unlabelled[predictors])
    y_pred = pd.DataFrame(y_pred)
    esa_sectors = y_pred.columns
    y_pred[idCol] = unlabelled[idCol]
    y_pred['target_max_prob'] = y_pred[esa_sectors].apply(lambda x: pd.Series(x).idxmax(), 1)
    y_pred['target_max_prob'] = le.inverse_transform(y_pred['target_max_prob'])
    y_pred['max_prob'] = y_pred[esa_sectors].apply(lambda x: pd.Series(x).max(), 1)
    y_pred = y_pred[[idCol, 'target_max_prob', 'max_prob']]
    y_pred = unlabelled.merge(y_pred, on=[idCol])

    new_names = [''.join(e for e in c if e.isalnum() or e=='_')
                 for c in y_pred.columns]
    y_pred.columns = new_names

    ml_labels = y_pred[y_pred['max_prob']>=min_max_prob]
    ml_labels = ml_labels.rename(columns={'target_max_prob':targetCol})
    ml_labels = ml_labels.drop('max_prob', 1)
    manual_labels = y_pred[y_pred['max_prob']<min_max_prob]

    return ml_labels, manual_labels


def distribute_sklearn(spark, df, idCol, targetCol):

    # split data in labelled and unlabelled
    labelled = df.filter(df[targetCol].isNotNull())
    unlabelled = df.filter(df[targetCol].isNull()).drop(targetCol)

    # use sklearn on labelled to build model
    # if labelled is big, get sample and train on sample
    # store model
    joblib.dump(model, 'model.pkl')

    # Load the model from local memory and broadcast it to the cluster.
    # This makes the model object available on worker nodes
    local_model = joblib.load('model.pkl')
    sc = spark.sparkContext
    broadcast_model = sc.broadcast(local_model)

    # Now use pandas UDFs to distribute a function

    @F.pandas_udf(returnType=DoubleType())
    def predict_udf(*cols):
        """
        Takes a series of df features (each a pd.Series)
        and returns a pd.Series of predicted species.
        """
        model = broadcast_model.value
        X = pd.concat(cols, axis="columns")
        predictions = pd.Series(model.predict(X))
        return predictions

    feature_columns = [c for c in df.columns if c != "id"]

    predictions = unlabelled.select(
        F.col("id"),
        predict_udf(*feature_columns).alias('prediction')
    )

    return predictions


def main(spark, df, config):

    df = df.drop('Owner_Name')

    use_spark = config.predict_label['use_spark']
    p_test = config.predict_label['p_test']
    weightCol = config.predict_label['weightCol']
    min_max_prob = config.predict_label['min_max_prob']
    cvs = config.columns['static']['categorical']
    idCol = config.columns['idCol']
    targetCol = config.columns['targetCol']
    model_dir = config.model_dir

    # df.info(memory_usage='deep')
    # df.memory_usage(deep=True).sum()

    df_size = utils.get_df_bytes_in_driver(df, fraction=0.01)
    if df_size < 1e9 and not use_spark:
        df = df.toPandas()
        ml_labels, manual_labels = using_python(
            df, targetCol, idCol, cvs, min_max_prob)
        ml_labels = spark.createDataFrame(ml_labels)
        manual_labels = spark.createDataFrame(manual_labels)
    else:
        ml_labels, manual_labels = using_spark(
            df, p_test, idCol, targetCol, weightCol, min_max_prob, model_dir)

    return ml_labels, manual_labels
