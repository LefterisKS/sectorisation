# import pyspark libraries
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover


def column_vocabulary(df, column):

    rgxtok = RegexTokenizer(
        inputCol=column,
        outputCol='tokenized',
        pattern="\\W"
    )

    df = rgxtok.transform(df)

    vocabulary = (
        df
        .withColumn('exploded', F.explode(F.col('tokenized')))
        .groupBy('exploded')
        .count()
        .sort("count", ascending=False)
    )

    sqlexp = "rank() over (order by {} desc)".format('count')
    vocabulary = vocabulary.withColumn('id', F.expr(sqlexp)-1)

    return vocabulary.select(['id', 'exploded', 'count'])


def lower_case_col(df, column, newcol):
    return df.withColumn(newcol, F.lower(F.col(column)))

def remove_punc_col(df, column, newcol):
    return df.withColumn(newcol, F.regexp_replace(column, r'[^\w\s]', ''))

def remove_nums_col(df, column, newcol):
    return df.withColumn(newcol, F.regexp_replace(column, r'[\d+]', ''))

def replace_vals_col(df, column, newcol, vals_to_replace):
    for value1 in vals_to_replace:
        value2 = vals_to_replace[value1]
        df = df.withColumn(newcol, F.regexp_replace(column, value1, value2))
    return df

def tokenize_col(df, column, newcol, minTokenLength):

    # tokenize the name
    rgxtok = RegexTokenizer(
        inputCol=column,
        outputCol='tokenized',
        minTokenLength=minTokenLength,
        pattern="\\W"
    )

    df = rgxtok.transform(df)

    df = df.drop(column)
    df = df.withColumnRenamed('tokenized', newcol)

    return df

def remove_stopwords_col(df, column, newcol, extra_stopwords):

    stopwords = StopWordsRemover.loadDefaultStopWords('english') + extra_stopwords
    remover = StopWordsRemover(inputCol=column, outputCol='filtered', stopWords=stopwords)
    df = remover.transform(df)
    df = df.drop(column)
    df = df.withColumnRenamed('filtered', newcol)

    return df

def sort_tokens_col(df, column, newcol):

    def def_sort(x):
        return sorted(x, key=lambda x:x, reverse=False)

    udf_sort = F.udf(def_sort, ArrayType(StringType()))

    # sort tokens in tokenized name to ensure matching between idbr and tr
    df = df.withColumn(newcol, udf_sort(column))

    return df

def process_str_col(
    df,
    column,
    newcol,
    lower_case=True,
    remove_punc=True,
    remove_nums=True,
    vals_to_replace=False,
    tokenize=True,
    minTokenLength=1,
    remove_stopwords=True,
    extra_stopwords=[],
    sort_tokens=True
):

    df = df.withColumn(newcol, df[column])

    if lower_case:
        df = lower_case_col(df, newcol, newcol)

    if remove_punc:
        df = remove_punc_col(df, newcol, newcol)

    if remove_nums:
        df = remove_nums_col(df, newcol, newcol)

    if vals_to_replace:
        df = replace_vals_col(df, newcol, newcol, vals_to_replace)

    if tokenize:
        df = tokenize_col(df, newcol, newcol, minTokenLength)

    if tokenize and remove_stopwords:
        df = remove_stopwords_col(df, newcol, newcol, extra_stopwords)

    if tokenize and sort_tokens:
        df = sort_tokens_col(df, newcol, newcol)

    return df
