# https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
# https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/
# https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
# https://sites.temple.edu/tudsc/2017/03/30/measuring-similarity-between-texts-in-python/

# import pyspark libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import types as T
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

# import python libraries
import os
import sys
from importlib import reload

# import python libraries for string matching
import Levenshtein
from pyjarowinkler import distance as pyjaro_distance
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import difflib
import distance as D
from difflib import SequenceMatcher
import nltk
nltk.download('punkt')

# import custom libraries
from sectorisation.steps import string_processing as sp
reload(sp)


def add_owner_cols(manual_labels, tr, owner_cols):
    tr = tr.select(owner_cols).dropDuplicates(subset=['Owner_ID'])
    manual_labels = manual_labels.select('Owner_ID')
    manual_labels = manual_labels.join(tr, ['Owner_ID'], 'left')
    manual_labels.repartition(1).cache().count()
    return manual_labels

def join_tr_idbr(tr, idbr, n_extra_stopwords, kwargs_rm):

    # FIND EXTRA STOPWORDS

    idbr_vocab = sp.column_vocabulary(idbr, 'idbr_name')
    extra_stopwords_idbr = (
        idbr_vocab
        .filter(idbr_vocab['id']<n_extra_stopwords)
        .select('exploded')
        .toPandas()['exploded']
        .tolist()
    )
    tr_vocab = sp.column_vocabulary(tr, 'Owner_Name')
    extra_stopwords_tr = (
        tr_vocab
        .filter(tr_vocab['id']<n_extra_stopwords)
        .select('exploded')
        .toPandas()['exploded']
        .tolist()
    )
    extra_stopwords = list(set(extra_stopwords_idbr + extra_stopwords_tr))

    # PROCESS NAMES

    idbrrm = sp.process_str_col(
        df=idbr,
        column='idbr_name',
        newcol='name_',
        extra_stopwords=extra_stopwords, #+ ['mr', 'mrs', 'dr', 'miss', 'ms'],
        **kwargs_rm
    )
    idbrrm.cache().count()
    trrm = sp.process_str_col(
        df=tr,
        column='Owner_Name',
        newcol='name_',
        extra_stopwords=extra_stopwords, #+ ['mr', 'mrs', 'dr', 'miss', 'ms'],
        **kwargs_rm
    )
    trrm.repartition(1).cache().count()

    # JOIN

    trrm = trrm.join(idbrrm, ['name_'], 'left')

    return trrm


def string_matching(df, s1Col, s2Col):

    df = df.toPandas()

    df[s1Col] = df[s1Col].apply(lambda x: x.lower())
    df[s2Col] = df[s2Col].apply(lambda x: x.lower())

    def jaccard_chars(s1, s2):
        return nltk.jaccard_distance(set(s1), set(s2))

    def jaccard_ngrams_chars(s1, s2):

        ng1_chars = set(nltk.ngrams(s1, n=3))
        ng2_chars = set(nltk.ngrams(s2, n=3))

        return nltk.jaccard_distance(ng1_chars, ng2_chars)

    def jaccard_tokens(s1, s2):

        tokens1 = set(nltk.word_tokenize(s1))
        tokens2 = set(nltk.word_tokenize(s2))

        return nltk.jaccard_distance(tokens1, tokens2)

    def jaccard_ngrams_tokens(s1, s2):

        tokens1 = nltk.word_tokenize(s1)
        tokens2 = nltk.word_tokenize(s2)

        ng1_tokens = set(nltk.ngrams(tokens1, n=3))
        ng2_tokens = set(nltk.ngrams(tokens2, n=3))

        return nltk.jaccard_distance(ng1_tokens, ng2_tokens)

    METHODS = {
        'jaccard_chars'         : jaccard_chars,
        'jaccard_ngrams_chars'  : jaccard_ngrams_chars,
    #    'jaccard_tokens'        : jaccard_tokens,
    #    'jaccard_ngrams_tokens' : jaccard_ngrams_tokens,

        'fuzz_ratio'            : fuzz.ratio,
        'fuzz_partial_ratio'    : fuzz.partial_ratio,
        'fuzz_token_sort_ratio' : fuzz.token_sort_ratio,
        'fuzz_token_set_ratio'  : fuzz.token_set_ratio,

        'edit_distance'         : nltk.edit_distance,
        'levenshtein'           : Levenshtein.ratio,
        'sorensen'              : D.sorensen,
        'sequencematcher'       : SequenceMatcher,
        'pyjaro'                : pyjaro_distance.get_jaro_distance,
    }

    strings = [s2Col, s2Col]
    for method in METHODS:
        print(method)
        df[method] = df[strings].apply(lambda s: METHODS[method](s[0], s[1]), 1)

    return df


def main(tr, manual_labels, idbr, config):

    """
    manual_labels will be joined with tr to get owner columns
    then manual_labels will be joined with idbr for string matching
    """

    owner_cols = config.split_owners['preprocess_tr']['owner_cols']
    n_extra_stopwords = config.recommend_label['n_extra_stopwords']
    kwargs_rm = config.recommend_label['kwargs_rm']
    targetCol = config.columns['targetCol']
    s1Col = config.recommend_label['s1Col']
    s2Col = config.recommend_label['s2Col']

    mlabels = add_owner_cols(manual_labels, tr, owner_cols)
    mlabels = join_tr_idbr(mlabels, idbr, n_extra_stopwords, kwargs_rm)

    mlabels_no_rm = mlabels.filter(mlabels[targetCol].isNull())
    mlabels_rm = mlabels.filter(mlabels[targetCol].isNotNull())
    mlabels_rm = string_matching(mlabels_rm, s1Col, s1Col)

    mlabels_rm = spark.createDataFrame(mlabels_rm)
    mlabels_no_rm = spark.createDataFrame(mlabels_no_rm)

    return mlabels_rm, mlabels_no_rm
