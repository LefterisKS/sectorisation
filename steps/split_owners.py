# import pyspark libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import types as T
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

# import python libraries
import sys
from importlib import reload

# import custom libraries
from sectorisation.steps import utils
from sectorisation.steps import string_processing as sp
reload(utils)
reload(sp)

def preprocess_idbr(df, min_death_date, esa_sectors):
    """
    Clean IDBR data before string matching.

    Args:
        df      :
        config  :

    Returns:
        df      :
        df_dups :

    Notes:
        Filter businesses that are live or have died after a certain date
        Select id, idbr_name, esa, drop missing, drop duplicates
        Regroup esa sectors
        Drop businesses that do not belong to the 16 sectors
        Split rows/businesses whose name appears only once
        from those with frequency>1
    """

    # filter businesses that are live or have died after a certain date
    df = df.filter(
        (df['death_date'] > F.lit(min_death_date)) |
        (df['death_date'].isNull())
    )

    # regroup esa sectors
    df = utils.extract_substring(df, 'esa', 'esa16', esa_sectors)
    df = df.drop('esa').withColumnRenamed('esa16', 'esa')

    # drop businesses that do not belong to the 16 sectors
    df = df.filter(df['esa'].isNotNull())

    # drop businesses that are mapped to more than one sectors
    df = df.select(['idbr_name', 'esa']).dropna().dropDuplicates()
    df, _ = utils.split_unique_duplicates(df, 'idbr_name')

    return df


def preprocess_tr(
    df,
    start_date,
    end_date,
    uk_countries,
    individuals,
    owner_cols
):
    """
    Clean Thomson Reuters data before string matching.

    Args:
        df      :
        config  :

    Returns:
        df : includes all shareowners to be sectorised
        ids_dups : wrong records where same id appears with different names
        dfsm :

    Notes:
        Filter UK owners
        Filter businesses that have transactions within specified period
        Drop individual investors, if selected
        Select owner id and owner name, drop missing, drop duplicates
        Split owners whose name appears only once
        from those with frequency>1
    """

    # select UK owners and transactions with a specified period
    df = df.filter(
        (df['Owner_Country_ID'].isin(uk_countries))&
        (df['Report_Date'] >= F.lit(start_date)) &
        (df['Report_Date'] <= F.lit(end_date))
    )

    # exclude individual investors (assuming they are all S14)
    if not individuals:
        df = df.filter(df['Owner_Type']!='Individual Investor')

    owners = (
        df
        .select(owner_cols)
        .dropna()
        .dropDuplicates(subset=['Owner_ID', 'Owner_Name'])
    )

    # same id different name: wrong records we need to drop
    ids_uniq, ids_dups = utils.split_unique_duplicates(owners, 'Owner_ID')

    # names_dups cannot be used for string matching but can be scored with ML
    df = utils.left_join(
        df1=df,
        df2=ids_uniq.select('Owner_ID'),
        id1='Owner_ID',
        id2='Owner_ID',
        mode='inner'
    )

    return df, ids_dups


def big_players(df, metric, maxval, owner_cols):

    owner_info = df.select(owner_cols).dropDuplicates(subset=['Owner_ID'])

    bp = df.withColumn('Value_Held', df['Value_Held'].cast(T.DoubleType()))

    bp = bp.select('Owner_ID', 'Value_Held').groupBy('Owner_ID').sum()
    bp = bp.withColumnRenamed('sum(Value_Held)', 'value')
    total_value = bp.select('value').groupBy().sum()
    bp = bp.crossJoin(total_value)
    # divide to get % of value share for each owner
    bp = bp.withColumn('share', bp['value']/bp['sum(value)'])
    w = (
        Window
        .partitionBy()
        .orderBy(F.desc('share'))
        .rowsBetween(-sys.maxsize, 0)
    )
    bp = bp.withColumn('cumsum_share', F.sum(bp['share']).over(w))
    bp = bp.select('*', F.rank().over(w).alias('rank'))

    bp = bp.filter(bp[metric] <= maxval)

    bp = bp.drop('sum(value)')

    bp = bp.join(owner_info, ['Owner_ID'], 'left')

    # remove big players from businesses to be sectorised by string matching
    # and machine learning. the big players will be sectorised manually.
    # the small players will continue to next stage
    df = utils.left_join(
        df1=df,
        df2=bp.select('Owner_ID'),
        id1='Owner_ID',
        id2='Owner_ID',
        mode='outer'
    )

    return df, bp


def string_matching(idbr, tr, kwargs_em, fm_active, kwargs_fm, n_tokens_min):

    # EXACT MATCHING

    # IDBR
    idbrem = sp.process_str_col(
        df=idbr, column='idbr_name', newcol='name_', **kwargs_em)
    # for cases when the same name_ happen to map to the same esa
    idbrem = idbrem.dropDuplicates(subset=['name_', 'esa'])
    # for cases when the same name_ map to different esa
    # we can trust only unique name_
    idbrem, _ = utils.split_unique_duplicates(idbrem, 'name_')

    # TR
    trem = sp.process_str_col(
        df=tr, column='Owner_Name', newcol='name_', **kwargs_em)
    # processed names unique and duplicates
    # only unique can be used for string matching
    # duplicates will be scored by ML
    pnames_uniq, _ = utils.split_unique_duplicates(trem, 'name_')

    # linking on processed name
    exact_labels = pnames_uniq.join(idbrem, ['name_'], how='inner')

    # remove from idbr the businesses that gave an exact match
    idbr = utils.left_join(
        df1=idbr,
        df2=exact_labels.select('idbr_name'),
        id1='idbr_name',
        id2='idbr_name',
        mode='outer'
    )

    # remove from tr the businesses that gave an exact match
    tr = utils.left_join(
        df1=tr,
        df2=exact_labels.select('Owner_ID'),
        id1='Owner_ID',
        id2='Owner_ID',
        mode='outer'
    )

    # FUZZY MATCHING

    if fm_active:

        # IDBR
        idbrfm = sp.process_str_col(
            df=idbr, column='idbr_name', newcol='name_', **kwargs_fm)
        # for cases when the same name_ happen to map to the same esa
        idbrfm = idbrfm.dropDuplicates(subset=['name_', 'esa'])
        # for cases when the same name_ map to different esa
        # we can trust only unique name_
        idbrfm, _ = utils.split_unique_duplicates(idbrfm, 'name_')
        # keep only longer tokenised names
        idbrfm = idbrfm.where(F.size(F.col("name_")) >= n_tokens_min)

        # TR
        trfm = sp.process_str_col(
            df=tr, column='Owner_Name', newcol='name_', **kwargs_fm)
        # processed names unique and duplicates
        # only unique can be used for string matching
        # duplicates will be scored by ML
        pnames_uniq, _ = utils.split_unique_duplicates(trfm, 'name_')
        # keep only longer tokenised names
        pnames_uniq = pnames_uniq.where(F.size(F.col("name_")) >= n_tokens_min)

        # linking on processed name
        fuzzy_labels = pnames_uniq.join(idbrfm, ['name_'], how='inner')

        # remove from idbr the businesses that gave a fuzzy match
        idbr = utils.left_join(
            df1=idbr,
            df2=fuzzy_labels.select('idbr_name'),
            id1='idbr_name',
            id2='idbr_name',
            mode='outer'
        )

    labels = exact_labels.union(fuzzy_labels).select('Owner_ID', 'esa')

    return idbr, labels


def main(idbr, tr, config):

    """
    """
    params = config.split_owners
    min_death_date = params['preprocess_idbr']['min_death_date']
    esa_sectors = params['preprocess_idbr']['esa_sectors']
    start_date = params['preprocess_tr']['start_date']
    end_date = params['preprocess_tr']['end_date']
    uk_countries = params['preprocess_tr']['uk_countries']
    individuals = params['preprocess_tr']['individuals']
    owner_cols = params['preprocess_tr']['owner_cols']
    metric = params['big_players']['metric']
    maxval = params['big_players']['maxval']
    kwargs_em = params['string_matching']['kwargs_em']
    fm_active = params['string_matching']['fm_active']
    kwargs_fm = params['string_matching']['kwargs_fm']
    n_tokens_min = params['string_matching']['n_tokens_min']

    idbr = preprocess_idbr(idbr, min_death_date, esa_sectors)
    idbr.cache().count()

    tr, ids_dups = preprocess_tr(
        tr, start_date, end_date, uk_countries, individuals, owner_cols)
    tr.cache().count()
    ids_dups.cache().count()

    tr, bp = big_players(tr, metric, maxval, owner_cols)
    tr.cache().count()
    bp.cache().count()

    # names_uniq can be used for string matching to get labels
    ids_uniq = tr.select('Owner_ID', 'Owner_Name').dropDuplicates()
    names_uniq, _ = utils.split_unique_duplicates(ids_uniq, 'Owner_Name')
    names_uniq.cache().count()

    idbr, labels = string_matching(
        idbr, names_uniq, kwargs_em, fm_active, kwargs_fm, n_tokens_min)
    idbr.cache().count()
    labels.cache().count()

    return idbr, ids_dups, bp, tr, labels
