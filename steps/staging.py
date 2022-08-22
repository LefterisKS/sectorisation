# import spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# import python libraries
import os
from functools import reduce

def main(spark, config):

    """

    """

    live = spark.read.csv(
        config['paths']['raw_live'],
        header=False,
        inferSchema=False,
        sep='\t'
    )
    live = live.toDF(*['id', 'idbr_name', 'sic', 'esa'])
    live = live.withColumn('death_date', F.lit(None))

    dead = spark.read.csv(
        config['paths']['raw_dead'],
        header=False,
        inferSchema=False,
        sep='\t'
    )
    dead = dead.toDF(*['id', 'idbr_name', 'sic', 'esa', 'death_date'])

    df = reduce(DataFrame.unionByName, [live, dead])
    df = df.withColumn('death_date', F.trim(df['death_date']))

    df = df.withColumn(
        'death_date',
        F.to_timestamp(F.col('death_date'), 'dd/MM/yyyy').cast(DateType())
    )

    df.write.parquet(config['paths']['staged'])

    return df

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
        .enableHiveSupport()
        .getOrCreate()
    )

    config = {
        'paths' : {
            'raw_live' : '/user/karace/thomson_reuters/sectorisation/raw/idbr/idbr331live',
            'raw_dead' : '/user/karace/thomson_reuters/sectorisation/raw/idbr/idbr331dead',
            'staged'   : '/user/karace/thomson_reuters/sectorisation/staged/idbr',
        }
    }

    import time
    start = time.time()

    df = main(spark, config)

    print('finished in', (time.time() - start)/60, 'minutes')


###################

# module metadata variables
__author__ = "Lefteris Karachalias"
__date__   = "13/03/2019"
__desc__   = ""
__jira__   = ""

# import spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# import python libraries
import os
from functools import reduce

def extract_transform(
    path,
    format_,
    cols_to_select = False,
    cols_to_drop = False,
    cols_to_rename = False,
    cols_to_filter = False,
    cols_to_cast = False
):
    """

    """

    # EXTRACT

    df = (
        spark.read.format(format_)
        .option('header', 'True')
        .option('inferSchema', 'False')
        .load(path)
    )

    # TRANSFORM

    # cast data types to columns
    if cols_to_cast:
        for column in cols_to_cast:
            dtype = cols_to_cast[column]
            df = df.withColumn(column, df[column].cast(dtype))

    # filter columns
    if cols_to_filter:
        for column in cols_to_filter:
            values = cols_to_filter[column]
            df = df.filter(df[column].isin(values))

    # select columns
    if cols_to_select:
        df = df.select(cols_to_select)

    # drop columns
    if cols_to_drop:
        df = df.drop(*cols_to_drop)

    # rename columns
    if cols_to_rename:
        for column in cols_to_rename:
            new_column = cols_to_rename[column]
            df = df.withColumnRenamed(column, new_column)

    # drop duplicates
    df = df.dropDuplicates()

    return df


def main(spark, config):

    """

    """

    dfs = {}

    # extract, transform
    for file in config['files_to_extract_transform']:
        kwargs = config['files_to_extract_transform'][file]
        dfs[file] = extract_transform(**kwargs)
        # dfs[file].cache().count()

    # union files
    for file in config['files_to_union']:
        files_to_union = config['files_to_union'][file]
        dfs_to_union = [dfs[f] for f in files_to_union]
        dfs[file] = reduce(DataFrame.unionByName, dfs_to_union)

    # join files
    files_to_join = config['files_to_join']
    first_file = files_to_join[0][0]
    df = dfs[first_file]
    for file in files_to_join[1:]:
        filename, keys = file
        df = df.join(dfs[filename], keys, 'left')

    # save final staged table in dictionary
    dfs['staged'] = df

    # load
    df.write.parquet(config['output_dir'])

    return dfs


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
        .enableHiveSupport()
        .getOrCreate()
    )

    input_dir = '/user/karace/thomson_reuters/sectorisation/raw/tr'

    config = {

        'files_to_extract_transform' : {

            'Consolidated_Funds_Holdings_History' : {
                'path'           : os.path.join(input_dir, 'Consolidated_Funds_Holdings_History'),
                'format_'        : 'json',
                'cols_to_drop'   : ['_corrupt_record'],
            },
            'Mutual_Funds_Holdings_History' : {
                'path'           : os.path.join(input_dir, 'Mutual_Funds_Holdings_History'),
                'format_'        : 'json',
            },
            'Owner_History' : {
                'path'           : os.path.join(input_dir, 'Owner_History'),
                'format_'        : 'csv',
                'cols_to_drop'   : ['Report_Date'],
                'cols_to_rename' : {'Country_ID':'Owner_Country_ID'},
            },
            'Exchange_Listed' : {
                'path'           : os.path.join(input_dir, 'Exchange_Listed'),
                'format_'        : 'csv',
                'cols_to_select' : ['Instrument_ID', 'Exchange_Listed_Flag'],
                'cols_to_rename' : {'Instrument_ID' : 'UID'},
            },
            'Owner_Type' : {
                'path'           : os.path.join(input_dir, 'Owner_Type'),
                'format_'        : 'csv',
                'cols_to_rename' : {'Description' : 'Owner_Type'},
            },
            'Investment_Style' : {
                'path'           : os.path.join(input_dir, 'Investment_Style'),
                'format_'        : 'csv',
            },
            'EOD_History_Lookup' : {
                'path'           : os.path.join(input_dir, 'EOD_History_Lookup'),
                'format_'        : 'parquet',
                'cols_to_rename' : {'isin_f' : 'UID'},
            },
            'Equity_Issuance_All': {
                'path'           : os.path.join(input_dir, 'Equity_Issuance_All'),
                'format_'        : 'parquet',
                'cols_to_select' : [
                    'Security_ID',
                    'Issuer_Name',
                    'UID',
                    'Industry_Description',
                    'Sector_Description',
                    'LET_Desc',
                    'Listed',
                    'Trade_Date',
                    'Security_Class',
                    'Mkt_Cap_USD',
                    'Shares_Outstanding'
                ],
                'cols_to_rename' : {'Trade_Date' : 'Quarter'},
            },
        },

        'files_to_union' : {
            'Funds_Holdings_History' : [
                'Consolidated_Funds_Holdings_History',
                'Mutual_Funds_Holdings_History'
            ]
        },

        'files_to_join' : [
            ['Funds_Holdings_History',  []],
            ['Owner_History',           ['Owner_ID']],
            ['Equity_Issuance_All',     ['Security_ID', 'Quarter']],
            ['EOD_History_Lookup',      ['UID']],
            ['Exchange_Listed',         ['UID']],
            ['Investment_Style',        ['Style_ID']],
            ['Owner_Type',              ['Owner_Type_ID']],
        ],

        'output_dir' : '/user/karace/thomson_reuters/sectorisation/staged/tr'
    }

    import time
    start = time.time()

    dfs = main(spark, config)

    print('finished in', (time.time() - start)/60, 'minutes')
