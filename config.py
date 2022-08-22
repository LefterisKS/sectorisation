raw_dir = ''
staged_dir = '/user/karace/sectorisation/staged'
model_dir = '/user/karace/sectorisation/test/model'

split_owners = {
    'preprocess_idbr' : {
        'min_death_date' : '2016-01-01',
        'esa_sectors' : [
            'S11001', # Public Non-financial Corporations
            'S11002', # S11003 (S1100N) – PNFC
            'S121',   # Central bank
            'S122',   # Deposit-taking corporations except the central bank
            'S123',   # Money market funds
            'S124',   # Non-MMF investment funds
            'S125',   # Other financial intermediaries
            'S126',   # Financial auxiliaries
            'S127',   # Captive financial institutions and money lenders
            'S128',   # Insurance corporations
            'S129',   # Pension funds
            'S1311',  # Central government
            'S1313',  # Local government
            'S14',    # Households
            'S15',    # Non-profit institutions serving households
            'S2'      # Rest of the world
        ]
    },
    'preprocess_tr' : {
        'start_date' : '2016-01-01',
        'end_date' : '2017-12-31',
        'uk_countries' : [
            '1',   # United Kingdom
            '249', # England
            '250', # Northern Ireland
            '251', # Scotland
            '252'  # Wales
        ],
        'individuals' : False,
        'owner_cols' : [
            'Owner_ID',
            'Owner_Name',
            'URL',
            'Owner_Type',
        ]
    },
    'big_players' : {
        'metric' : 'rank', # cumsum_share, rank
        'maxval' : 10, # positive integer for rank, float 0...1 for cumsum_share
    },
    'string_matching' : {
        'kwargs_em' : {
            'lower_case': True,
            'remove_punc': False,
            'remove_nums': True,
            'vals_to_replace': {'ltd' : 'limited'},
            'tokenize': True,
            'minTokenLength': 1,
            'remove_stopwords': False,
            'extra_stopwords': [],
            'sort_tokens': True
        },
        'fm_active' : True,
        'kwargs_fm' : {
            'lower_case': True,
            'remove_punc': False,
            'remove_nums': True,
            'vals_to_replace': {'ltd' : 'limited'},
            'tokenize': True,
            'minTokenLength': 5,
            'remove_stopwords': True,
            'extra_stopwords': [], # ['mr', 'mrs', 'dr', 'miss', 'ms'],
            'sort_tokens': True
        },
        'n_tokens_min' : 3,
    },
}

columns = {
    'idCol' : 'Owner_ID',
    'targetCol' : 'esa',
    'static' : {
        'categorical' : [
            'Style_Description', # 26
            'Owner_Country_ID',  # 5
            'Turnover_Rating',   # 4 # may be replaced by Turnover_Value
            'Orientation',       # 2
            'Active_Status',     # 2
            'Owner_Type',        # 21
        ],
        'numerical' : [
            'Total_Equity_Assets',
            'Securities_Held',
            'Securities_Bought',
            'Securities_Sold',
            'Turnover_Value',
        ],
        'date' : [],
        'text' : ['Owner_Name']
    },
    'dynamic' : {
        'categorical' : [
            'Currency_Code',        # 76
            'Industry_Description', # 127
            'Sector_Description',   # 40
            'LET_Desc',             # 10
            'Security_Class',       # 14
        ],
        'numerical' : [
            'Shares_Changed',
            'Shares_Held',
            'Value_Held',
            'Value_Of_Shares_Changed',
            'Mkt_Cap_USD',
            'Shares_Outstanding'
        ],
        'date' : ['Report_Date'],
        'text' : []
    },
    'keywords' : [
        'ltd', 'limited', 'plc', 'management', 'mr', 'mrs', 'capital', 'llp',
        'investments', 'group', 'benefit', 'holdings', 'sir', 'asset', 'lord',
        'partners', 'fund', 'company', 'trustees', 'services'
    ],
}

predict_label = {
    'use_spark' : False,
    'p_test' : 0,
    'weightCol' : False,
    'min_max_prob' : 0.6,
}

recommend_label = {
    'n_extra_stopwords' : 10,
    'kwargs_rm' : {
        'lower_case': True,
        'remove_punc': True,
        'remove_nums': True,
        'vals_to_replace': {'ltd' : 'limited'},
        'tokenize': True,
        'minTokenLength': 3,
        'remove_stopwords': True,
        'sort_tokens': True
    },
    's1Col' : 'Owner_Name',
    's2Col' : 'idbr_name',
}
