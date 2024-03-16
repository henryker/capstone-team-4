####################################################
# stores dataset info, arguments for load_df
#    description: notes to self. Ignored by load functions
#    name: from the actual name of the file, ignoring extra info (e.g., train/train_{NAME}_1.csv)
#    features: specify columns to keep (ignore all others)
#    feature_types: from kept features, select only those ending with these tags
#    depth: from kaggle description. If .0, aggregation will be performed
#    agg_max (default True): if depth>0, return the max for each case_id for each a feature
#    agg_min (default False): if depth>0, return the min for each case_id for each a feature
#    agg_median (default False): if depth>0, return the max for each case_id for each a feature
#####################################################

dataset_full = {
    "base":{
        "description": "links case_id to WEEK_NUM and target",
        "name":"base",
    },
    "static_0":{
        "description":"contains transaction history for each case_id (late payments, total debt, etc)",
        "name":"static_0",
        "feature_types":["A", "M"],
    },
    "static_cb":{
        "description":"data from an external cb: demographic data, risk assessment, number of credit checks",
        "name":"static_cb",
        "feature_types":["A", "M"],
    },
    "person_1":{
        "description":" internal demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "features":["mainoccupationinc_384A", "incometype_1044T", "housetype_905L"],
        "depth":1,
    },
    "credit_bureau_b_2":{
        "description":"historical data from an external source, num and value of overdue payments",
        "name":"credit_bureau_b_2",
        "features":["pmts_pmtsoverdue_635A","pmts_dpdvalue_108P"],
        "depth":2,
    }
}


dataset_small = {
    "base":{
        "description": "links case_id to WEEK_NUM and target",
        "name":"base",
    },
    "static_0":{
        "description":"contains transaction history for each case_id (late payments, total debt, etc)",
        "name":"static_0",
        "feature_types":["A", "M"],
    },
    "static_cb":{
        "description":"data from an external cb: demographic data, risk assessment, number of credit checks",
        "name":"static_cb",
        "feature_types":["A", "M"],
    },
    "person_1_feats_1":{
        "description":" internal demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "features":["mainoccupationinc_384A", "incometype_1044T"],
        "depth":1,
    },
    "person_1_feats_2":{
        "description":" internal demographic information: zip code, marital status, gender etc (all hashed)",
        "name":"person_1",
        "features":["housetype_905L"],
        "depth":1,
    },
    "credit_bureau_b_2":{
        "description":"historical data from an external source, num and value of overdue payments",
        "name":"credit_bureau_b_2",
        "features":["pmts_pmtsoverdue_635A","pmts_dpdvalue_108P"],
        "depth":2,
    }
}