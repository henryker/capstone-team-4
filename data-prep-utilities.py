import polars as pl



# helper function from contest creator starter notebook
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

    return df



# this function loads a given dataset, performs aggregation if depth >0, and returns train and test sets
def load_df(name, depth=0, features=None, feature_types=None, agg_max=True, agg_min=False, agg_median=False, description=None):
    dataPath = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/"
    if features is not None:
        if "case_id" not in features:
            features = ['case_id']+features
    
    results = []
    for split in ["train", "test"]:
        # load file; it may have been partitioned into multiple csvs
        filenames = os.listdir(dataPath + f"{split}")
        matching_filenames = [f for f in filenames if f.startswith(f"{split}_{name}")]
        # load all partitions
        df_list = []
        for file in matching_filenames:
            df_list.append(pl.read_csv(dataPath+split+"/"+file).pipe(set_table_dtypes))
        df = pl.concat(df_list, how="vertical_relaxed")
        
        # select the columns specified by features and feature_types
        if features is None:
            features = df.columns
        if split == "test":
            features= list(filter(lambda x: x != "target", features))
        if feature_types is not None:
            features = ['case_id']+[f for f in features if f[-1] in feature_types]
        df = df.select(features)
        
        # if depth > 0, aggregate
        if depth > 0:
            # determine the aggregations to perform
            agg_features = list(filter(lambda x: x !='target' and x !='case_id', features))
            nested_lists = [
                [pl.max(f).name.suffix("_max") if agg_max else None, 
                  pl.min(f).name.suffix("_min") if agg_min else None, 
                  pl.median(f).name.suffix("_median") if agg_median else None
                 ] for f in agg_features]
            agg_list = []
            for l in nested_lists:
                agg_list.extend(l)
            agg_list = list(filter(lambda x: x is not None, agg_list))
            # groupby and aggregate
            df = df.group_by("case_id").agg(agg_list)
        results.append(df)
        
    return results



# given a dict of dataset info, this calls load_df to pull data for each and then joins all.
def load_all_dfs(datasets):
    train = {}
    test = {}
    for name in datasets:
        train_df, test_df = load_df(**datasets[name])
        train[name]=train_df
        test[name]=test_df
    
    train_base = train.pop('base')
    test_base = test.pop('base')
    
    for dataset in train:
        train_base = train_base.join(train[dataset], how='left', on='case_id')
        test_base = test_base.join(test[dataset], how='left', on='case_id')
    return train_base, test_base