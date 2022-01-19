import openml
import pandas as pd
import numpy as np

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
    
    # impute missing values
    cateogry_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
    obsolete_cols = []
    for column in df:
        if df[column].isnull().any():
            if(column in cateogry_columns):
                if (all(df[column].values == None)):
                    obsolete_cols.append(column)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
    df = df.drop(columns=obsolete_cols)
    
    # prepare label column as numpy array
    y = np.array(list(df[ds.default_target_attribute].values))
    
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    
    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= (len(pd.unique(df[att])) - 1)
        if expansion_size > 10**6:
            break
    
    if expansion_size < 10**6:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], drop_first=True).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    
    print(f"Shape of the data after expansion: {X.shape}")
    if X.shape[0] != num_rows:
        raise Exception("Number of rows not ok!")
    return X, y