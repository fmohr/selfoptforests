import openml
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
from rotationforest import *
from tqdm import tqdm

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


def get_scores_of_trained_trees(basis, s, X_test, y_test):
    points = get_all_combos_for_sum(s)
    scores_valid = []
    scores_test = []
    for t in points:
        rf_mixted = get_mixted_rf(basis, t)
        scores_valid.append(rf_mixted.get_oob_error())
        scores_test.append(sklearn.metrics.accuracy_score(y_test, rf_mixted.predict(X_test)))
    return points, scores_valid, scores_test
    

def get_mixted_rf(rfs, nums):
    rf = RandomForest()
    rf.setup(rfs[0].X, rfs[0].y)
    for rf_orig, num in zip(rfs, nums):
        rf.indices_per_tree.extend(rf_orig.indices_per_tree[:num])
        rf.trees.extend(rf_orig.trees[:num])
    return rf



def get_votes_of_ensemble(vote_schemes, nums):
    votes_valid = None
    votes_test = None
    
    
    for (votes_valid_i, votes_test_i), num in zip(vote_schemes, nums):
        
        violations = votes_valid_i[num] > num
        if np.count_nonzero(violations) > 0:
            raise Exception(f"Too many votes for some instances. Only {num} votes theoretically possible, but here we see: {votes_valid_i[num][violations]}")
        
        
        if num > 0:
            if votes_valid is None:
                votes_valid = votes_valid_i[num].copy()
                votes_test = votes_test_i[num].copy()
            else:
                votes_valid += votes_valid_i[num]
                votes_test += votes_test_i[num]
                
    violations = np.sum(votes_valid, axis=1) > sum(nums)
    if np.count_nonzero(violations) > 0:
        raise Exception(f"Too many votes for some instances. Only {sum(nums)} votes theoretically possible, but here we see: {votes_valid[violations]}")
    return votes_valid, votes_test

def get_performance_of_ensemble(votes_scheme, nums, labels, y_valid, y_test):
    votes_valid, votes_test = get_votes_of_ensemble(votes_scheme, nums)
    y_hat_valid = [labels[l] for l in np.argmax(votes_valid, axis=1)]
    y_hat_test = [labels[l] for l in np.argmax(votes_test, axis=1)]
    return tuple([np.round(sklearn.metrics.accuracy_score(y, y_hat), 4) for y, y_hat in zip([y_valid, y_test], [y_hat_valid, y_hat_test])])

def get_all_combos_for_sum(num_types, num_trees, step_size = 1):
    if num_types == 1:
        return [[num_trees]]
    combos = []
    for c1 in range(0, num_trees + 1, 1):
        sub_combos = get_all_combos_for_sum(num_types - 1, num_trees - c1, step_size = step_size)
        combos.extend([[c1] + sub_combo for sub_combo in sub_combos])
    return combos

def get_results_of_mixed_trees(openmlid, seed, max_num_trees):
    
    # load and split data
    X, y = get_dataset(openmlid)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state = np.random.RandomState(seed))
    
    light_weight_split_points = False
    max_number_of_components_to_consider = 10

    # create random forests
    rf1 = RandomForest(n_trees = max_num_trees, enable_lda_projections = False, enable_pca_projections = False, max_number_of_components_to_consider = max_number_of_components_to_consider, light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    rf2 = RandomForest(n_trees = max_num_trees, enforce_projections = True, enable_lda_projections = True, max_number_of_components_to_consider = max_number_of_components_to_consider, enable_pca_projections = False, lda_on_canonical_projection = True, light_weight_split_point = light_weight_split_points,  rs = np.random.RandomState(seed))
    rf3 = RandomForest(n_trees = max_num_trees, enforce_projections = True, enable_lda_projections = False, enable_pca_projections = True, pca_classes = 0, pca_kernel="linear", light_weight_split_point = light_weight_split_points, max_number_of_components_to_consider = max_number_of_components_to_consider, rs = np.random.RandomState(seed))
    rf4 = RandomForest(n_trees = max_num_trees, enforce_projections = True, enable_lda_projections = False, enable_pca_projections = True, pca_classes = 0, pca_kernel="rbf", max_number_of_components_to_consider = max_number_of_components_to_consider, light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    rf4a = RandomForest(n_trees = max_num_trees, enforce_projections = True, enable_lda_projections = False, enable_pca_projections = True, pca_classes = 0, pca_kernel="poly", max_number_of_components_to_consider = max_number_of_components_to_consider, light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    rf5 = RotationForest(n_trees = max_num_trees, max_number_of_components_to_consider = max_number_of_components_to_consider, light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    rf6 = RotationForest(n_trees = max_num_trees, max_number_of_components_to_consider = max_number_of_components_to_consider, pca_kernel="rbf", light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    rf7 = RotationForest(n_trees = max_num_trees, max_number_of_components_to_consider = max_number_of_components_to_consider, pca_kernel="poly", light_weight_split_point = light_weight_split_points, rs = np.random.RandomState(seed))
    
    
    forests = {
        "standard RF": rf1,
        "LDA RF": rf2,
        "PCA RF": rf3,
        "RBF-PCA RF": rf4,
        "Poly-PCA RF": rf4a,
        "Rotation RF": rf5,
        "RBF-Rotated RF": rf6,
        "Poly-Rotated RF": rf7
    }
    
    #for key in list(forests.keys()):
     #   if X.shape[0] > 100000 and key in ["KPCA RF", "RBF-Rotated RF", "Poly-Rotated RF"]:
      #      del forests[key]
    
    keys = []
    for name, rf in forests.items():
        print(f"Training {name}")
        keys.append(name)
        rf.train(X_train, y_train)
    
    # get votes of the trees in the ensembles
    print("Ready, computing votes")
    votes = []
    for name in tqdm(keys):
        rf = forests[name]
        votes_valid, votes_test = rf.get_cummulative_votes(X_test)
        votes.append((votes_valid.tolist(), votes_test.tolist()))
    print("Votes ready, computing scores.")
    for i in range(len(votes)):
        for j in range(i):
            print(i, j, votes[i][0][0] == votes[j][0][0])
    
    # compute scores for all possible partial and mixed ensembles
    if False:
        scores = {}
        for n_trees in tqdm(range(1, max_num_trees + 1)):
            scores_for_size = []
            points = get_all_combos_for_sum(len(votes), n_trees, 5)
            for c in points:
                scores_for_size.append(get_performance_of_ensemble(votes, c, rf1.labels, y_train, y_test))
            scores[n_trees] = (points, scores_for_size)
        
    return {"algorithms": keys, "votes": votes, "times": [list(np.round(forests[name].get_train_times(), 4)) for name in keys]}