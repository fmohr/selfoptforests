from multiprocessing import set_start_method
if __name__ == '__main__':
    set_start_method("spawn")

# core stuff
import argparse
import os
import resource
import logging
import json
import time

import sklearn
from evalutils import *
from rotationforest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--algorithm', type=str, choices=['randomforest', 'fastrandomforest', 'rotationforest', 'ldaforest-b10', 'ldaforest-c10', 'ldaforest-c10-andpca', 'ldaforest-c20', 'pcaforest', 'zhangpcaforest', 'zhangldaforest', 'wangpcaforest', 'ensembleforest', 'ensembleforest-c10', 'ensembleforest-b10'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folder', type=str, default='./tmp/')
    return parser.parse_args()

def get_learner(args, X):
    num_trees = 100
    learner_name = args.algorithm
    if learner_name == "randomforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = False, timeout = 12 * 3600, rs = np.random.RandomState(args.seed))
    if learner_name == "fastrandomforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = False, light_weight_split_point = True, granularity = 8, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "pcaforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = False, project_before_select = True, allow_global_projection = True, pca_classes = 0, max_number_of_components_to_consider = None, light_weight_split_point = True, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ldaforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = 5, light_weight_split_point = True, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ldaforest-b10":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = 5, light_weight_split_point = True, granularity = 10, beam = 1, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ldaforest-c10":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = 5, light_weight_split_point = True, granularity = 10, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ldaforest-c10-andpca":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, adjust_lda_via_pca = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = 5, light_weight_split_point = True, granularity = 10, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ldaforest-c20":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = 5, light_weight_split_point = True, granularity = 20, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "zhangldaforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = False, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = False, enforce_projections = True, max_number_of_components_to_consider = None, light_weight_split_point = False, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ensembleforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = True, enforce_projections = False, max_number_of_components_to_consider = int(np.sqrt(X.shape[1])), light_weight_split_point = False, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ensembleforest-c10":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = True, enforce_projections = False, max_number_of_components_to_consider = int(np.sqrt(X.shape[1])), light_weight_split_point = True, granularity = 10, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "ensembleforest-b10":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = True, lda_on_canonical_projection = True, project_before_select = False, allow_global_projection = True, enforce_projections = False, max_number_of_components_to_consider = int(np.sqrt(X.shape[1])), light_weight_split_point = True, granularity = 10, beam = 1, timeout = 12 * 3600,rs = np.random.RandomState(args.seed))
    if learner_name == "zhangpcaforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = False, project_before_select = False, allow_global_projection = True, pca_classes = 0, max_number_of_components_to_consider = None, light_weight_split_point = False, timeout = 12 * 3600, rs = np.random.RandomState(args.seed))
    if learner_name == "wangpcaforest":
        return RandomForest(n_trees = num_trees, enable_pca_projections = True, enable_lda_projections = False, project_before_select = False, allow_global_projection = True, pca_classes = 0, max_number_of_components_to_consider = 1, light_weight_split_point = False, timeout = 12 * 3600, rs = np.random.RandomState(args.seed))

if __name__ == '__main__':
    
    # avoid deadlocks in parallelization
    #set_start_method("spawn")
    
    # get params
    args = parse_args()
    
    # ger logger
    logger = logging.getLogger('experimenter')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # memory limits
    memory_limit = 16 * 1024
    logger.info(f"Setting memory limit to {memory_limit}MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    # show CPU settings
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
    
    # folder
    folder = args.folder
    logger.info(f"Folder is: {folder}")
    os.makedirs(folder, exist_ok=True)
    
    # get dataset
    X, y = get_dataset(args.dataset_id)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state=args.seed)
    logger.info(f"{X_train.shape[0]} training instances and {X_test.shape[0]} test instances.")
    logger.info(f"Number of classes in train data: {len(pd.unique(y_train))}")
    logger.info(f"Number of classes in test data: {len(pd.unique(y_test))}")
    labels = list(pd.unique(y))
    
    # train model
    learner = get_learner(args, X_train)
    time_train_start = time.time()
    learner.train(X_train, y_train)
    time_train_end = time.time()
    y_hat_train = learner.predict(X_train)
    y_hat_test = learner.predict(X_test)
    time_predict_end = time.time()
    acc_train = sklearn.metrics.accuracy_score(y_train, y_hat_train)
    acc_test = sklearn.metrics.accuracy_score(y_test, y_hat_test)
    
    # serialize results
    results = {
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "traintime":  int(1000 * (time_train_end - time_train_start)),
        "testtime":  int(1000 * (time_predict_end - time_train_end)),
        "depths": list(learner.get_depths()),
        "numnodes": list(learner.get_numbers_of_nodes())
    }
    logger.info(f"These are the results:\n{results}")
    with open(folder + "/results.json", "w") as outfile: 
        json.dump(results, outfile)