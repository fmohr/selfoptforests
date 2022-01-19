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
    parser.add_argument('--algorithm', type=str, choices=['randomforest', 'rotationforest', 'pcaforest'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folder', type=str, default='./tmp/')
    return parser.parse_args()

def get_learner(args):
    num_trees = 100
    learner_name = args.algorithm
    if learner_name == "randomforest":
        return RandomForest(n_trees = num_trees, rotation = False, rs = np.random.RandomState(args.seed))
    if learner_name == "pcaforest":
        return RandomForest(n_trees = num_trees, rotation = True, rs = np.random.RandomState(args.seed))

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
    learner = get_learner(args)
    time_train_start = time.time()
    learner.train(X_train, y_train)
    time_train_end = time.time()
    y_hat = learner.predict(X_test)
    time_predict_end = time.time()
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    
    # serialize results
    results = {
        "accuracy": acc,
        "traintime":  int(1000 * (time_train_end - time_train_start)),
        "testtime":  int(1000 * (time_predict_end - time_train_end)),
        "depths": list(learner.get_depths()),
        "numnodes": list(learner.get_numbers_of_nodes())
    }
    logger.info(f"These are the results:\n{results}")
    with open(folder + "/results.json", "w") as outfile: 
        json.dump(results, outfile)