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
from rotationforest import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--folder', type=str, default='./tmp/')
    return parser.parse_args()

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
    
    # compute results
    results = get_results_of_mixed_trees(args.dataset_id, args.seed, 200)
    
    # store results
    logger.info(f"Ready, now writing results.")
    with open(folder + "/results.json", "w") as outfile: 
        json.dump(results, outfile)