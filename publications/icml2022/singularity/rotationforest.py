import itertools as it
import numpy as np
import sklearn.datasets
import sklearn.decomposition
import sklearn.discriminant_analysis
from tqdm import tqdm
import time
#from bayes_opt import BayesianOptimization

def plot_dataset(X, y):
    fig, ax = plt.subplots()
    labels = list(np.unique(y))
    if X.shape[1] == 1:
        X = np.column_stack([X, np.linspace(0, 1, X.shape[0])])
    for label in labels:
        ax.scatter(X[y == label,0], X[y == label,1], s=5)
    return fig, ax

def entropy(p):
    return -sum([q * np.log(q) / np.log(2) if q > 0 else 0 for q in p])

class Node:
    def __init__(self):
        self.split_point = None
        
    def get_depth(self):
        if self.split_point is None:
            return 0
        return 1 + max(self.lc.get_depth(), self.rc.get_depth())
    
    def get_number_of_nodes(self):
        if self.split_point is None:
            return 1
        return 1 + self.lc.get_number_of_nodes() + self.rc.get_number_of_nodes()

    
class Projection:

    def __init__(self, vec):
        self.vec = vec
    
    def transform(self, X):
        return np.array([np.dot(self.vec, x) for x in X])
    
class DecisionTreeClassifier:
    
    def __init__(self, eta = 1, pi = 0.99, l = np.inf, p = None, enable_pca_projections = True, enable_lda_projections = True, rs = None, pca_classes = 2, project_before_select = False, enforce_projections = False, max_number_of_components_to_consider = 1, allow_global_projection = False, light_weight_split_point = False, lda_on_canonical_projection = False, granularity = 10, beam = None, adjust_lda_via_pca = False):
        self.eta = eta
        self.pi = pi
        self.l = l
        self.p = p # the maximum number of (random) attributes to consider in each split. Specify None to use all
        self.pca_classes = pca_classes
        self.max_class_combos = 1
        self.min_instances_for_rotation = 5
        self.min_score_to_not_rotate = 0.2
        self.rs = rs if rs is not None else np.random.RandomState()
        self.project_before_select = project_before_select
        self.enforce_projections = enforce_projections
        self.max_number_of_components_to_consider = max_number_of_components_to_consider if max_number_of_components_to_consider is not None else 10**10
        self.allow_global_projection = allow_global_projection
        self.enable_pca_projections = enable_pca_projections
        self.enable_lda_projections = enable_lda_projections
        self.lda_on_canonical_projection = lda_on_canonical_projection
        self.rotation = self.enable_pca_projections or self.enable_lda_projections
        self.light_weight_split_point = light_weight_split_point
        self.granularity = granularity
        self.beam = beam
        self.adjust_lda_via_pca = adjust_lda_via_pca
        
        # train time stats
        self.time_projections = 0
        self.time_splitpoints = 0
    
    def train(self, X, y):
        self.num_atts = X.shape[1]
        self.labels = list(np.unique(y))
        self.majority_label = self.labels[np.argmax([np.count_nonzero(self.labels == l) for l in self.labels])]
        
        # scale data
        if self.rotation:
            scaler = sklearn.preprocessing.StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            self.scaler = None
            X_scaled = X
        
        # train time stats
        self.time_projections = 0
        self.time_splitpoints = 0
        
        time_start = time.time()
        self.model = self.train_tree(X_scaled, y)
        self.train_time = time.time() - time_start
    
    def train_tree(self, X, y):
        
        # preliminaries (lines 1-3)
        n = X.shape[0]
        if n <= 0:
            print("Warning: empty node!")
            n = Node()
            n.label = self.majority_label
            return n
        
        n_unique = len({tuple(row) for row in X})
        ni = [np.count_nonzero(y == l) for l in self.labels]
        majority_label = np.argmax(ni)
        label_order = np.argsort(ni)
        purity = max(ni) / n
        
        # is this a leaf node?
        if n_unique <= self.eta or purity >= self.pi:
            label = self.labels[majority_label]
            n = Node()
            n.label = label
            return n
        
        # initialize decision variables
        split_point, best_score = None, -np.inf
        best_is_numeric_split = True
        X_decision = None
        
        # define the indices of attributes over which the projection should take place and, partially, which attributes to use
        if self.project_before_select:
            indices_of_attributes_to_project_over = list(range(X.shape[1]))
        else:
            indices_of_attributes_to_project_over = list(range(X.shape[1]))
            if self.p is not None:
                num_new_features = min(X.shape[1], max(2, int(self.p)), self.max_number_of_components_to_consider)
                indices_of_attributes_to_project_over = sorted(self.rs.choice(indices_of_attributes_to_project_over, num_new_features, replace=False))
        X_for_projection = X[:,indices_of_attributes_to_project_over]
        
        # compute dataset modifications that may be considered for splits
        time_ds_start = time.time()
        datasets = []
        if not self.enforce_projections:
            datasets.append((X, None, "None", None))
        if self.rotation and X.shape[0] > self.min_instances_for_rotation:
            
            # lda projections
            if self.enable_lda_projections:
                n_components = min(self.max_number_of_components_to_consider, X_for_projection.shape[1], len(np.unique(y)) - 1)
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components = n_components)
                try:
                    
                    if self.lda_on_canonical_projection:
                            if self.adjust_lda_via_pca and X_for_projection.shape[0] < X_for_projection.shape[1] * len(np.unique(y)):
                                transformer = sklearn.pipeline.Pipeline([("pca", sklearn.decomposition.PCA()), ("lda", lda)])
                                transformer.fit(X_for_projection, y)
                                print("Reducing with PCA")
                            else:
                                lda.fit(X_for_projection, y)
                                transformer = lda
                            
                            X_projected = transformer.transform(X_for_projection)
                            datasets.append((X_projected, transformer, f"LDA class projection", None))
                    
                    else:
                    
                        lda.fit(X_for_projection, y)
                        accepted_labels = [self.labels[l] for l in label_order[-self.max_number_of_components_to_consider:] if self.labels[l] in y]
                        for label, direction in zip(lda.classes_, lda.coef_):
                            if label not in accepted_labels:
                                continue

                            projection = Projection(direction)
                            X_projected = projection.transform(X_for_projection).reshape((X_for_projection.shape[0], 1))
                            
                            #means = np.array([np.mean(X_projected[(y == label) == p]) for p in [True, False]])
                            #offsets_min = np.array([min(means)])
                            #offsets_max = np.array([max(means)])
                            datasets.append((X_projected, projection, f"LDA subspace projection", None))
                except:
                    print("Observed error in LDA, ignoring this dimension")
                
            # pca projections
            if self.enable_pca_projections:
                
                # one PCA to the full data
                if self.allow_global_projection:
                    pca = sklearn.decomposition.PCA()
                    datasets.append((pca.fit_transform(X_for_projection)[:,:self.max_number_of_components_to_consider], pca, "global pca", None))

                # one PCA to the data reduced to each class
                for k in range(1, self.max_class_combos + 1):
                    for labelset in it.combinations([self.labels[l] for l in label_order[-self.pca_classes:]], k):#label_order:#[-self.pca_classes:]:
                        pca = sklearn.decomposition.PCA()
                        instances_for_label = X_for_projection[[l in labelset for l in y]]
                        if len(instances_for_label) >= self.min_instances_for_rotation and np.var(instances_for_label) > 0:
                            try:
                                pca.fit(instances_for_label)
                                X_transformed = pca.transform(X_for_projection)[:,:self.max_number_of_components_to_consider]
                                if X_transformed.shape == X_for_projection.shape:
                                    datasets.append((X_transformed, pca, f"PCA over labelset {labelset}", None))
                            except np.linalg.LinAlgError:
                                print("observed error, ignoring result of PCA.")
        time_ds_end = time.time()
        self.time_projections += (time_ds_end - time_ds_start)
        #print(f"Time to compute datasets: {time_ds_end - time_ds_start}")
        
        # if there are no datasets, just run normal split
        if len(datasets) == 0:
            print("No dataset, falling back to standard.")
            datasets.append((X, None, "None", None))
        
        #print(f"Now considering {len(datasets)} datasets")
        att_cnt = 0
        for ds_index, (X_local, transformation, trans_name, offsets) in enumerate(datasets):
            
            # plot dataset
            #fig, ax = plot_dataset(X_local, y)
            #ax.set_title(trans_name)
            #fig.show()
            
            # if attributes were NOT selected BEFORE projection, select them now
            indices_of_possible_split_attributes = list(range(min(X_local.shape[1], self.max_number_of_components_to_consider)))
            if transformation is None or self.project_before_select:
                if self.p is not None:
                    num_new_features = min(len(indices_of_possible_split_attributes), int(self.p))
                    indices_of_possible_split_attributes = sorted(self.rs.choice(indices_of_possible_split_attributes, num_new_features, replace=False))
            
            # get the split decision
            for att_index in indices_of_possible_split_attributes:
                att_cnt += 1
                col = X_local[:, att_index]
                numeric_split = col.dtype in [float, int]
                time_att_eval_start = time.time()
                if numeric_split:
                    
                    # get offsets
                    offset_min = -np.inf if offsets is None else offsets[0][att_index]
                    offset_max = np.inf if offsets is None else offsets[1][att_index]
                    v, score = self.evaluate_numeric_attribute(col, y, best_score, min_offset = offset_min, max_offset = offset_max)
                else:
                    v, score = self.evaluate_categorical_attribute(col, y)
                if v is not None and score > best_score:
                    split_point, best_score = (att_index, v), score
                    best_is_numeric_split = numeric_split
                    X_decision = X_local
                    transformation_for_decision = transformation
                time_att_eval_end = time.time()
                self.time_splitpoints += (time_att_eval_end - time_att_eval_start)
            
            # break if the standard split was already good enough
            if ds_index == 0 and best_score >= - self.min_score_to_not_rotate:
                break
        
        #print(f"Split point computed after {time.time() - time_ds_end}s. Considered {att_cnt} attributes.")
        
        # we cannot distinguish points here anymore
        if split_point is None:
            label = self.labels[np.argmax(ni)]
            n = Node()
            n.label = label
            return n
        
        # create node with two children
        mask = X_decision[:,split_point[0]] <= split_point[1] if best_is_numeric_split else X_decision[:,split_point[0]] == split_point[1]
        node = Node()
        node.split_point = split_point
        node.indices_of_attributes_to_project_over = indices_of_attributes_to_project_over
        if transformation_for_decision is None:
            node.transformer = None 
        else:
            #projection_matrix = np.matmul(transformation_for_decision.components_, transformation_for_decision.components_.T)
            #node.transformation_vector = transformation_for_decision.components_[split_point[0]]
            node.transformer = transformation_for_decision
        node.is_numeric = best_is_numeric_split
        node.lc = self.train_tree(X[mask], y[mask])
        node.rc = self.train_tree(X[~mask], y[~mask])
        return node
    
    def evaluate_numeric_attribute(self, col, y, score_to_beat, min_offset = -np.inf, max_offset = np.inf):
        
        indices = np.argsort(col)
        M = []
        ni = np.zeros(len(self.labels))
        
        # idfentify all possible split points
        Nvi = {}
        for k, j in enumerate(indices):
            xj = col[j]
            
            # ignore split point candidates that are outside the range
            #if xj < min_offset or xj > max_offset:
             #   continue
                
            yj = y[j]
            ni[self.labels.index(yj)] += 1
            
            if k < len(col) - 1:
                xjp1 = col[indices[k+1]]
                if xj != xjp1:
                    v = (xj + xjp1) / 2
                    M.append(v)
                    Nvi[v] = ni.copy()
        
        #print(f"Considering {np.round(len(M) / (len(col) - 1), 2)}% of the possible split points.")
        
        # if there are no split points, return -np.inf
        if not M:
            return None, None
        
        # now evaluate different candidates
        best_v, best_score = None, -np.inf
        
        def get_score_of_point(v):
                nY = sum(Nvi[v])
                n = len(col)
                nN = n - nY
                wY = nY / n
                wN = nN / n
                pY = np.array(Nvi[v]) / nY
                pN = (ni - np.array(Nvi[v]))
                pnSum = sum(pN)
                if pnSum > 0:
                    pN /= pnSum
                return self.gain(wY, wN, pY, pN)
            
        def get_best_in_bin(points, num_splits):
            if len(points) < num_splits:
                best_v, best_score = None, -np.inf
                for v in points:
                    score = get_score_of_point(v)
                    if score > best_score:
                        best_v, best_score = v, score
                return best_v, best_score, len(points)
            
            # otherwise get the most promising bin based on a random sample
            bins = np.array_split(points, num_splits)
            best_bin = None
            best_bin_score = -np.inf
            for b in bins:
                sample_point = b[int(len(b)/2)]
                score = get_score_of_point(sample_point)
                if score > best_bin_score:
                    best_bin_score = score
                    best_bin = b
            best_v, best_score, num_evals = get_best_in_bin(best_bin, num_splits)
            return best_v, best_score, num_evals + len(bins)
        
        def get_best_from_beams(points, num_samples, reach, best_sample_point_list):
            
            #if len(points) < min(num_samples * 2 * reach, 100):
             #   best_v, best_score = None, -np.inf
              #  for v in points:
               #     score = get_score_of_point(v)
                #    if score > best_score:
                 #       best_v, best_score = v, score
                #return best_v, best_score, len(points)
            
            # otherwise get the most promising bin based on a random sample
            bins = np.array_split(points, num_samples)
            best_prev_sample_point = best_sample_point_list[0]
            best_bin = best_prev_sample_point[0]
            best_bin_score = best_prev_sample_point[1]
            sample_points = []
            best_bin_index = -1
            for i, b in enumerate(bins):
                sample_point = b[int(len(b)/2)]
                sample_points.append(sample_point)
                score = get_score_of_point(sample_point)
                if score > best_bin_score:
                    best_bin_score = score
                    best_bin = b
                    best_bin_index = i
                    best_sample_point_list[0] = (sample_point, score)
            if best_bin_index < 0:
                return
            lower_index = max(0, best_bin_index - reach)
            upper_index = min(best_bin_index + reach, len(sample_points) - 1)
            new_beam = [p for p in points if p >= sample_points[lower_index] and p <= sample_points[upper_index]]
            if len (new_beam) < 10:
                return
            get_best_from_beams(new_beam, num_samples, reach, best_sample_point_list)
            
            
        num_splits = self.granularity
        if self.light_weight_split_point and len(M) > num_splits:
            #best_v, best_score, num_evals = get_best_in_bin(M, 10)
            
            if False and len(M) >= 1000: #BO approach
                
                # Bounded region of parameter space
                pbounds = {'x': (0, len(M) - 1)}

                optimizer = BayesianOptimization(
                    f=lambda x: get_score_of_point(M[int(x)]),
                    pbounds=pbounds,
                    random_state=self.rs,
                )
                optimizer.maximize(
                    init_points=10,
                    n_iter=10,
                )
                best_v = M[int(optimizer.max["params"]["x"])]
                best_score = optimizer.max["target"]
                return best_v, best_score
                
            
            elif self.beam:
                reach = self.beam
                best = [(None, -np.inf)]
                get_best_from_beams(M, num_splits, reach, best)
                return best[0][0], best[0][1]
            
            elif False:
                return tuple(get_best_in_bin(M, num_splits)[:2])
            
            else:
                
                # create info about splits
                time_start_light = time.time()
                bins = np.array_split(M, num_splits)
                bin_scores = []
                for i, b in enumerate(bins):
                    sample_point = b[int(len(b)/2)]
                    score = get_score_of_point(sample_point)
                    if score > best_score:
                        best_score = score
                        best_v = sample_point
                time_end_light = time.time()
                return best_v, best_score
        
        else:
            scores = []
            time_start_full = time.time()
            for v in M:
                score = get_score_of_point(v)
                scores.append(score)
                if score > best_score:
                    best_v, best_score = v, score
            time_end_full = time.time()
            
            if False:
                
                #print(f"Plotting results. Best val {best_v} with score {best_score} and best solution found within {num_evals} iterations. Best BO solutions: {optimizer.max}. Gap of BO solution: {np.round(np.abs(best_score - optimizer.max['target']), 3)}. Full runtime: {np.round(time_end_bo - time_start_bo, 2)}, BO runtime: {np.round(time_end_bo - time_start_bo, 3)}, Full search runtime: {np.round(time_end_full - time_start_full, 3)}. Runtime light: {np.round(time_end_light - time_start_light, 3)}. Gap light: {np.round(np.abs(best_score - best_mega_greedy_score), 3)}")
                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
                ax1.plot(scores)
                #ax1.scatter(bin_scores[:,0], bin_scores[:,1], color="C1")
                #ax2.plot(M, scores)
                #ax2.scatter(best_v_greedy_search, best_score_greedy_search,  color="C1")
                #ax2.scatter(best_v, best_score, color="C2")
                #ax2.axhline(best_, linestyle="--")
                plt.show()
            
            return best_v, best_score
    

    def evaluate_categorical_attribute(self, col, y):
        
        # compute nvi counters
        dom = list(np.unique(col))
        if len(dom) == 1:
            return dom[0], -np.inf
        Nvi = np.zeros((len(dom), len(self.labels)))
        for i, v in enumerate(dom):
            for j, label in enumerate(self.labels):
                Nvi[i][j] = np.count_nonzero((col == v) & (y == label))
        
        # compute gains
        best_v, best_score = None, -np.log(len(self.labels)) / np.log(2)
        for v_index, v in enumerate(dom):
            sums_Y = Nvi[v_index]
            sums_N = np.sum(Nvi[[i for i in range(len(dom)) if i != v_index]], axis=0)
            nY = sum(sums_Y)
            nN = sum(sums_N)
            pY = sums_Y / nY if nY > 0 else np.zeros(len(self.labels))
            pN = sums_N / nN if nN > 0 else np.zeros(len(self.labels))            
            score = self.gain(nY / len(col), nN / len(col), pY, pN)
            if score > best_score:
                best_v = v
                best_score = score
        return best_v, best_score
    
    def gain(self, wY, wN, pY, pN):
        return -(wY * entropy(pY) + wN * entropy(pN))
    
    def predict_recursively(self, X, node):
        
        if X.shape[0] == 0:
            return []
        
        # get predictions in leaf
        if node.split_point is None:
            return X.shape[0] * [node.label]
        
        # get split point of node
        att, v = node.split_point
        
        # project data correspondingly and compute mask of those points going to the left
        if node.is_numeric:
            if node.transformer is None:
                X_projected = X[:,att]
            else:
                X_transformed = node.transformer.transform(X[:,node.indices_of_attributes_to_project_over])
                X_projected = X_transformed[:,att]
            mask  = X_projected <= v
        else:
            mask = X[att] == v
        
        # get predictions of points going to left
        predictions = np.empty(X.shape[0], object)
        predictions[mask] = self.predict_recursively(X[mask], node.lc)
        predictions[~mask] = self.predict_recursively(X[~mask], node.rc)
        return predictions
    
    def predict(self, X):
        X_scaled = X if self.scaler is None else self.scaler.transform(X)
        return self.predict_recursively(X_scaled, self.model)
    
    def get_depth(self):
        return self.model.get_depth()
    
    def get_number_of_nodes(self):
        return self.model.get_number_of_nodes()
    
class RandomForest:
    
    def __init__(self, n_trees = 100, pi = 0.9, eta = 5, p = None, enable_pca_projections = False, enable_lda_projections = False, lda_on_canonical_projection = False, pca_classes = 2, allow_global_projection = True, project_before_select = False, max_number_of_components_to_consider = None, enforce_projections = False, light_weight_split_point = False, granularity = 10, beam = None, adjust_lda_via_pca = False, rs = None):
        self.n_trees = n_trees
        self.pi = pi
        self.eta = eta
        self.p = p
        self.enable_pca_projections = enable_pca_projections
        self.enable_lda_projections = enable_lda_projections
        self.lda_on_canonical_projection = lda_on_canonical_projection
        self.rs = rs if rs is not None else np.random.RandomState()
        self.project_before_select = project_before_select
        self.pca_classes = pca_classes
        self.allow_global_projection = allow_global_projection
        self.max_number_of_components_to_consider = max_number_of_components_to_consider
        self.enforce_projections = enforce_projections
        self.light_weight_split_point = light_weight_split_point
        self.granularity = granularity
        self.beam = beam
        self.adjust_lda_via_pca = adjust_lda_via_pca
    
    def train(self, X, y):
        self.labels = list(np.unique(y))
        self.trees = []
        num_instances = X.shape[0]
        p = np.sqrt(X.shape[1]) if self.p is None else self.p
        for i in tqdm(range(self.n_trees)):
            indices = self.rs.choice(num_instances, num_instances)
            Xi = np.array([X[j] for j in indices])
            yi = np.array([y[j] for j in indices])
            dt = DecisionTreeClassifier(pi = self.pi, eta = self.eta, p = p, enable_pca_projections = self.enable_pca_projections, enable_lda_projections = self.enable_lda_projections, lda_on_canonical_projection = self.lda_on_canonical_projection, rs = self.rs, project_before_select = self.project_before_select, allow_global_projection = self.allow_global_projection, pca_classes = self.pca_classes, max_number_of_components_to_consider = self.max_number_of_components_to_consider, enforce_projections = self.enforce_projections, light_weight_split_point = self.light_weight_split_point, granularity = self.granularity, beam = self.beam, adjust_lda_via_pca = self.adjust_lda_via_pca)
            dt.train(Xi, yi)
            self.trees.append(dt)
    
    def predict(self, X):
        predictions = []
        opinions = np.array([t.predict(X) for t in self.trees]).T
        for inst in opinions:
            prediction = None
            maxcnt = 0
            for label in self.labels:
                cnt = np.count_nonzero(inst == label)
                if cnt > maxcnt:
                    maxcnt = cnt
                    prediction = label
            predictions.append(prediction)
        return predictions
    
    def get_depths(self):
        return [t.get_depth() for t in self.trees]
    
    def get_numbers_of_nodes(self):
        return [t.get_number_of_nodes() for t in self.trees]