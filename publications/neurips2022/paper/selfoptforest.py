from rotationforest import *


import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
import time

class Benchmark():
    
    def __init__(self, df_learning_curves):
        self.cache = df_learning_curves
    
    def get_curve(self, openmlid, algorithm, seed, oob = True):
        df = self.cache[(self.cache["openmlid"] == openmlid) & (self.cache["seed"] == seed)][["size", "lc_" + algorithm + "_" + ("oob" if oob else "test")]]
        return df.values.T
    
    def get_times(self, openmlid, algorithm, seed):
        df = self.cache[(self.cache["openmlid"] == openmlid) & (self.cache["seed"] == seed)][["size", "train_time_" + algorithm]]
        return df.values.T
    
    def reset(self, openmlid, seed):
        self.openmlid = openmlid
        self.indices = [0, 0, 0]
        self.oob_curves = {}
        self.test_curves = {}
        self.train_times = {}
        self.algos = ["rf", "lda", "pca"]
        for algo in self.algos:
            curves_of_algo = []
            self.oob_curves[algo] = self.get_curve(openmlid, algo, seed, oob = True)[1]
            self.test_curves[algo] = self.get_curve(openmlid, algo, seed, oob = False)[1]
            self.train_times[algo] = self.get_times(openmlid, algo, seed)[1]
    
    def reveal_next_oob(self, algo):
        arm = self.algos.index(algo)
        curve = self.oob_curves[algo]
        if len(curve) == 0:
            return 0
        index = self.indices[arm]
        val = curve[index]
        self.indices[arm] += 1
        return val
    
    def get_test_performance_at_size(self, algo, size):
        return self.oob_curves[algo][size - 1]
    
    def get_oob_performance_at_size(self, algo, size):
        return self.test_curves[algo][size - 1]
    
    def get_train_time(self, algo, trees):
        return self.train_times[algo][trees - 1]
    
    def get_current_train_time(self, algo = None):
        algos = self.algos if algo is None else [algo]
        return sum([self.get_train_time(algo, self.indices[self.algos.index(algo)]) for algo in algos])
    

def get_mmf(sizes, scores):
    def mmf(beta):
        a, b, c, d = tuple(beta.astype(float))
        fun = lambda x: (a * b + c * x ** d)/(b + x ** d)
        penalty = []
        for i, size in enumerate(sizes):
            penalty.append((fun(size) - scores[i])**2)
        return np.array(penalty)

    a, b, c, d = tuple(sp.optimize.least_squares(mmf, np.array([1,1,1,1]), method="lm").x)
    return (a, b, c, d), lambda x: (a * b + c * x ** d)/(b + x ** d)


def get_ipl(sizes, scores):
    def ipl(beta):
        a, b, c = tuple(beta.astype(float))
        pl = lambda x: a - b * x **(-c)
        penalty = []
        for i, size in enumerate(sizes):
            penalty.append((pl(size) - scores[i])**2)
        return np.array(penalty)

    a, b, c = tuple(scipy.optimize.least_squares(ipl, np.array([1,1,1]), method="lm").x)
    return (a, b, c), lambda x: a - b * x **(-c)

def get_exp(sizes, scores):
    def exp(beta):
        a, b, c = tuple(beta.astype(float))
        pl = lambda x: a - b * x **(-c)
        penalty = []
        for i, size in enumerate(sizes):
            penalty.append((pl(size) - scores[i])**2)
        return np.array(penalty)

    a, b, c = tuple(scipy.optimize.least_squares(exp, np.array([1,1,1])).x)
    return (a, b, c), lambda x: 1 - (a * np.exp(-b*x) + c)

def get_last_one(sizes, scores):
    return tuple([scores[-1]]), lambda x: scores[-1]

    
class LearningCurveModel:
    
    def __init__(self, max_trees = 10 ** 10, model_class = "mmf"):
        self.anchors = []
        self.curve = []
        self.curve_smooth = []
        self.forecast_history = []
        self.slopes = []
        self.current_model = None
        self.max_trees = max_trees
        
        self.knowledge = {
            "mmf": {
                "fit_fun": get_mmf,
                "forecast_index": 2,
                "derivative": lambda a, b, c, d, x: (b * (c - a) * d * x**(d-1)) / (x**d + b)**2
            },
            "ipl": {
                "fit_fun": get_ipl,
                "forecast_index": 0,
                "derivative": lambda a, b, c, x: b * c * x**(- c - 1)
            },
            "exp": {
                "fit_fun": get_exp,
                "forecast_index": None,
                "derivative": lambda a, b, c, x: (c / x - b/(x **2))*np.exp(c*np.log(x)+b/x+a)
            },
            "last": {
                "fit_fun": get_last_one,
                "forecast_index": 0,
                "derivative": lambda a, x: 0
            }
        }
        
        if model_class not in ["mmf", "ipl", "last", "exp"]:
            raise Exception(f"Unknown curve class {model_class}. Use 'mmf', 'ipl', 'exp', or 'last'")
        print(f"Projecting with model class {model_class}")
        self.fit_fun = self.knowledge[model_class]["fit_fun"]
        self.forecast_index = self.knowledge[model_class]["forecast_index"]
        self.derivative = self.knowledge[model_class]["derivative"]
        #elif model_class == "ipl":
         #   self.fit_fun = get_ipl
#            self.forecast_index = 0
#        elif model_class == "last":
#            self.fit_fun = get_last_one
#            self.forecast_index = 0
        
    
    def add(self, anchor, val):
        self.anchors.append(anchor)
        self.curve.append(val)
        
        if len(self.curve) >= 2:
            slope_empirical = self.curve[-1] - self.curve[-2]
            
            if len(self.anchors) >= 4:
                
                self.current_model = self.fit_fun(self.anchors, self.curve)
                forecast = self.current_model[1](self.max_trees) if self.forecast_index is None else self.current_model[0][self.forecast_index]
                self.forecast_history.append(forecast)
            
                if len(self.curve) > 5:
                    self.curve_smooth.append(np.mean(self.curve[-5:]))
                    if len(self.curve_smooth) >= 2:
                        self.slopes.append(self.curve_smooth[-1] - self.curve_smooth[-2])
            
          
    def is_stale(self, w = 10, eps = 0.0001):
        
        if len(self.curve_smooth) < w:
            return False
        
        last_slopes = [self.curve_smooth[-i] - self.curve_smooth[-i - 1] for i in range(1, w)]
        slope = np.mean(last_slopes)
        
        if False:
            if self.current_model is None or len(self.curve) < 10:
                return False;
            params, lc = self.current_model
            x = self.anchors[-1]
            slope_analytical = self.derivative(*params, x)
            print(slope_analytical)
            
            
        return slope < eps
    
    def has_stable_forecast(self, w = 10, eps = 0.001):
        if len(self.forecast_history) < w:
            return False
        std_in_forecast = np.std(self.forecast_history[-w:])
        return std_in_forecast < eps
    
    def can_reach_threshold(self, target_anchor, threshold):
        if threshold is None or len(self.slopes) == 0:
            return True
        slope = self.slopes[-1]
        possible = self.curve[-1] + (target_anchor - len(self.curve)) * slope
        if not possible:
            print(self.curve[-1], slope, possible)
            raise Exception()
        return True
        

class SelfOptRF:
    
    def __init__(self, model_class = "mmf"):
        self.models = {}
        self.model_class = model_class
    
    def train_new_tree_and_return_new_oob_accuracy(self, algo):
        self.rfs[algo].step()
        return self.rfs[algo].get_oob_accuracy()
        
    
    def grow_curve(self, reveal_func, algo, threshold, max_size, stop_on_stable_forecast = True, visualize = False, fig = None, ax = None):
        
        if algo in self.models:
            model = self.models[algo]
        else:
            model = LearningCurveModel(model_class = self.model_class, max_trees = max_size)
            self.models[algo] = model
            
        
        if visualize:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 2))
        
        variance_in_forecast = 1
        crit_flattened = False
        while True:
            
            if len(model.curve) >= max_size:
                print("Forest has reach maximum size. Stopping.")
                break
                
            if model.is_stale():
                print("Learning curve is stale. Stopping!")
                break
            
            if stop_on_stable_forecast and model.has_stable_forecast():
                print("We have a stable forecast. Stopping!")
                break
                
            if not model.can_reach_threshold(max_size, threshold): # create linear extension
                break
            
            model.add(len(model.anchors) + 1, reveal_func(algo))
            
            if visualize:
                ax.cla()
                ax.step(model.anchors, model.curve, color="C0", where="post", label=f"Empirical LC of {algo}")
                #ax.axhline(threshold, color="red", linestyle="--")
                if len(model.forecast_history) > 0:
                    ax.axhline(model.forecast_history[-1], color="blue", linestyle="--", label="Current forecast.")

                if len(model.curve) >= 5:
                    domain = np.linspace(0, max_size, 100)
                    ax.plot(domain, model.current_model[-1](domain), color="C1", label="LC Extrapolation")
                    ax.step(range(len(model.curve_smooth)), model.curve_smooth, color="C2", where="post", label="Smoothened LC")
                
                #ax.set_ylim([0,1])
                ax.legend()
                ax.set_title(f"Performance after {len(model.curve)} steps.")
                fig.canvas.draw()
                time.sleep(.5)
                
            

            
        if len(model.curve) < 10:
            raise Exception(f"Curve has only length {len(model.curve)}. Entries: {model.curve}")
    
    
    
    def compute_optimal_forest(self, reveal_func, algos, visualize = False, max_forest_size = 10**6, fig = None, ax = None):
        
        # phase 1: explore to detect best
        curves = {}
        forecast_history = {}
        
        best_seen = 0
        for i, algo in enumerate(algos):
            if visualize:
                if ax is None:
                    fig, ax = plt.subplots()
        
            self.grow_curve(reveal_func, algo, best_seen, max_forest_size, visualize = visualize, fig = fig, ax = ax)
            model = self.models[algo]
            curves[algo] = model.curve_smooth
            forecast_history[algo] = model.forecast_history
            best_seen = max(best_seen, model.curve[-1])
            plt.show()
                        
        # phase 2: explore to detect best
        print("Phase 1 finished. Summary: " + "".join(["\n\t" + f"{algo}: last value {curves[algo][-1]} (at anchor {len(curves[algo])}. Projection: {forecast_history[algo][-1]})" for algo in algos]))
        best_algo_by_history = algos[np.argmax([curves[a][-1] for a in algos])]
        best_algo_by_forecast = algos[np.argmax([forecast_history[a][-1] for a in algos])]
        
        repaired = False
        while best_algo_by_history != best_algo_by_forecast:
            candidates = [best_algo_by_history, best_algo_by_forecast]
            curve_lengths = [len(self.models[algo].curve) for algo in candidates]
            algo_smaller_model = candidates[np.argmin(curve_lengths)]
            steps_to_take = max(curve_lengths) - min(curve_lengths)
            if min(curve_lengths) >= max_forest_size:
                print("Max sized reached.")
                break
            
            if steps_to_take == 0:
                steps_to_take = 1
                #print("Requireing at least one evalution.")
                
            model = self.models[algo_smaller_model]
            for i in range(steps_to_take):
                model.add(len(model.curve) + 1, reveal_func(algo_smaller_model))
            best_algo_by_history = algos[np.argmax([curves[a][-1] for a in algos])]
            best_algo_by_forecast = algos[np.argmax([forecast_history[a][-1] for a in algos])]
            repaired = True
        
        if repaired:
            print("Repair phase finished. Summary: " + "".join(["\n\t" + f"{algo}: last value {curves[algo][-1]} (at anchor {len(curves[algo])}. Projection: {forecast_history[algo][-1]})" for algo in algos]))
        
        
        # fixing the best model
        best_algo = best_algo_by_history
        
        if visualize:
            if ax is None:
                fig, ax = plt.subplots()
        
        self.grow_curve(reveal_func, best_algo, None, max_forest_size,  stop_on_stable_forecast = False, visualize = visualize, fig = fig, ax = ax)
        model = self.models[best_algo]
        while False and len(model.curve) < max_forest_size and np.mean(model.curve[-5:]) < model.forecast_history[-1]:
            model.add(len(model.curve) + 1, reveal_func(best_algo))
            if visualize:
                ax.cla()
                ax.step(model.anchors, model.curve, color="C0", where="post")
                if len(model.forecast_history) > 0:
                    ax.axhline(model.forecast_history[-1], color="blue", linestyle="--")
                
                domain = np.linspace(0, max_forest_size, 100)
                ax.plot(domain, model.current_model[-1](domain), color="C1")
                ax.step(range(len(model.curve_smooth)), model.curve_smooth, color="C2", where="post")
                fig.canvas.draw()
                time.sleep(1)
                ax.set_title(f"Performance after {len(model.curve)} steps.")
        
        print("stopped with curve size", len(model.curve))
        
        self.choice = best_algo
        self.lc_model = model
        if visualize:
            plt.show()
        
        

    def fit(self, X, y, visualize = False, max_forest_size = 10**6, seed = None):
        
        # initialize empty random forests
        self.rfs = {
            "standard": RandomForest(n_trees = max_forest_size, enable_lda_projections = False, enable_pca_projections = False, light_weight_split_point = False, rs = np.random.RandomState(seed)),
            "lda": RandomForest(n_trees = max_forest_size, enforce_projections = True, enable_lda_projections = True, enable_pca_projections = False, lda_on_canonical_projection = True, light_weight_split_point = False, rs = np.random.RandomState(seed)),
            "pca": RandomForest(n_trees = max_forest_size, enforce_projections = True, enable_lda_projections = False, enable_pca_projections = True, pca_classes = 0, light_weight_split_point = False, rs = np.random.RandomState(seed))
        }
        for rf in self.rfs.values():
            rf.setup(X, y)
        
        self.compute_optimal_forest(self.train_new_tree_and_return_new_oob_accuracy, ["standard", "lda", "pca"], max_forest_size = max_forest_size, visualize  = visualize)
        self.prediction_model = self.rfs[self.choice]
    
    def simulate_training_with_benchmark(self, bm, visualize = False, max_forest_size = 10**6):
        algos = ["rf", "lda", "pca"]
        self.compute_optimal_forest(bm.reveal_next_oob, algos, max_forest_size = max_forest_size, visualize = visualize)
        
    def predict(self, X):
        return self.prediction_model.predict(X)