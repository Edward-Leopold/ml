import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector, target_vector):
    feature_values = np.array(feature_vector)
    target_labels = np.array(target_vector)
    
    sorted_idx = np.argsort(feature_values)
    sorted_features = feature_values[sorted_idx]
    sorted_targets = target_labels[sorted_idx]
    
    diff = np.diff(sorted_features)
    change_points = np.where(diff > 0)[0]
    
    if len(change_points) == 0:
        return None, None, None, None
    
    thresholds = (sorted_features[change_points] + sorted_features[change_points + 1]) / 2
    
    left_counts = change_points + 1
    right_counts = len(target_vector) - left_counts
    
    left_ones = np.cumsum(sorted_targets)[change_points]
    total_ones = np.sum(sorted_targets)
    right_ones = total_ones - left_ones
    
    left_p = left_ones / left_counts
    right_p = right_ones / right_counts
    
    gini_left = 1 - left_p**2 - (1-left_p)**2
    gini_right = 1 - right_p**2 - (1-right_p)**2
    
    total = len(target_vector)
    gini_scores = -(left_counts/total * gini_left + right_counts/total * gini_right)
    
    best_idx = np.argmax(gini_scores)
    
    return thresholds, gini_scores, thresholds[best_idx], gini_scores[best_idx]

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        for ft in feature_types:
            if ft not in ["real", "categorical"]:
                raise ValueError("неизвестный тип признака")
        
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        best_feature = None
        best_threshold = None
        best_gini = None
        best_split = None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            
            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                ones = Counter(sub_X[sub_y == 1, feature])
                
                ratio = {}
                for key in counts:
                    if key in ones:
                        ones_count = ones[key]
                    else:
                        ones_count = 0
                    ratio[key] = ones_count / counts[key]
                
                sorted_categories = [k for k, v in sorted(ratio.items(), key=lambda x: x[1])]
                mapping = {val: i for i, val in enumerate(sorted_categories)}
                
                feature_vector = np.array([mapping[x] for x in sub_X[:, feature]])
            else:
                raise ValueError
            
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if threshold is None:
                continue
            
            split_mask = feature_vector < threshold
            
            if self._min_samples_leaf is not None:
                left_size = np.sum(split_mask)
                right_size = len(sub_y) - left_size
                if left_size < self._min_samples_leaf or right_size < self._min_samples_leaf:
                    continue
            
            if best_gini is None or gini > best_gini:
                best_gini = gini
                best_feature = feature
                best_split = split_mask
                
                if feature_type == "real":
                    best_threshold = threshold
                elif feature_type == "categorical":
                    best_threshold = [val for val in mapping if mapping[val] < threshold]
        
        if best_feature is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        
        if self._feature_types[best_feature] == "real":
            node["threshold"] = best_threshold
        else:
            node["categories_split"] = best_threshold
        
        node["left_child"] = {}
        node["right_child"] = {}
        
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"], depth + 1)
    
    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        
        if self._feature_types[feature_idx] == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)
        return self
    
    def predict(self, X):
        results = []
        for sample in X:
            results.append(self._predict_node(sample, self._tree))
        return np.array(results)
    
class DecisionTreeWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types):
        self.feature_types = feature_types

    def fit(self, X, y):
        self.tree = DecisionTree(feature_types=self.feature_types)
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        return self.tree.predict(X)