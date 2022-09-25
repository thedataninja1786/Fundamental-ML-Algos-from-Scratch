class Node():
  def __init__(self, feature_index=None,
                     threshold=None, 
                     left=None, 
                     right=None,
                     info_gain=None,
                     value=None):
    
      self.feature_index = feature_index
      self.threshold = threshold
      self.left = left
      self.right = right
      self.info_gain = info_gain
      self.value = value


class DecisionTree():
  def __init__(self, X_train, y_train, min_samples = 3, max_depth = 3):
    self.root_node = None 
    self.min_samples = min_samples
    self.max_depth = max_depth 
    self.X = X_train.tolist()
    self.Y = y_train.tolist()
  
  @staticmethod
  def _transpose(X):
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
  
  def _gini(self,labels):
    labels = [y[0] for y in labels]
    g = 0 
    for l in list(set(labels)):
      g += (len([y for y in labels if y == l]) / len(labels)) ** 2
    return 1 - g
    
  def _info_gain(self, left_child_y, right_child_y):
    lcw = len(left_child_y) / len(self.Y)
    rcw = len(right_child_y) / len(self.Y)
    return self._gini(self.Y) - \
           ((lcw * self._gini(left_child_y)) + (rcw * self._gini(right_child_y)))

  def _partition(self,x,y,idx,val):
    XT = self._transpose(x)
    left_tree_x = [x[i] for i in range(len(XT[idx])) if XT[idx][i] <= val]
    left_tree_y = [y[i] for i in range(len(XT[idx])) if XT[idx][i] <= val]
    right_tree_x = [x[i] for i in range(len(XT[idx])) if XT[idx][i] > val]
    right_tree_y = [y[i] for i in range(len(XT[idx])) if XT[idx][i] > val]
    return left_tree_x, left_tree_y, right_tree_x, right_tree_y     

  def _find_best_split(self,x,y):
    XT = self._transpose(x)
    split = {}
    max_info = float('-inf')
    for i in range(len(XT)):
      tmp = list(set(XT[i]))
      for v in tmp:
        left_x ,left_y, right_x, right_y = self._partition(x,y,i,v)
        if len(left_y) and len(right_y):
          current_info = self._info_gain(left_y, right_y)
          if current_info > max_info:
            split["feature_index"] = i
            split["value"] = v
            split["left_x"] = left_x 
            split["left_y"] = left_y
            split["right_x"] = right_x 
            split["right_y"] = right_y
            split["info_gain"] = current_info
    return split
  
  def _build_tree(self,x,y,depth):
    if len(x) >= self.min_samples and depth <= self.max_depth:
      split = self._find_best_split(x,y)
      if split["info_gain"]:
        left_subtree = self._build_tree(split["left_x"],split["left_y"],depth + 1)
        right_subtree = self._build_tree(split["right_x"],split["right_y"],depth + 1)
        return Node(split["feature_index"], split["value"],\
                    left_subtree, right_subtree, split["info_gain"])
      
    leaf_label = max(y, key = y.count)
    return Node(value = leaf_label)

  def _predictions(self,x,tree):
    if tree.value != None: return tree.value 
    feature_val = x[tree.feature_index]
    if feature_val <= tree.threshold:
      return self._predictions(x,tree.left)
    else:
      return self._predictions(x, tree.right)

  def fit(self):
    self.root_node = self._build_tree(self.X,self.Y,depth = 1)
    
  def predict(self,X_test):
    predictions = [self._predictions(x, self.root_node) for x in X_test]
    return predictions 
