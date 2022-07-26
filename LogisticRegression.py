class LogisticRegression():
  def __init__(self,lr:float, n_iters:int):
    self.weights = None 
    self.bias = None 
    self.lr = lr 
    self.n_iters = n_iters

  @staticmethod
  def _accuracy_score(predictions,Y) -> float:
    return sum([1 for pred,y in zip(predictions,Y) if pred == y]) / len(Y)

  def fit(self,X_train,y_train) -> None:
    m = len(X_train)
    # Initialize the weights and bias with random values 
    self.weights = [5 for x in range(len(X_train[0]))]
    self.bias = 0.5 

    # Training loop 
    for _ in range(self.n_iters):
      predictions = []
      for features, label in zip(X_train,y_train):
          derivatives = []
          prediction = 0
          actual = label 
          for weight, feature in zip(self.weights,features):
              prediction += feature * weight
          prediction += self.bias
          sigmoid = 1 / (1 + np.exp(-prediction))
          predictions.append(round(sigmoid))

          for i in range(len(self.weights)):
              # Calculate derivatives
              dw = (1 / m) * (features[i] * (sigmoid - actual)) * self.lr
              derivatives.append(dw)

          for i, derivative in enumerate(derivatives):
              # Update weights
              self.weights[i] -= derivative

          # Calculate bias
          db = (1 / m) * (sigmoid - actual) * self.lr
          # Update bias
          self.bias -= db
    print(f'Accuracy score after {self.n_iters} iterations: {round(self._accuracy_score(y_train,predictions)*100,2)}%.')
  
  def predict(self,X_test) -> list:
    predictions = []
    for features in X_test:
      prediction = 0 
      for weight,feature in zip(self.weights, features):
        prediction += feature * weight
      prediction += self.bias
      sigmoid = 1 / (1 + np.exp(-prediction))
      predictions.append(round(sigmoid))
    return predictions 
