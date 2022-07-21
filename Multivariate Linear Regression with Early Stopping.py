class LinearRegression():
  def __init__(self,n_iters:int,lr:float):
    self.n_iters = n_iters
    self.lr = lr
    self.X_train = None
    self.y_train = None
    self.weights = None 
    self.bias = 10
  
  @staticmethod 
  def _RMSE(Y,predictions):
    rmse = 0 
    for y,pred in zip(Y,predictions):
      rmse += (y-pred) ** 2
    return (rmse / len(Y)) ** 0.5 
    
  def gradient_descent(self,predictions) -> list:
    new_weights = []
    for j in range(len(self.weights)):
      weight = self.weights[j]
      weight_error = 0; bias_error = 0  
      for i in range(len(self.X_train)):
        weight_error += (-self.X_train[i][j]) * (self.y_train[i] - predictions[i])
        bias_error += (Y[i] - predictions[i])
      weight -= (2 * weight_error / len(self.y_train)) * self.lr
      self.bias -= bias_error * self.lr
      new_weights.append(weight)
    return new_weights

  def fit(self,X_train,y_train) -> None:
    actual_iters = 0 
    previous_error = float('inf'); count = 0
    self.X_train = X_train; self.y_train = y_train 
    # choose a random value for initializing the weights -> 30
    self.weights = [30 for x in range(len(self.X_train[0]))] 
    for _ in range(self.n_iters):
      predictions = []
      actual_iters += 1
      for i in range(len(self.X_train)):
        y_hat = 0 
        for j in range(len(self.weights)):
          y_hat += (self.weights[j] * self.X_train[i][j]) 
        predictions.append(y_hat + self.bias)
      self.weights = self.gradient_descent(predictions)
      current_error = self._RMSE(self.y_train,predictions)
      # early stopping
      if current_error < previous_error:
        previous_error = current_error
      else:
        count += 1 
        if count > 4: break 
    print(f'After {actual_iters} iterations the LM was fitted with a {round(self._RMSE(self.y_train,predictions),2)} RMSE.')

  def predict(self,X_test,y_test) -> list:
    predictions = []
    for i in range(len(X_test)):
      pred = 0 
      for j in range(len(X_test[i])):
        pred += (self.weights[j] * X_test[i][j]) + self.bias
      predictions.append(pred)
    print(f'After {self.n_iters} iterations the LM was fitted with a {round(self._RMSE(y_test,predictions),2)} RMSE.')
    return predictions 
