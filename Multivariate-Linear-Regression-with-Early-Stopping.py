class LinearRegression():
  def __init__(self,n_iters:int,lr:float):
    self.n_iters = n_iters
    self.lr = lr
    self.weights = None 
    self.bias = None
  
  @staticmethod 
  def _RMSE(Y,predictions) -> float:
    rmse = 0 
    for y,pred in zip(Y,predictions):
      rmse += (y-pred) ** 2
    return (rmse / len(Y)) ** 0.5 
    
  def fit(self,X_train,y_train) -> None:
    actual_iters = 0 
    previous_error = float('inf'); not_improving = 0
    m = len(X_train)
    # Initialize the weights and bias with random values 
    self.weights = [5 for x in range(len(X_train[0]))]
    self.bias = 0.5 

    for iter in range(self.n_iters):
      predictions = []
      for x, y in zip(X_train,y_train):
          gradients = []; prediction = 0
          for weight, x in zip(self.weights,features):
              prediction += x * weight
          prediction += self.bias
          predictions.append(prediction)

          for j in range(len(self.weights)):
              # Calculate derivatives
              dw = (1 / m) * (x[j] * (prediction - actual)) * self.lr
              gradients.append(dw)

          for i, g in enumerate(gradients):
              # Update weights
              self.weights[i] -= g

          # Update bias
          self.bias -= (1 / m) * (prediction - actual) * self.lr

      current_error = self._RMSE(y_train,predictions)
      # Stope early if the model does not improve after the nth consecutive iteration
      if current_error < previous_error:
        previous_error = current_error
        not_improving = 0 
      else:
        not_improving += 1 
        if not_improving > 20: break 
      if iter % 500 == 0: # check on progress 
        print(f'At iteration {iter} the RMSE is {round(self._RMSE(y_train,predictions),2)}.')
    print(self.weights)
    print(f'After {actual_iters} iterations the LM was fitted with a {round(self._RMSE(y_train,predictions),2)} RMSE.')

  def predict(self,X_test,y_test) -> list:
    predictions = []
    for i in range(len(X_test)):
      pred = 0 
      for j in range(len(X_test[i])):
        pred += (self.weights[j] * X_test[i][j]) + self.bias
      predictions.append(pred)
    print(f'RMSE on test data {round(self._RMSE(y_test,predictions),2)}.')
    return predictions 
