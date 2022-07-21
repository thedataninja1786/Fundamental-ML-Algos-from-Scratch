class KNearestNeighbors():
    def __init__(self,k):
        self.k = k 
        self.X_train = None  
        self.classes = []
    
    @staticmethod 
    def euclidean_distance(row1,row2):
      sum = 0
      for i,j in zip(row1,row2):
        sum += (i-j) ** 2 
      distance = sum ** 0.5 
      return distance


    def _predict(self,classes,distances,k):
      from collections import Counter
      # Sort in descending order the distances and retrieve their index 
      # which maps to their respective class 
      idxs = sorted(range(len(distances)),key = lambda x:distances[x])[:self.k] 
      # Find the class for each neighbor  
      neighbors = [self.classes[idx] for idx in idxs]
      # Choose the most ocurring class 
      prediction = Counter(neighbors).most_common(1)
      return prediction[0][0]

    def fit(self,X_train,y_train):
        from sklearn.metrics import accuracy_score
        self.X_train = X_train
        predictions = []
        # For the current entry in the training dataset:
        for i in range(len(X_train)):
            distances = []
            classes = []
            # Estimate the Euclidean distance for the current entry
            # with all other passengers 
            for x,y in zip(X_train,y_train):
                distances.append(self.euclidean_distance(X_train[i],x))
                # Append the class that corresponds to this entry 
                self.classes.append(y)
            # Predict the class for the current entry 
            prediction = self._predict(self.classes,distances,self.k)
            predictions.append(prediction)
        print(f'Utilizing {self.k} groups of nearest-neighbors the training accuracy is at {round(accuracy_score(predictions,y_train)*100,2)}%.')
        return predictions   

    def predict(self,X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = []
            for x in self.X_train:
                distances.append(self.euclidean_distance(X_test[i],x))
            prediction = self._predict(self.classes,distances,self.k)
            predictions.append(prediction)
        return predictions 
