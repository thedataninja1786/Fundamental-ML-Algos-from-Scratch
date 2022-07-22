class NaiveBayes():
    def __init__(self,df,Y):
        self.label_name = Y
        self.priors = None 
        self.df = df[:]
        self.Y = df[Y].tolist()
        self.X = df.loc[:, df.columns != Y].values.tolist()
        self.labels = list(set(df[Y]))
        self.posteriors = []

    def calculate_priors(self) -> dict:
        priors = dict()
        for l in self.labels:
            priors[l] = self.Y.count(l) / len(self.Y)
        return priors 
    
    def calculate_posteriors(self,x) -> list:
        self.priors = self.calculate_priors()
        predictions = []
        for l in self.labels:
            tmp_df = self.df[self.df[self.label_name] == l]
            del tmp_df[self.label_name]
            XT = tmp_df.T.values.tolist()
            likelihood = 1
            for i,el in enumerate(x):
                mu = np.mean(XT[i]); sd = np.std(XT[i])
                # pdf of the Gausssian distribution
                likelihood *= np.exp(-(((el-mu)**2)/(2*sd**2))) * (1 / (6.28*sd**2)**0.5)
            posterior = likelihood * self.priors[l]
            predictions.append(posterior)
        return predictions 
    
    def predict(self):
        for x in self.X:
            self.posteriors.append(self.calculate_posteriors(x))
        self.posteriors = [y.index(max(y)) for y in self.posteriors]
        return self.posteriors
