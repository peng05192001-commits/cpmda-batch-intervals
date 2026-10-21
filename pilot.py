import numpy as np
class Prep:
    def transform(self,X):
        a=np.column_stack((np.where(np.isnan(X),self.mean,X),np.isnan(X)))
        return (a-self.center)/self.scale
