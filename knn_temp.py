import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y
    
  def compute_distances_two_loops(self, X):
    # cham nhat
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
       
        dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
     
    return dists

  def compute_distances_one_loop(self, X):
    # cham nhi
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
  
      dists[i,:] = np.sqrt( np.sum( ( X[i] - self.X_train ) ** 2, axis=1) )
  
    return dists
  
  def compute_distances_no_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
   
    dists = np.reshape(np.sum(X**2, axis=1), [num_test,1]) + np.sum(self.X_train**2, axis=1) \
            - 2 * np.matmul(X, self.X_train.T)
    dists = np.sqrt(dists)
    
    return dists

  def predict_labels(self, dists, k=1):
  
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
     
      closest_y = []
      closest_y = self.y_train[np.argsort(dists[i])][0:k]
     
      y_pred[i] = np.bincount(closest_y).argmax()


    return y_pred
