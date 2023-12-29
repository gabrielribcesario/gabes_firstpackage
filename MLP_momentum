from copy import deepcopy
import numpy as np

class MyLittlePonyM():
  
  class Func():

    def sigmoid(X):
      return (1/(1+np.exp(-X)))

    def der_sigmoid(X):    
      def sigmoid(X):
        return (1/(1+np.exp(-X)))
      sigma = sigmoid(X)
      return (sigma*(1 - sigma))

    def der_tanh(X):
      return np.cosh(X)**-2

    def ReLU(X):
      return (X + abs(X))/2

    def der_ReLU(X):
      return (X > 0) * 1

    def sinc(X):
      return np.sin(X)/X

    def der_sinc(X):
      return (X*np.cos(X) - np.sin(X))/X**2

    def x_squared(X):
      return X**2

    def der_x_squared(X):
      return 2*X

    def lse(x, y):
      return (x - y)**2

    def acc(x, y):
      return ( (x >= .5)*1 != y)*1
      
  def __init__(self, x_shape, n_epocas = 100, learning_rate = .001, neurons_layer = 2, layers = 2, momentum = 0.0001, func_intermed = 'ReLU', metric = 'lse'):
    
    self.ran_epochs = 0
    self.max_epochs = n_epocas
    self.eta = learning_rate
    self.alpha = momentum
    self.erro = []; self.erro_val = []

    self.hiddenLayers = layers
    self.neuronsPerHiddenLayer = neurons_layer
    self.Input = [0]*(layers+1); self.output = [0]*(layers+2)
    self.init_weights(x_shape)

    self.g = getattr(self.Func, func_intermed)
    self.der_g = getattr(self.Func, 'der_' + func_intermed)
    self.metric = getattr(self.Func, metric)

    self.es = False
    self.patience = 0
    self.startEpoch = 0

  def init_weights(self, x_shape):
    self.weights = []
    self.weights.append(np.squeeze(np.random.uniform(-1,1, size = [x_shape[1] + 1, self.neuronsPerHiddenLayer])).T)
    for i in range(self.hiddenLayers - 1):
      self.weights.append(np.squeeze(np.random.uniform(-1,1, size = [self.neuronsPerHiddenLayer + 1, self.neuronsPerHiddenLayer])).T)
    self.weights.append(np.random.uniform(-1,1, size = (self.neuronsPerHiddenLayer + 1,)).T)
    return

  def SetES(self, patience = 50, start = 0):
    self.es = True
    if patience <= 0 or start < 0:
      raise TypeError('Patience ou Start inválidos (Patience <= 0 ou Start < 0).')
    self.patience = patience
    self.startEpoch = start
    return

  def foward(self, X):
    self.output[0] = X
    for i, weights in enumerate(self.weights):
      self.Input[i] = np.dot(weights, np.append(-1, self.output[i]))
      if i != len(self.weights) - 1:
        self.output[i + 1] = self.g(self.Input[i])
      else:
        self.output[i + 1] = self.Func.sigmoid(self.Input[i])
    return


  def backward(self, X, y, prev):
    delta = (y - self.output[-1]) * self.Func.der_sigmoid(self.Input[-1])
    self.weights[-1] = self.weights[-1] + self.eta * delta * np.append(-1, self.output[-2])
    e_k = delta * self.weights[-1][1:]

    for L in range(2, self.hiddenLayers + 2):
      self.weights[-L] = np.add(self.weights[-L],
                                np.dot((self.eta * e_k * self.der_g(self.Input[-L]))[:][np.newaxis].T,
                                       np.append(-1, self.output[-L - 1])[:][np.newaxis])) + self.alpha * (self.weights[-L] - prev[-L])

      e_k = np.dot(self.weights[-L].T[1:], e_k)
    return

  def treino(self, X, y, x_val, y_val):
    peso_ant = [deepcopy(self.weights), deepcopy(self.weights)]
    best_metric = 0; patienceSpent = 0
    for i in range(self.max_epochs):
      erro = 0; erro_val = 0; metric = 0

      for j, k in zip(X, y):
        self.foward(j)
        erro += self.Func.lse(self.output[-1], k)
        self.backward(j, k, peso_ant[-1])
      self.ran_epochs += 1
      self.erro.append(erro)
      peso_ant.pop(); peso_ant.insert(0, deepcopy(self.weights))

      for m,n in zip(x_val, y_val):
        self.foward(m)
        erro_val += self.Func.lse(self.output[-1], n)
        metric += self.metric(self.output[-1], n)
      self.erro_val.append(erro_val)

      if metric < best_metric or i == 0:
        best_metric = metric; melhor_peso = deepcopy(self.weights)

      if self.es and self.startEpoch <= self.ran_epochs:
        if self.erro_val[-1] <= min(self.erro_val):
          patienceSpent = 0
        else:
          patienceSpent += 1
        if patienceSpent == self.patience:
          print('Early Stopping. Ran Epochs: ', self.ran_epochs)
          break
    self.weights = melhor_peso
    return

  def classificador(self, X):
    y = []
    for i in X:
        self.foward(i)
        y.append(self.output[-1])
    return (np.array(y) >= .5)*1
