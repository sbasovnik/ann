import numpy as np
from sklearn.metrics import mean_squared_error
from math import log

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(y):
  return y * (1.0 - y)

def tanh(x):
  return np.tanh(0.25 * x)

def dtanh(y):
  return 0.25 * (1.0 - y**2)

class NeuralNet:
  def __init__(self, layers, train_alg='rprop', func='binary', learning_rate=0.7):
    self.w = []
    self.train_alg = train_alg
    for i in xrange(len(layers) - 1):
      self.w.append(0.1 * (2 * np.random.rand(layers[i] + 1, layers[i + 1]) - 1))
    self.learning_rate = learning_rate
    if func == 'bipolar':
      self.func = tanh
      self.dfunc = dtanh
    elif func == 'binary':
      self.func = sigmoid
      self.dfunc = dsigmoid
    else:
      raise Exception('Unknown activation function.')
    if train_alg == 'rprop':
      self.dEw_prev = []
      self.delta_prev = []
      self.delta_w_prev = []
      for i in xrange(len(self.w)):
        self.dEw_prev.append(np.zeros(self.w[i].shape))
        self.delta_prev.append(0.0125 * np.ones(self.w[i].shape))
        self.delta_w_prev.append(np.zeros(self.w[i].shape))
      self.mse_prev = 0
    self.neurons = []
    for i in xrange(len(layers) - 1):
      self.neurons.append(np.ones(layers[i] + 1))
    self.neurons.append(np.zeros(layers[-1]))

  def update(self, X):
    self.neurons[0][0:-1] = X
    for i in xrange(len(self.w) - 1):
      self.neurons[i + 1][0:-1] = self.func(self.neurons[i].dot(self.w[i]))
    self.neurons[-1] = self.func(self.neurons[-2].dot(self.w[-1]))

  def get_MSE(self):
    return self.mse / 4 if self.func == tanh else self.mse

  def train_epoch(self, X, Y):
    if self.train_alg == 'backprop':
      return self.train_backprop(X, Y)
    elif self.train_alg == 'backprop_batch':
      return self.train_backprop(X, Y, batch=True)
    elif self.train_alg == 'rprop':
      return self.train_backprop(X, Y, rprop=True)
    else:
      raise Exception('Unknown training algorithm.')

  def train_backprop(self, X, Y, batch=False, rprop=False):
    if batch or rprop:
      dEw_sum = []
      for i in xrange(len(self.w)):
        dEw_sum.append(np.zeros(self.w[i].shape))
    if rprop:
      delta_max = 50.0
      delta_min = 0.0
      n_pos = 1.2
      n_neg = 0.5
    size = X.shape[0]
    mse_sum = 0
    d0 = [[]] * len(self.w)
    for i in xrange(size):
      self.update(X[i])
      mse_sum += mean_squared_error(Y[i], self.neurons[-1])
      d = d0
      d[-1] = np.atleast_2d((Y[i] - self.neurons[-1]) * self.dfunc(self.neurons[-1]))
      for j in xrange(len(self.w) - 1, 0, -1):
        d[j - 1] = (d[j].dot(self.w[j].T) * self.dfunc(self.neurons[j]))[:, 0:-1]
      for j in xrange(len(self.w)):
        dEw = np.atleast_2d(self.neurons[j]).T.dot(d[j])
        if batch or rprop:
          dEw_sum[j] += dEw
        else:
          self.w[j] += self.learning_rate * dEw
    self.mse = mse_sum / size
    if batch:
      for j in xrange(len(self.w)):
        self.w[j] += self.learning_rate * (log(size) / size) * dEw_sum[j]
    if rprop:
      for j in xrange(len(self.w)):
        # iRPROP+
        same_sign = dEw_sum[j] * self.dEw_prev[j] > 0
        opposite_sign = dEw_sum[j] * self.dEw_prev[j] < 0
        zero_sign = dEw_sum[j] * self.dEw_prev[j] == 0
        delta = np.clip(
          same_sign * self.delta_prev[j] * n_pos + \
          opposite_sign * self.delta_prev[j] * n_neg + \
          zero_sign * self.delta_prev[j], delta_min, delta_max)
        delta_w = (same_sign | zero_sign) * np.sign(dEw_sum[j]) * delta + \
          (opposite_sign & (self.mse > self.mse_prev)) * -self.delta_w_prev[j]
        dEw_sum[j] -= opposite_sign * dEw_sum[j]
        self.w[j] = delta_w
        self.dEw_prev[j] = dEw_sum[j]
        self.delta_prev[j] = delta
        self.delta_w_prev[j] = delta_w
      self.mse_prev = self.mse

  def test(self, X, Y):
    size = X.shape[0]
    mse_sum = 0
    for i in xrange(size):
      self.update(X[i])
      mse_sum += mean_squared_error(Y[i], self.neurons[-1])
    self.mse = mse_sum / size
