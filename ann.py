import numpy as np
from sklearn.metrics import mean_squared_error
from math import log, sqrt

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(y):
  return y * (1.0 - y)

def tanh(x):
  return np.tanh(0.25 * x)

def dtanh(y):
  return 0.25 * (1.0 - y**2)

class NeuralNet:
  def __init__(self, layers, train_alg='rprop', func='binary', learning_rate=0.7, l1=0.0, l2=0.0):
    self.w = []
    self.train_alg = train_alg
    for i in xrange(len(layers) - 1):
      r = sqrt(6) / sqrt(layers[i] + layers[i + 1] + 1)
      self.w.append(r * (2 * np.random.rand(layers[i] + 1, layers[i + 1]) - 1))
    self.learning_rate = learning_rate
    self.l1 = l1
    self.l2 = l2
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

  @staticmethod
  def from_weights(w, train_alg='rprop', func='binary', learning_rate=0.7):
    layers = tuple([x.shape[0] - 1 for x in w])
    layers = layers + (w[-1].shape[1],)
    nn = NeuralNet(layers, train_alg, func, learning_rate)
    nn.w = w
    return nn

  def update(self, X):
    self.neurons[0] = X
    for i in xrange(len(self.w)):
      self.neurons[i] = np.append(self.neurons[i], np.ones((len(self.neurons[i]), 1)), 1)
      self.neurons[i + 1] = self.func(self.neurons[i].dot(self.w[i]))

  def get_MSE(self):
    return self.mse / 4 if self.func == tanh else self.mse

  def train_epoch(self, X, Y):
    if self.train_alg == 'batch' or self.train_alg == 'rprop':
      dEw = []
      for i in xrange(len(self.w)):
        dEw.append(np.zeros(self.w[i].shape))
    if self.train_alg == 'rprop':
      delta_max = 50.0
      delta_min = 0.0001
      n_pos = 1.2
      n_neg = 0.5
    size = X.shape[0]
    d0 = [[]] * len(self.w)
    if self.train_alg == 'batch' or self.train_alg == 'rprop':
      self.update(X)
      self.mse = mean_squared_error(Y, self.neurons[-1])
      d = d0
      d[-1] = np.atleast_2d((Y - self.neurons[-1]) * self.dfunc(self.neurons[-1]))
      for j in xrange(len(self.w) - 1, 0, -1):
        d[j - 1] = (d[j].dot(self.w[j].T) * self.dfunc(self.neurons[j]))[:, 0:-1]
      for j in xrange(len(self.w)):
        dEw[j] = np.atleast_2d(self.neurons[j]).T.dot(d[j]) - self.l2 * self.w[j] - self.l1 * np.sign(self.w[j])
    else:
      mse_sum = 0
      for i in xrange(size):
        self.update([X[i]])
        mse_sum += mean_squared_error(Y[i], self.neurons[-1][0])
        d = d0
        d[-1] = np.atleast_2d((Y[i] - self.neurons[-1]) * self.dfunc(self.neurons[-1]))
        for j in xrange(len(self.w) - 1, 0, -1):
          d[j - 1] = (d[j].dot(self.w[j].T) * self.dfunc(self.neurons[j]))[:, 0:-1]
        for j in xrange(len(self.w)):
          self.w[j] += self.learning_rate * (np.atleast_2d(self.neurons[j]).T.dot(d[j]) - self.l2 * self.w[j] - self.l1 * np.sign(self.w[j]))
      self.mse = mse_sum / size
    if self.train_alg == 'batch':
      for j in xrange(len(self.w)):
        self.w[j] += self.learning_rate / size * dEw[j]
    if self.train_alg == 'rprop':
      for j in xrange(len(self.w)):
        # iRPROP+
        same_sign = dEw[j] * self.dEw_prev[j] > 0
        opposite_sign = dEw[j] * self.dEw_prev[j] < 0
        zero_sign = dEw[j] * self.dEw_prev[j] == 0
        delta = np.clip(
          same_sign * self.delta_prev[j] * n_pos + \
          opposite_sign * self.delta_prev[j] * n_neg + \
          zero_sign * self.delta_prev[j], delta_min, delta_max)
        delta_w = (same_sign | zero_sign) * np.sign(dEw[j]) * delta + \
          (opposite_sign & (self.mse > self.mse_prev)) * -self.delta_w_prev[j]
        dEw[j][opposite_sign] = 0
        self.w[j] += delta_w
        self.dEw_prev[j] = dEw[j]
        self.delta_prev[j] = delta
        self.delta_w_prev[j] = delta_w
      self.mse_prev = self.mse

  def test(self, X, Y):
    self.update(X)
    self.mse = mean_squared_error(Y, self.neurons[-1])
