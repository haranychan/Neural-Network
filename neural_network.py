import numpy                as np
import matplotlib.pyplot    as plt

class NN:

    def __init__(self, cnf, log):
        self.cnf = cnf
        self.log = log
        self.C = np.zeros(self.cnf.N).astype('int')
        self.Y = np.zeros((self.cnf.N, self.cnf.out_lay))    


    def initialization(self):
        self.hid_weight      = self.cnf.rd.rand(self.cnf.hid_lay, self.cnf.inp_lay+1)
        self.out_weight      = self.cnf.rd.rand(self.cnf.out_lay, self.cnf.hid_lay+1)
        self.hid_momentum    = np.zeros((self.cnf.hid_lay, self.cnf.inp_lay+1))
        self.out_momentum    = np.zeros((self.cnf.out_lay, self.cnf.hid_lay+1))

    def training(self):
        self.error = np.zeros(self.cnf.max_epoch)
        for epo in range(self.cnf.max_epoch):
            for i in range(self.cnf.N):
                x = self.cnf.X[i, :]
                t = self.cnf.T[i, :]
                self.__update_weight(x, t)
            self.error[epo] = self.__calc_error()
            self.log.logging(epo+1, self.error[epo], self.C, self.hid_weight, self.out_weight)
            if epo % 100 == 99:
                self.log.logging_detail(epo+1, self.Y, self.C)
        self.log.outLog()

    def predict(self):
        for i in range(self.cnf.N):
            x = self.cnf.X[i, :]
            z, y = self.__forward(x)
            self.Y[i] = y
            self.C[i] = y.argmax()

    def out_errorgraph(self):
        plt.xlim({0, self.cnf.max_epoch})
        plt.plot(np.arange(0, self.error.shape[0]), self.error)
        plt.show()



    def __sigmoid(self, arr):
        return np.vectorize(lambda x: 1.0 / (1.0 + np.exp(-x)))(arr)

    def __forward(self, x):
        # z: output in hidden layer
        # y: output in output layer
        z = self.__sigmoid(self.hid_weight.dot(np.r_[np.array([1]), x]))
        y = self.__sigmoid(self.out_weight.dot(np.r_[np.array([1]), z]))
        return (z, y)

    def __update_weight(self, x, t):
        z, y = self.__forward(x)

        # update output_weight
        out_delta = (y - t) * y * (1.0 - y)
        _out_weight = self.out_weight
        self.out_weight -= self.cnf.epsilon * out_delta.reshape((-1, 1)) * np.r_[np.array([1]), z] - self.cnf.mu * self.out_momentum
        self.out_momentum = self.out_weight - _out_weight

        # update hidden_weight
        hid_delta = (self.out_weight[:, 1:].T.dot(out_delta)) * z * (1.0 - z)
        _hid_weight = self.hid_weight
        self.hid_weight -= self.cnf.epsilon * hid_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        self.hid_momentum = self.hid_weight - _hid_weight

    def __calc_error(self):
        err = 0.0
        for i in range(self.cnf.N):
            x = self.cnf.X[i, :]
            t = self.cnf.T[i, :]
            z, y = self.__forward(x)
            self.Y[i] = y
            self.C[i] = y.argmax()
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0
        return err