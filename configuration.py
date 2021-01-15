import numpy as np
import os.path
import datetime, shutil
from sklearn.datasets import load_iris

class Configuration:

    def __init__(self):

        # Data setting 
        self.iris       = load_iris()
        self.X          = self.iris.data
        self.N          = self.X.shape[0]

        self.T          = []
        for i in range(self.N):
            if self.iris.target[i] == 0:
                self.T.append([1,0,0])
            elif self.iris.target[i] == 1:
                self.T.append([0,1,0])
            elif self.iris.target[i] == 2:
                self.T.append([0,0,1])
        self.T          = np.array(self.T)


        # Experimental setting 
        self.max_trial  = 30
        self.max_epoch  = 10000
        self.parallel   = True
        self.predict    = False     # True: predict(test)   / False: training

        # NN setting
        self.inp_lay    = self.X.shape[1]
        self.hid_lay    = 2
        self.out_lay    = 3

        self.epsilon    = 0.1
        self.mu         = 0.9      

        # I/O setting
        self.path_out   = "./"
        now = datetime.datetime.now()
        self.log_name   = "_result_" + "NN" +\
            "_" + str(now.year) +\
                "-" + str(now.month) +\
                    "-" + str(now.day) +\
                        "-" + str(now.hour) +\
                            "-" + str(now.minute)



    def setRandomSeed(self, seed=1):
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)



    # out config in txt
    def outSetting(self):
    
        body_setting = "+++++ Experimental Setting +++++\n"
        body_setting += "\n< Environmental Setting >\n"

        # Environment Setting
        item_env = ["trials", "epoch", "input layer", "hidden layer", "output layer", "epsilon", "mu"]
        val_env = [self.max_trial, self.max_epoch, self.inp_lay, self.hid_lay, self.out_lay, self.epsilon, self.mu]

        for i in range(len(item_env)):
            body_setting += item_env[i].ljust(12) + ": " + str(val_env[i]) + "\n"

        path_out = self.path_out + self.log_name + "/"

        # save
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        with open( path_out +"experimental_setting.txt", "w") as f:
            f.write(body_setting)
