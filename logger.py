
import os
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt


class Logger:
    def __init__(self, cnf):
        self.dat, self.acc, self.cnf = [], 0, cnf
        self.path_out = cnf.path_out
        self.path_out += '/{0}/'.format(self.cnf.log_name)
        self.path_trial = self.path_out + 'trials' 
        if not os.path.isdir(self.path_trial):
            os.makedirs(self.path_trial)

    def logging(self, epo, error, C, hid_w, out_w):
        self.acc = 0
        for i in range(self.cnf.N):
            if C[i] == self.cnf.iris.target[i]:
                self.acc += 1
        self.acc = self.acc/self.cnf.N
        sls = [epo, error, self.acc]
        sls.extend(hid_w.flatten())
        sls.extend(out_w.flatten())

        self.dat.append(sls)

    def logging_detail(self, epo, Y, C):
        dat_dtl = []
        for i in range(self.cnf.N):
            sls_dtl = [i+1]
            sls_dtl.extend(self.cnf.X[i])
            sls_dtl.extend(Y[i])
            if C[i] == self.cnf.iris.target[i]:
                acc = 1
            else:
                acc = 0
            sls_dtl += [int(C[i]), int(self.cnf.iris.target[i]), acc]
            dat_dtl.append(sls_dtl)

        head = "N,sepal_l,sepal_w,petal_l,petal_w,setosa,versicolor,virginica,predict,answer,true"
        np.savetxt(self.path_trial +'/trial{}_epo{}.csv'.format(self.cnf.seed, epo), np.array(dat_dtl), delimiter=',', header = head)
        print("trial: {:03}\tepoch: {}\taccuracy: {}".format(self.cnf.seed, epo, self.acc))


    def outLog(self):
        head = "epoch,error,accuracy," + ','.join(["hid_w{}".format(i) for i in range(self.cnf.hid_lay*(self.cnf.inp_lay+1))]) + ',' + ','.join(["out_w{}".format(i) for i in range(self.cnf.out_lay*(self.cnf.hid_lay+1))])
        np.savetxt(self.path_trial +'/trial{}.csv'.format(self.cnf.seed), np.array(self.dat), delimiter=',', header = head)      
        self.dat = []
    
class Statistics:
    def __init__(self, cnf, path_out, path_dat):
        self.path_out = path_out
        self.path_dat = path_dat
        self.cnf      = cnf

    def outStatistics(self):
        df = None
        for i in range(self.cnf.max_trial):    
            dat = pd.read_csv(self.path_dat+'/trial{}.csv'.format(i+1), index_col = 0)
            if i == 0:
                df = pd.DataFrame({'trial{}'.format(i+1) : np.array(dat['accuracy'])}, index = dat.index)
            else:
                df['trial{}'.format(i+1)] = np.array(dat['accuracy'])
        df.to_csv(self.path_out + "all_trials.csv")

        _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
        for i in range(len(df.index)):
            dat = np.array(df.loc[df.index[i]])
            res = np.percentile(dat, [25, 50, 75])
            _min.append(dat.min())
            _max.append(dat.max())
            _q25.append(res[0])
            _med.append(res[1])
            _q75.append(res[2])
            _ave.append(dat.mean())
            _std.append(dat.std())

        _out = pd.DataFrame({
            'min' : np.array(_min),
            'q25' : np.array(_q25),
            'med' : np.array(_med),
            'q75' : np.array(_q75),
            'max' : np.array(_max),
            'ave' : np.array(_ave),
            'std' : np.array(_std)
            },index = df.index)
        _out.to_csv(self.path_out + "statistics.csv")

