
import numpy as np
import os
import pandas as pd
import config_sum as cf
import matplotlib.pyplot as plt


class Summary:
    def __init__(self, cnf):
        self.cnf = cnf
        self.path_out = self.cnf.path_out + '/fig'
        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)
        
        #plt.rcParams["font.size"] = 14


    def outGraph(self):
        fig = plt.figure(figsize=(12, 5))
        ax  = fig.add_subplot(1,1,1)
        for i in range(len(self.cnf.log_name)):
            path_dat = self.cnf.path_out + '/_result_' + self.cnf.log_name[i] + '/' + '/statistics.csv'
            if os.path.exists(path_dat):
                dat = pd.read_csv(path_dat, index_col = 0)
                if self.cnf.mode == "med":
                    ax.plot(dat.index, dat['med'] , linestyle='solid', color = self.cnf.color[i], label= self.cnf.log_name[i]) 
                    ax.fill_between(dat.index, dat['q25'], dat['q75'], facecolor=self.cnf.color[i], alpha=0.1)
                elif self.cnf.mode == "ave":
                    ax.plot(dat.index, dat['ave'] , linestyle='solid', color = self.cnf.color[i], label= self.cnf.log_name[i]) 
        #ax.set_xlim({0, self.cnf.max_epoch})
        ax.set_xlim({0, self.cnf.max_epoch})
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        #ax.set_title()

        #ax.legend()        
        #ax.legend(["epsilon=0.025", "epsilon=0.05", "epsilon=0.1", "epsilon=0.2", "epsilon=0.3"])
        ax.legend(["hidden=2", "hidden=3", "hidden=4", "hidden=5", "hidden=6"])

        plt.subplots_adjust(left=0.075, right=0.96, bottom=0.11, top=0.94)

        #plt.yscale('log')

        fig.savefig(self.path_out + "/compare_NN_hidden_10000.png" , dpi=150)

if __name__ == '__main__':
    cnf = cf.Configuration()
    smy = Summary(cnf)
    smy.outGraph()
    print("done")
