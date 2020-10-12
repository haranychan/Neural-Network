import numpy as np

class Configuration:

    def __init__(self):

        self.path_out   = "./"

        self.max_epoch  = 10000

        self.mode = "med" #[med, ave]

        """
        self.log_name   = [ 
            "NN_e0025-h5",
            "NN_e005-h5",
            "NN_e01-h5",
            "NN_e02-h5",
            "NN_e03-h5"
        ]

        """
        self.log_name   = [ 
            "NN_e01-h2",
            "NN_e01-h3",
            "NN_e01-h4",
            "NN_e01-h5",
            "NN_e01-h6"
        ]

        self.color      = [
            "blue",
            "red",
            "green",
            "orange",
            "purple"
        ]
