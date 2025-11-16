import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from badtorch.training.core import train_mlp

if __name__ == "__main__":

    num_trials_per_experiment = 1

    num_epochs = 40
    lr = 1e-2
    batch_size = 64
    in_dims = [784, 256, 128]
    out_dims = [256, 128, 10]
    data_dir = "./data"

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        for f in files:
            os.remove(os.path.join(data_dir, f))

    experiment_settings = [

        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "sgd",
            "dropout": False,
            "dropout_p": 0.0
        },

        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "adam",
            "dropout": False,
            "dropout_p": 0.0
        },

        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "sgd",
            "dropout": True,
            "dropout_p": 0.25
        },
    
        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "adam",
            "dropout": True,
            "dropout_p": 0.25
        },

        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "sgd",
            "dropout": True,
            "dropout_p": 0.1
        },
    
        {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_dims": in_dims,
            "out_dims": out_dims,
            "optimiser_name": "adam",
            "dropout": True,
            "dropout_p": 0.1
        }


    ]

    seed = 0
    for setting in tqdm(experiment_settings, "Experiment Settings"):
        for trial in range(num_trials_per_experiment):
            np.random.seed(seed)
            results = train_mlp(**setting)
            df = pd.DataFrame(results)
            filename = f"{data_dir}/mlp_{setting['optimiser_name']}_{setting['dropout_p']}_{trial}.csv"
            df.to_csv(filename, index=False)
            seed += 1