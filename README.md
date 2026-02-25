## Introduction

FTneuralCBF is a toolbox for designing NN-based fault-tolerant control as well as fault-detection and isolation (FDI) mechanisms. For more details, please refer to the paper: 

Garg, K., Dawson, C., Xu, K., Ornik, M., & Fan, C. (2023). Model-free neural fault detection and isolation for safe control. IEEE Control Systems Letters, 7, 3169-3174.

## Installation

git clone https://github.com/kunalgarg42/FTneuralCBF.git

conda create --name [CONDA ENV NAME] python=3.9

conda activate [CONDA ENV NAME]

pip install -r . requirements.txt

## Training

In order to setup the learning, first create a python file for your control-affine systems in the dynamics folder (using control_affine_system_new.py) as the base file.

Then, for CBF + u learning, create a train file following the setup of [Crazyflie_train_new file](https://github.com/kunalgarg42/FTneuralCBF/blob/main/train/Crazyflie_train_new.py). For training FDI, use [CF_train_Gamma_linear_ALL file](https://github.com/kunalgarg42/FTneuralCBF/blob/main/train/CF_train_Gamma_Linear_All.py). 

Finally, for testing the performance of FDI, you can use [CF_test_Gamma_continuous_single.py](https://github.com/kunalgarg42/FTneuralCBF/blob/main/test/CF_test_Gamma_continuous_single.py) as the base file, while for testing the performance of the trained CBFs, use [CF_test_plot_gamma_single.py](https://github.com/kunalgarg42/FTneuralCBF/blob/main/test/CF_test_plot_gamma_single.py)

