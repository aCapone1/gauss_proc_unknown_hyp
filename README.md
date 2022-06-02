Gaussian Process Uniform Error Bounds with Unknown Hyperparameters for Safety-Critical Applications
===================================================================================================

This code can be used to reproduce the experiments shown in the paper 
"Gaussian Process Uniform Error Bounds with Unknown Hyperparameters for Safety-Critical Applications".


Usage
-------

The best way to run the code is by running either the regression.py or backstepping.py examples using python. When running the regression examples, you will have to choose which data set you want to run by pressing B (Boston house prices), M (Mauna Loa CO2), W (Wine) or S (Sarcos), and pressing enter.
The number of repetitions per regression experiment is set to 10. To change this just edit line 14 in the regression.py file.
To see the mean squared error of the different models, uncomment lines 289-292 in the regression.py file.

License
-------

The code is licenced under the MIT license and free to use by anyone without any restrictions.
