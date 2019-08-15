# CNN
## Program Description
The algorithm implemented in this program is CNN.  
The goal of the network is to recieve a sound file and predict which word is expressed in it (out of a limited set of words).  
The program extracts the sound files and their labels, and treats them as matrices and vectors respectively.  
It continues with defining the model - the layers, the convolution values, the activation function, and the forward function.  
Afterwards, the program executes the training process on the training set, using some built-in functions of PyTorch.  
Then, It makes predictions on the validation set, in order to get an impression of the model's accuracy.  
Finally, the program runs the test, and writes the results to a file.  

## Parameters And Hyper-Parameters
The network has 3 layers. The parameters and the hyper-parameters where chosen after trial and error process.  
At the beginning, the values chosen for the network were extreme (10*10 matrix, 35 filters),  
so they got reduced with time, during the work on the project, and another hidden layer was added.  
Eventually, the values were set as follows:  
Layer1: depth=1, filters=15, matrix=5X5, stride=1, padding=2  
Layer2: depth=15, filters=20, matrix=5X5, stride=1, padding=2  
Layer3: depth=20, filters=32, matrix=5X5, stride=1, padding=2  
Epochs = 5  
etha=0.001  
Activation Function = Relu  
Normalization = torch built-in softmax  
Optimization function = Adam  

## Technical Issues
This project is meant to run on Google Colab.  
Before you begin, open Google Colab and set the Runtime type to GPU.  
For the data set, go to this address: https://github.com/orYakov/data_set.git and follow the intructions in the README file.  
Now upload cnn.py and gcommand_loader.py and run the code that is in cnn.py.
