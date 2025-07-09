1. Configure the experimental environment (refer to the picture in the paperexperiments folder for the experimental environment).
2. Open the process folder from the terminal.
3. Open the datahandlers folder, find the _init_.py file, and modify the path of the atlas dataset to the path of your atlas dataset, as follows
4. Open the process folder in the terminal, run python train.py, this step will output a prographer_detector.pth file, this file is the trained model, you can modify the path of the output model file in train.py.
5. Find test_graphs.py and change the path of the model to the output path of your train.py.
6. Continue to run python test_graphs.py in the process folder of the terminal to get the corresponding evaluation data.

