# Aircraft Classification using PyTorchLabFlow 

Here are the step by step procedure of using PyTorchLabFlow.

## step 1: create the project folder

see last cell of .untitled.ipynb file, we have given a project name, AirCraft_Classy and the function setup_project creates the internal file structure inside the folderas the project name, AirCraft_Classy.

## step 2: design components

here we only focused on model because  other comnpoents like loss, optimizers  are very basic  so we directly written inside files inside ./AirCraft_Classy/Libs/

we started from using entire pretrained efficientnet_b3, resnet50, vgg19 to fine-tuning last 10, 15 and 50 etc layers. all modelloing  are done in `./AirCraft_Classy/Modellings.ipynb` file one after another.

after designing each( after using/running summary function  to see  dimention change throut the architecture) we pasted  them inside ./ AirCraft_Classy/Libs/models.py. after pasting inside .`/AirCraft_Classy/Libs/models.py` we created  and trained experiments in 2 files `./AirCraft_Classy/Training.ipynb` and `./AirCraft_Classy/Libs/models.py` 

## training and observation

during training we have monitored real time performance of running experiemnts and completed experiements  in `./AirCraft_Classy/Observe.ipynb`

## Selecting best

After doing dozens of experiments we got the best* experiment which is `exp12`  and  added  details  of that experiements in `./AirCraft_Classy/Submit.ipynb`. and the final model is resnet50 with last 40 layer fine-tuned.

## extraa

we have also created a basic interface that allows to use weights  of all 12 experiments to predict  for new data/images  in a streamlite  interface `./AirCraft_Classy/streamlit_web.py`