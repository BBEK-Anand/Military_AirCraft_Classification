# Aircraft Classification using PyTorchLabFlow 

Here are the step by step procedure of using PyTorchLabFlow.

## step 1: create the project folder

see last cell of .setup.ipynb file, we have given a project name, AirCraft_Classy and the function setup_project creates the internal file structure inside the folderas the project name, AirCraft_Classy.

## step 2: design components

To design the models or other components,  we used vs code  and Jupyter notebook(`designing.ipynb`) parallely, coding in vs code for different comoponet in `./MyCompDir/`  and using it in `designing.ipynb` just by dictionary representation of the components. 

## Initiating a pipeline

after designing components  and  debuging etc, we  congigured  pipelines  in `./ppl.ipynb` with a unique pplid(`trl*`)

## training and observation

And then trained them in `./Training....ipynb` where per epoch metrics are  printed as we used  tqdm, in runnig loop.

during training we have monitored real time performance of running experiemnts and completed experiements  in `./Observe.ipynb`

## for corupted pipelines

for corupted/ incompatible pipelinhes we  removed teh  sqlite entry for pipelines  in `./Ops.ipynb`,  it also used to stop the training pipeline using `PipeLine.stop_running` 

## Analysis

In `Ana.ipynb` we used filtering,  and  finding  pipelines that shares same components. Also 1st level experiment details and  status  are there


