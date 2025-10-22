# Aircraft Classification using PyTorchLabFlow

This repository is a demonstration of how to use the **PyTorchLabFlow** framework for organizing, managing, and running deep‑learning experiments. The demo applies this framework to a military aircraft classification task.

---

## Workflow Overview

### Step 1: Create the Project Folder

In the last cell of `.setup.ipynb`, you call `setup_project()` with the project name `MilitaryAircraftClassification`. This creates the internal folder structure inside the directory `MilitaryAircraftClassification`.

### Step 2: Design Components

Use VS Code and a Jupyter notebook (`designing.ipynb`) in parallel:

* Write component code (models, data loaders, utilities) inside `./MyCompDir/`.
* In `designing.ipynb`, import and assemble components via dictionary representations for rapid experimentation.

### Step 3: Initiate Pipelines

After components are debugged, configure pipelines in `./ppl.ipynb`. Each pipeline is given a unique ID (e.g., `trl*`). You define how your components are connected, hyper‑parameters, and execution logic.

### Step 4: Training & Observation

* Training is done in `./Training.ipynb`, where per‑epoch metrics are printed (via `tqdm`) inside the training loop.
* Real‑time monitoring and review of running/completed experiments are done in `./Observe.ipynb`.

### Step 5: Handling Corrupted Pipelines

In `./Ops.ipynb`, you can manage problematic pipelines:

* Remove incompatible or corrupted pipeline entries from the SQLite database.
* Stop a running pipeline using `PipeLine.stop_running`.

### Step 6: Analysis

Use `Ana.ipynb` to analyse your experiments:

* Filter pipelines, locate those that share the same components
* View first‑level experiment details and status
* Generate summary comparisons across experiments.

---

## Dataset Directory Instruction

Place the **training split** of the [Kaggle dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset?select=crop) into the designated data directory inside the project.
For example:

```
AirCrafts_data/
    └── Training/             ← insert the training split here
        ├── class1/
        ├── class2/
        └── …
    └── Valid/             ← insert the vlidation split here
        ├── class1/
        ├── class2/
        └── …
```


## Notes

* `./MyCompDir/`: Contains reusable and customizable component source code (models, dataset, loss  and  metric function etc.)
* `./ProjDir/MilitaryAircraftClassification/`: This directory is auto-created by `lab.create_project()` during [Step 1](#step1-create-the-project-folder). It stores:

  * Pipeline definitions
  * Logs & metrics
  * Checkpoints & artifacts
  * Experiment metadata (SQLite database)

* The dataset is **not included** in the repository to keep the repo size minimal. Users are expected to download the dataset separately and place it in the appropriate directory as shown [Dataset Directory Instruction](#dataset-directory-structure).

* Although **PyTorchLabFlow** supports saving model weights at every epoch, this demo retains **only the final epoch's checkpoint** to reduce storage overhead.
