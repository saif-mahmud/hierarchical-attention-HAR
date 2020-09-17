# Hierarchical Self Attention Based Human Activity Recognition and Novel Activity Detection


* In order to install the dependencies required for this project in the conda environment:

		conda create --name <env> --file requirements.txt

* Command line argument used for referring dataset names in python scripts are given below:

		[dataset_name]: opp / mex / mhealth / pamap2 / realdisp

* As the size of the datasets is beyond the specified maximum size of code and data submission, all the datasets are not included in this repository. These publicly available datasets can be obtained from the UCI repository through the python script dataset_download.py as per the terminal command given below:

		python dataset_download.py [dataset_name]

	- E.g. in order to download raw Opportunity dataset:

			python dataset_download.py opp

* The project contains several packages for various functionality. The project structure is discussed below:
	- `configs/`: contains YAML files for the specification of hyperparameters and metadata for dataset download, preprocessing, model training, and testing.

    - `data/`: contains raw and processed datasets for experiments and human-readable map for activity labels of each dataset.

    - `model/`: contains code for the different modules of the proposed hierarchical self-attention model.

    - `preprocessing/`: contains necessary python scripts for data cleaning and preprocessing.

    - `experiments/`: contains python scripts for different experimental setup and sample visualizations of interpretable feature attention maps and diagrams

    - `saved_models/`: contains trained weights for the hierarchical self-attention model that may be loaded for inference while experimenting with the respective dataset. 

    - `train.py` and `test.py`: python scripts for model training and testing based on given dataset and hyperparameters arguments.

* In order to run experiments, please run the file `test.py` with the following required arguments:
The first argument must be the name of the dataset, options are opp, mex, mhealth, pamap2 and realdisp
Optional arguments are as follows:
    - `save_weights`: to save the trained model weights
    - `include_novelty_exp`: train the VAE on top of the hierarchical attention model and generate results
    - `use_pretrained`: use pre-trained weights to initialize model instead of training
    - Command for running experiments:
    		
		python test.py [dataset_name] [save_weights] [include_novelty_exp] [use_pretrained]
	
* E.g. if we want to run experiments on the Opportunity dataset with saved pre-trained weights and conduct the novelty detection experiment, the command should be:
		
		python test.py opp use_pretrained




