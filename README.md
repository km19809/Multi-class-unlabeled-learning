# Prototype based Multi-Positive and Unlabelled Learning approach
Python implementation for the paper: 

* Amedeo Racanati, Roberto Esposito, Dino Ienco. [Dealing with Multi-Positive Unlabelled learning combining metric learning and deep clustering](https://ieeexplore.ieee.org/document/9773176) in IEEE Access, vol. 10, pp. 51839-51849, 2022, doi: 10.1109/ACCESS.2022.3174590.

## Usage
1. Install the required packages.   
`pip install -r requirements.txt`
2. Clone the code.
3. Prepare datasets. All datasets are already given with the code except for reuters. For this one you have to download the files running the code written in 'data/reuters/get_data.sh'
4. Run the code!   
`python main_experiments.py` if you want to compare the competing methods on a standard setting   
`python ablation_labeled_samples.py` if you want to compare the competing methods with a varying number of labeled samples   
`python ablation_positive_classes.py` if you want to compare the competing methods with a varying number of positive classes   
`python ablation_protoMPUL.py` if you want to study the performances of each stage of ProtoMPUL (ablation study)   

You can customize experiments passing arguments to the interpreter. For example, if you want reproducible results, you have to specify   
`python main_experiments.py --generate_dataset=False`   
in this way the folds generated for each experiment will remain always the same. Inspect the code if you want to get more information 
