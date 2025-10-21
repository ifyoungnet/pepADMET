# pepADMET
![GitHub repo size](https://img.shields.io/github/repo-size/ifyoungnet/pepADMET)
![GitHub License](https://img.shields.io/github/license/ifyoungnet/pepADMET)
![GitHub watchers](https://img.shields.io/github/watchers/ifyoungnet/pepADMET?style=social)
![PyPi python](https://img.shields.io/badge/python-3.7.16-green)
![PyPi numpy](https://img.shields.io/badge/numpy-1.21.5-blue)
![PyPi scikitlearn](https://img.shields.io/badge/scikit--learn-1.0.2-blue)
![PyPi torch](https://img.shields.io/badge/torch-1.13.1-blue)

## pepADMET: a novel computational platform for systematic ADMET evaluation of peptides
## Aims
pepADMET is a comprehensive AI-driven platform for predicting absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties of peptides. Designed for pharmacologists and clinicians, it accelerates peptide-based therapeutic development.
## Explanation about the files
The following are the key files and codes used for model training in this project. For more information, please visit: [https://pepadmet.ddai.tech/documentation](https://pepadmet.ddai.tech/documentation). \

**toxicity_early_stop.pth**: pre-trained predictive model that can be directly loaded for toxicity prediction. The model include trained network parameters and necessary configuration details, allowing users to perform inference without retraining. \
**build_dataset.py** : Defines the parameters and data structures for preprocessing Toxicity.csv, including graph representation settings, sample grouping information, and data loading configurations. \
**My_GNN.py**: Contains the implementation of the Graph Neural Network (GNN) model, including architecture definition, parameter settings, and training logic. It defines the forward propagation, loss functions, optimization strategies, and performance evaluation methods for multi-task toxicity prediction. \
**weight_visualization.py**: Visualizes per-atom weights on molecules, highlighting important atoms and bonds, and supports display or image export. \
**build_graph_dataset.py**: Responsible for generating intermediate files such as Toxicity.bin and Toxicity_group.csv, which support efficient data access during training and validation. \
**calculate_descriptors.py**: Used to compute 2,133 molecular descriptors based on example.csv and output the corresponding results in example_feature_result.csv \
**Train.ipynb**: Demonstrates the complete workflow for model building, training, and prediction using the aforementioned files.
___

## Publication
> Xiaorong Tan, Qianhui Liu, Yanpeng Fang, Mengting Zhou, Defang Ouyang, Wenbin Zeng*, Jie Dong*. pepADMET: a novel computational platform for systematic ADMET evaluation of peptides. *Journal of Chemical information and modeling*, submitted.

## Contact
  
  * Prof. Jie Dong: <jiedong@csu.edu.cn> 

