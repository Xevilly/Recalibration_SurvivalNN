## Description
Code related to the paper "**Recalibration of Survival Neural Networks in Predicting 10-year Cardiovascular Disease Risk in UK and Chinese Populations**". This repo is a python package for applying two survival neural network(SNN) models to predict 10-year CVD risk for general population. And we also proposed a population-based recalibration method for SNN and misestimation could largely corrected after recalibration when SNN models were applied to a new target population. Here is our research paper.

###  Methods
The SNN is based on the deepsurv and deephit, the original implementation can be found [here](https://github.com/jaredleekatzman/DeepSurv) and [here](https://github.com/chl8856/DeepHit).This code primarily utilizes the following Python packages: [pycox](https://github.com/havakv/pycox)

## File Descriptions
- `run.ipynb`: This Jupyter Notebook serves as an analysis example based on example data and includes the following:
    - Output of original risk using pre-trained model parameters.
    - Calculation of rescaling factors based on the incidence rate of the target population.
    - Computation of recalibrated risk for 10-year cardiovascular disease (CVD).
- `Oct_XX`: A folder containing parameters for models.

## Usage Instructions
1. Clone this repository:
`git clone https://github.com/Xevilly/Recalibration_SurvivalNN.git`  
`cd repository`
2. Install the required dependencies:
3. Run the example:  
    Open `run.ipynb` and follow the instructions provided.
## Notes and contact information

- The code and data in this project are for academic research purposes only and should not be used for commercial purposes.
- If you use this code in your research, please cite our paper in your relevant academic publications.
- Please refer to the LISENSE for more details.
- For any questions or suggestions, feel free to reach out: xiaofeiliu@bjmu.edu.cn
