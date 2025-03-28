# QDSR: Quality-Diversity Symbolic Regression  

This repository provides the implementation of **QDSR**, a state-of-the-art **symbolic regression** engine that integrates:  
- **Quality-Diversity (QD) search** for improved exploration,  
- **Dimensional analysis (DA)** for physics-inspired constraints (optional).  

QDSR achieves **high exact recovery rates** on benchmark datasets, surpassing existing methods by a large margin.  

🔗 **Reference:** If you use QDSR in your work, please cite our paper: **[arXiv:2503.19043](https://arxiv.org/abs/2503.19043)**.  

---

## 🚀 Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/jpbruneton/QDSR.git
   cd QDSR

2. Install requirements:
   ```bash
   pip install -r requirements.txt

🔧 Usage
### Running QDSR on Benchmark
To test QDSR on the Feynman-AI dataset from SR-Bench:

1. Download the dataset (with units) from the Feynman-AI website.
2. Unzip it into the following directory:
Targets/Feynman_with_units/raw_data/
There should be 117 csv files (do not forget to unzip bonus equation as well, named test_1... to test_20.csv)
4. Check config/config.py to define metaparameters such as max length, time bufget, verbosity level, etc
3. Run python Launch_Feynman.py

### Running QDSR on Custom Data
To apply QDSR to your own dataset:

1. Modify Launch_from_file.py
2. Set the name and path of your dataset
3. Set the number of variables in your dataset.
4. Define their physical dimensions (if applicable; if not, set them all to zero)
5. Specify the dimension of the target variable (if using DA).
6. Run the script: python Launch_from_file.py 

### Checking Recoverability of a Given Formula
To test whether QDSR can rediscover a known equation:
1. Modify Launch_custom_run.py
2. Set equation you want to use, name the dataset and dimension of variables
3. Make sure your equation is dimensionnaly consistent
4. Run the script: python Launch_custom_run.py

### Customizing Config.py
In config/config.py, you can
1. Change the number of allowed nested functions (set to 2, ie, sin(sin(tan(...))) is not allowed.
2. Change the metric from R2 to NMRSE
3. Change the termination_loss
4. Enforce or not bounds on the free scalars parameters (default False)
5. Modify at will the list of unaries used
6. Directly set the number of CPU by hand by changing def default_n_cores to return n, where n is the desired number
7. Allow or not dimensional analysis
8. Allow or not leaf vocabulary extension as explained in the paper
9. Allow or not sqrt and power nodes
10. Add noise to the data
11. Change the run parameter (number of iterations, etc)
12. Change the maximal length of equations
13. Change the QD grid size
14. Other misc options, like verbosity


# About QDSR
QDSR leverages Quality-Diversity (QD) to improve exploration, allowing for more exact recoveries compared to standard evolutionary methods.

Dimensional Analysis (DA): When enabled through 'apply_dimensional_analysis' flag in config.py, QDSR ensures all expressions are dimensionally consistent, significantly boosting performance on physics-based datasets.

Flexible vocabulary: The engine supports a wide range of functions, including trigonometric, hyperbolic, and inverse functions, for better expression discovery (you can modify the list in config.py).
The engine will also automatically create dimensionless variables and scalar products as explained in the paper via the other flags : 'add_adim_variables', 'add_inverse', 'add_products', and 'add_norms'


📄 License
This project is released under the MIT License.

For questions or contributions, feel free to open an issue or submit a pull request.
