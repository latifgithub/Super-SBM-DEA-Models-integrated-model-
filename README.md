# Super-SBM-DEA-Models-integrated-model-
A Python implementation of Super Slack-Based Measure (Super-SBM) models for Data Envelopment Analysis (DEA), including traditional and unified formulations.

Super-SBM DEA Models: Python Implementation
 
A Python implementation of Super Slack-Based Measure (Super-SBM) models for Data Envelopment Analysis (DEA), including traditional and unified formulations.
Overview
This package implements three approaches to Super-SBM efficiency measurement:
	Simple-SupSBM: Traditional two-stage approach (SBM → Super-SBM for efficient units)
	Tone-Unified: Unified MILP formulation from Tone et al. (2021)
	New-Unified: Enhanced unified formulation with improved constraint structure
Features
	Multiple Super-SBM model variants
	Efficient identification of frontier DMUs
	Benchmark datasets from DEA literature
	Random data generation for scalability testing
	Comprehensive Excel output with detailed results
	Performance comparison across methods
Requirements
numpy>=1.20.0
scipy>=1.11.0
pandas>=1.3.0
openpyxl>=3.0.0
Installation
                                            content_copy                        bash
pip install numpy scipy pandas openpyxl
Quick Start
                                            content_copy                        python
import numpy as np
from dea_super_sbm import get_data_set, SBM_SSBM_algorithm, NEw_Unified_SBM_SuperSBM_algorithm

# Load benchmark data (Table 1 from Lee et al., 2021)
X, Y = get_data_set(table_num=1, m=2, s=1, n=4)

# Run traditional Super-SBM
eff_scores, efficient_units, super_scores, time_elapsed = SBM_SSBM_algorithm(X, Y, epsilon=1e-6)

# Run unified formulation
scores, n_efficient, time_elapsed = NEw_Unified_SBM_SuperSBM_algorithm(X, Y, epsilon=1e-6)

print(f"Efficiency scores: {super_scores}")
print(f"Efficient DMUs: {efficient_units}")
Usage
Basic Example
                                            content_copy                        python
# Define inputs (m × n) and outputs (s × n)
X = np.array([[2, 2, 3, 1],
              [1, 3, 4, 2]], dtype=float)
Y = np.array([[1, 1, 1, 1]], dtype=float)

# Evaluate all DMUs
scores, efficient_set, super_scores, runtime = SBM_SSBM_algorithm(X, Y, epsilon=1e-6)
Benchmark Datasets
The package includes datasets from Lee et al. (2021):
                                            content_copy                        python
# Available tables: 1, 3, 5, 7, 9, 11
X, Y = get_data_set(table_num=5, m=4, s=2, n=6)
Random Data Generation
                                            content_copy                        python
# Generate random dataset
X, Y = get_data_set(table_num=20, m=8, s=10, n=100)
Batch Processing
                                            content_copy                        python
TABLES = [1, 3, 5, 7, 9, 11]
METHODS = ['Simple-SupSBM', 'Tone-Unified', 'New-Unified']

for table_num in TABLES:
    X, Y = get_data_set(table_num, m=2, s=2, n=2)
    
    # Compare methods
    results = {}
    for method in METHODS:
        if method == 'Simple-SupSBM':
            _, _, scores, time = SBM_SSBM_algorithm(X, Y, 1e-6)
        elif method == 'New-Unified':
            scores, _, time = NEw_Unified_SBM_SuperSBM_algorithm(X, Y, 1e-6)
        
        results[method] = {'scores': scores, 'time': time}
Model Descriptions
Traditional Super-SBM (Simple-SupSBM)
Two-stage approach:
	Stage 1: Identify efficient DMUs using SBM model
	Stage 2: Rank efficient DMUs using Super-SBM model
Advantages: Well-established, numerically stable
Limitations: Requires two optimization problems per DMU
Unified Formulations
Single-stage MILP models that combine efficiency identification and ranking:
min⁡θ-1/m ∑_(i=1)^m  (s_i^-)/x_ip 

subject to unified constraints with binary variable δ∈{0,1}
Advantages: Single optimization per DMU, theoretically elegant
Limitations: Computationally intensive for large problems
Output
The code generates Excel files with:
	Summary sheet: Timing comparison, efficiency counts, maximum deviations
	Method sheets: Detailed scores for each DMU
	Timing comparison: Performance across all datasets
Example output structure:
dea_results_table1.xlsx
├── Summary (timing, speedup, efficiency count)
├── Simple-SupSBM (DMU scores)
├── Tone-Unified (DMU scores)
└── New-Unified (DMU scores)
Configuration
Key parameters in the code:
                                            content_copy                        python
EPSILON = 1e-6          # Efficiency threshold
unified_M_BIG = 1e+4    # Big-M for unified models
SHOW_DETAILS = True     # Display detailed scores
OUTPUT_FILE = 'dea_results.xlsx'
Performance Notes
	Simple-SupSBM: Fastest for small-medium datasets (n < 100)
	Unified models: Suitable for theoretical analysis, slower for large n
	Scalability: Linear programming (Simple-SupSBM) scales better than MILP (unified models)
Typical runtimes (Intel i7, 16GB RAM):
	n=50: Simple-SupSBM ~0.5s, Unified ~2-5s
	n=100: Simple-SupSBM ~2s, Unified ~10-30s
Mathematical Background
The models implement Super-SBM formulations for ranking efficient Decision Making Units (DMUs) in DEA. Key references:
	SBM model: Tone (2001) - Slacks-based measure of efficiency
	Super-SBM: Tone (2002) - Super-efficiency extension
	Unified formulation: Tone, Toloo & Izadikhah (2021) - Single-stage MILP
Citation
If you use this code in your research, please cite:
                                            content_copy                        bibtex
@software{Karimi2026Code,
  author       = {[Your Name]},
  title        = {Python Implementation of Super-SBM DEA Models},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
References
	Lee, H.-S., et al. (2021). “Integrated data envelopment analysis…” European Journal of Operational Research.
	Tone, K., Toloo, M., & Izadikhah, M. (2021). “A modified slacks-based measure of efficiency…” European Journal of Operational Research, 287(2), 560-571.
	Tone, K. (2001). “A slacks-based measure of efficiency in data envelopment analysis.” European Journal of Operational Research, 130(3), 498-509.
License
MIT License - see LICENSE file for details
Contributing
Contributions are welcome! Please:
	Fork the repository
	Create a feature branch
	Add tests for new functionality
	Submit a pull request
Contact
For questions or issues, please open an issue on GitHub or contact [your email].
Acknowledgments
	Benchmark datasets from Lee et al. (2021)
	Unified formulation based on Tone et al. (2021)
	Built with NumPy, SciPy, and Pandas
________________________________________
Note: Replace XXXXXXX in the DOI badge with your actual Zenodo DOI after creating the release.

