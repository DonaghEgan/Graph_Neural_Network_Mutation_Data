# Graph Isomorphism Neural Network

This repository processes genomic and clinical data from the MSK Immuno 2019 dataset. Using the Reactome database, a gene network is created. This gene network is used to preform message passing using the previously described Graph Isomorphism Network approach (https://arxiv.org/abs/1810.00826). A Cox prorportional hazard loss using patient survival data is used to train the message passing approach. The pipeline handles mutation data, structural variants, clinical features and gene interaction networks to predict patient survival outcomes

## Features
- **Data Processing**: Processes mutation, structural variant, and clinical data into structured arrays for model input.
- **Graph Neural Network**: Implements a GIN-based (Graph Isomorphism Network) model to integrate omics and clinical data with gene interaction networks.
- **Survival Analysis**: Uses Cox partial likelihood loss (Efron approximation) and concordance index for training and evaluation.
- **Scalable Design**: Handles large genomic datasets with utilities for data normalization, harmonization, and batch processing.
- **Reactome Integration**: Incorporates gene interaction networks from Reactome for graph-based learning.

## Dependencies
- Python 3.8+
- PyTorch (`torch>=2.0`)
- PyTorch Geometric (`torch-geometric>=2.3`)
- NumPy (`numpy>=1.21`)
- Pandas (`pandas>=1.5`)
- TQDM (`tqdm>=4.66`) for progress bars
- Glob (`glob`) for file handling

Install dependencies using:
requirements.txt file and install with venv
