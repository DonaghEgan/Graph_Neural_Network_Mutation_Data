# Graph Isomorphism Neural Network

This repository contains a pipeline for processing genomic and clinical data from cancer studies, such as the MSK Immuno 2019 dataset, and training a Graph Neural Network (GNN) model for survival analysis using a Cox proportional hazards loss. The pipeline handles mutation data, structural variants, clinical features, and gene interaction networks to predict patient survival outcomes.

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
