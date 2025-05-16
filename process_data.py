import glob
from pathlib import Path
import numpy as np
import pandas as pd
import re
import download_study as dd
import csv
from typing import List, Dict, Tuple
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statistics

def unify_files(mut_file: str, sv_file: str, patient_file: str, sample_file: str):    

    """
    Read through each file. Remove samples with missing data. Create sample index.
    Order all rows by sample index. Return consistently ordered files.
    """

    # Clinical (patient) file
    head_pat = None
    row_pat = []
    pat_list = set()
    with open(patient_file, 'r') as pf: 
        for line in pf:
            line = line.strip().split('\t')
            if not head_pat and 'PATIENT_ID' in line:
                head_pat = {val.upper(): idx for idx, val in enumerate(line)}
            elif head_pat and len(line) == len(head_pat):
                nan_values = ['', 'NA', 'NaN', 'N/A', 'NULL', 'None', '.']
                os_value = line[head_pat['OS_MONTHS']]
                os_status = line[head_pat['OS_STATUS']]
                # Skip rows with NaN values in OS fields
                if (os_value in nan_values or os_status in nan_values or
                os_value is None or os_status is None):
                    continue
                row_pat.append(line)
                pat_list.add(line[head_pat['PATIENT_ID']])

    # Mut file
    head_mut = None
    row_mut = []
    seen_mut = set()
    mut_samples = set()
    with open(mut_file, 'r') as mf:
        for line in mf:
            line = line.strip().split('\t')
            if not head_mut and 'Tumor_Sample_Barcode' in line:
                head_mut = {val.upper(): idx for idx, val in enumerate(line)}
            elif head_mut and len(line) == len(head_mut):
                line_tuple = tuple(line)
                if line_tuple not in seen_mut:
                    seen_mut.add(line_tuple)
                    row_mut.append(line)
                    mut_samples.add(line[head_mut['TUMOR_SAMPLE_BARCODE']])

    # SV file    
    head_sv = None
    row_sv = []
    seen_sv = set()
    sv_samples = set()
    with open(sv_file, 'r') as sf:
        for line in sf:
            line = line.strip().split('\t')
            if not head_sv and 'Sample_Id' in line:
                head_sv = {val.upper(): idx for idx, val in enumerate(line)}
            elif head_sv and len(line) == len(head_sv):
                line_tuple = tuple(line)
                if line_tuple not in seen_sv:
                    seen_sv.add(line_tuple)
                    row_sv.append(line)
                    sv_samples.add(line[head_sv['SAMPLE_ID']])

    # Unify samples across mut, sv, and clin using sample file mapping
    mut_sv_union = list(mut_samples.union(sv_samples))  
    # Sample_file 
    patient_to_samples = {}
    head_sample = None
    row_sample = []
    with open(sample_file, 'r') as sf:
        for line in sf:
            line = line.strip().split('\t')
            if not head_sample and 'PATIENT_ID' in line:
                # Create header indices
                head_sample = {val.upper(): idx for idx, val in enumerate(line)}    
            elif head_sample and len(line) == len(head_sample):
                row_sample.append(line)
                # Get patient and Sample IDs
                patient_id = line[head_sample['PATIENT_ID']]
                sample_id = line[head_sample['SAMPLE_ID']]
                if patient_id in pat_list and sample_id in mut_sv_union:
                    if patient_id not in patient_to_samples:
                        patient_to_samples[patient_id] = []
                    patient_to_samples[patient_id].append(sample_id)

    # Sort samples for each patient for consistency
    for patient_id in patient_to_samples:
        patient_to_samples[patient_id].sort()

    # Create a new dictionary with sorted patient IDs
    patient_to_samples = dict(sorted(patient_to_samples.items()))

        # Package data into a dictionary
    data_unified = {
        'clinical': {
            'header': head_pat,
            'rows': row_pat,
        },
        'mutation': {
            'header': head_mut,
            'rows': row_mut,
        },
        'sv': {
            'header': head_sv,
            'rows': row_sv,
        },
        'sample': {
            'patient_to_samples': patient_to_samples
        } }
    
    return data_unified

def process_clin(clin_dict: Dict[str, Dict[str, List[str]]]):

    head_pat = clin_dict['clinical']['header']
    row_pat = clin_dict['clinical']['rows']
    sample_dict = clin_dict['sample']['patient_to_samples']

    # Create sample index
    sample_index = {}
    pat_index = {}
    pat_idx = 0
    samp_idx = 0
    for patient, sample_list in sample_dict.items():
        pat_index[patient] = pat_idx
        pat_idx += 1
        for sample in sample_list:
            sample_index[sample] = samp_idx
            samp_idx += 1
 
    os_array = np.zeros((len(sample_index), 2)) # patient survival data
    clin_array = np.zeros((len(sample_index), 2)) # currently sex and smoking status

    # Load column indices
    patient_col_idx = head_pat.get('PATIENT_ID', -1)
    os_status_col_idx = head_pat.get('OS_STATUS', -1)
    os_months_col_idx = head_pat.get('OS_MONTHS', -1)
    sex_col_idx = head_pat.get('SEX', -1)

    # Define mappings from string values to numeric
    os_status_map: Dict[str, int] = {'0:LIVING': 0, '1:DECEASED': 1} 
    # Add other potential status strings if necessary e.g., {'LIVING': 1, 'DECEASED': 0}
    sex_map: Dict[str, float] = {'FEMALE': 0.0, 'MALE': 1.0}

    # Check if crucial columns are missing
    if patient_col_idx == -1:
        print("Warning: 'PATIENT_ID' column not found. Cannot process clinical data.")

    # Processing Rows
    for row in row_pat:
        # Basic check for malformed rows (too short to even contain patient ID)
        if len(row) <= patient_col_idx:
            print(f"Warning: Skipping malformed row (too short): {row}") 
            continue
        # Get id
        patient_id = row[patient_col_idx]
        # Only patient ids with sv and mut data
        if patient_id not in sample_dict:
            continue 
        # Parse OS month
        if os_months_col_idx != -1:
            current_os_months_str = row[os_months_col_idx]
            if current_os_months_str:
                current_os_months_num = float(current_os_months_str)
            else: continue     
            # Parse OS status
            if os_status_col_idx != -1:
                current_os_status_str = row[os_status_col_idx]
                current_os_status_num = os_status_map.get(current_os_status_str, np.nan)
            # Parse Sex Value
            if sex_col_idx != -1:
                current_sex_str = row[sex_col_idx].upper()
                # Look up the sex in the map, default to NaN if not found
                current_sex_num = sex_map.get(current_sex_str, np.nan)
                # For each sample per patient add survial info
            smoking_val = row[head_pat['SMOKING_HISTORY']].upper().strip()
            if smoking_val == 'PREV/CURR SMOKER':
                smoking_status = 1
            else:
                smoking_status = 0
            for sample in sample_dict[patient_id]:
                sample_array_idx  = sample_index[sample]
                os_array[sample_array_idx, 0] = current_os_months_num   
                os_array[sample_array_idx, 1] = current_os_status_num
                clin_array[sample_array_idx, 0] = current_sex_num
                clin_array[sample_array_idx, 1] = smoking_status

    return os_array, clin_array, sample_index

def create_gene_list(clin_dict: Dict[str, Dict[str, List[str]]]) -> Dict[str, int]:

    head_mut = clin_dict['mutation']['header']
    row_mut = clin_dict['mutation']['rows']
    head_sv = clin_dict['sv']['header']
    row_sv = clin_dict['sv']['rows']
    
    # Specify gene columns
    gene_mut = set()
    for row in row_mut:
        gene_id = row[head_mut['HUGO_SYMBOL']]
        gene_mut.add(gene_id)
    gene_sv = set()
    for row in row_sv:
        gene1_id = row[head_sv['SITE1_HUGO_SYMBOL']]
        gene2_id = row[head_sv['SITE2_HUGO_SYMBOL']]
        gene_sv.add(gene1_id)
        gene_sv.add(gene2_id)
    
    combined_genes = list(sorted(gene_mut.union(gene_sv)))

    gene_index = {gene: idx for idx, gene in enumerate(combined_genes)}

    return gene_index

def calc_gene_muts(clin_dict: Dict[str, Dict[str, List[str]]]) -> None:
    
    head_mut = clin_dict['mutation']['header']
    row_mut = clin_dict['mutation']['rows']
    head_sv = clin_dict['sv']['header']
    row_sv = clin_dict['sv']['rows']
     
    # Count sample - gene occurences. Needed to determine maximum number of features/genes
    mut_counts_dict = {}
    for row in row_mut:
        gene = row[head_mut['HUGO_SYMBOL']]
        sample = row[head_mut['TUMOR_SAMPLE_BARCODE']]
        # increment count for the (sample, gene) tuple
        mut_counts_dict[(sample,gene)] = mut_counts_dict.get((sample, gene), 0) + 1 # get the number or 0 - then add 1

    sv_counts_dict = {}
    # Only count if columns have been found
    for row in row_sv:
        gene = row[head_sv['SITE1_HUGO_SYMBOL']]
        sample = row[head_sv['SAMPLE_ID']]
        # increment count for the (sample, gene) tuple
        sv_counts_dict[(sample,gene)] = sv_counts_dict.get((sample, gene), 0) + 1

     # Extract values from dictionaries
    mut_values = list(mut_counts_dict.values())
    sv_values = list(sv_counts_dict.values())
    
    # Safety check for empty data
    max_mut = max(mut_values) if mut_values else 0
    max_sv = max(sv_values) if sv_values else 0
    combined_max = max(max_mut, max_sv) * 1.1  # Add 10% padding

    mean = statistics.mean(mut_values)
    std = statistics.stdev(mut_values)

    one_sd_range = (mean - std, mean + std)
    two_sd_range = (mean - 2*std, mean + 2*std)

    print(f"±1 SD: {one_sd_range}")
    print(f"±2 SD: {two_sd_range}")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot mutations
    sns.kdeplot(mut_values, ax=axes[0], color='blue', fill=True)
    axes[0].set_title('Density Plot of Mutation Counts')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(0, combined_max)

    # Plot structural variants
    sns.kdeplot(sv_values, ax=axes[1], color='orange', fill=True)
    axes[1].set_title('Density Plot of Structural Variant Counts')
    axes[1].set_xlabel('Count')
    axes[1].set_xlim(0, combined_max)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('/home/degan/msk/figures/density_separate_muts_sv.png', dpi=300, bbox_inches='tight')
    plt.show()

def process_mutations(clin_dict: Dict[str, Dict[str, List[str]]], sample_index: Dict[str, int],
                      gene_index: Dict[str, int]):
    
    # maximum number of (sample, gene) pairs. Based on exploratory analysis
    max_muts = 5
    head_mut = clin_dict['mutation']['header']
    row_mut = clin_dict['mutation']['rows']
    # Get unique var types and amino acids
    var_type_set = set() # SNP, INS, DEL
    amino_set = set()  # A P S 
    var_class_set = set() # missense, frame shift, splice
    chromosome_set = set() 
    for row in row_mut:
        # Process Variant_Type
        var_type = row[head_mut['VARIANT_TYPE']]
        if var_type is not None and var_type != '':
            var_type_set.add(var_type)
        # Process HGVSp_Short for amino acids
        hgvsp = row[head_mut['HGVSP_SHORT']]
        if hgvsp is not None and hgvsp != '':
            matches = re.findall(r'[A-Z]|\*|splice|del', str(hgvsp))
            for match in matches:
                amino_set.add(match)
        var_class = row[head_mut['VARIANT_CLASSIFICATION']]
        if var_class is not None and var_class != '':
            var_class_set.add(var_class)
        chr_num = row[head_mut['CHROMOSOME']]
        if chr_num is not None and chr_num != '':
            chromosome_set.add(chr_num)

    # Sorted list before classification
    var_type_list = sorted(list(var_type_set)) 
    unique_aminos = sorted(list(amino_set))
    var_class_list = sorted(list(var_class_set))
    chromosome_list = sorted(list(chromosome_set))

    # Create amino and var type index. 
    amino_index = {amino: i for i, amino in enumerate(unique_aminos)}
    var_type_index = {var: i for i, var in enumerate(var_type_list)}
    var_class_index = {var_class: i for i, var_class in enumerate(var_class_list)} 
    chromosome_index = {chr_num: i for i, chr_num in enumerate(chromosome_list)}
    # Initialize output arrays
    var_type_np = np.zeros((len(sample_index), len(gene_index), len(var_type_list), max_muts))
    aa_sub = np.zeros((len(sample_index), len(gene_index), len(unique_aminos) * 2, max_muts)) # x2 ref and alt
    var_class_np = np.zeros((len(sample_index), len(gene_index), len(var_class_list), max_muts))
    protein_pos = np.zeros((len(sample_index), len(gene_index), max_muts))
    chromosome_np = np.zeros((len(sample_index), len(gene_index), len(chromosome_list), max_muts))

    # create sample gene pairs -> handle together
    # grouped data
    grouped_data = {}  
    for row in row_mut:
        sample = row[head_mut['TUMOR_SAMPLE_BARCODE']]
        gene = row[head_mut['HUGO_SYMBOL']]
        if sample in sample_index and gene in gene_index:
            group_key = (sample, gene)
            grouped_data.setdefault(group_key, []).append(row)

    for (sample, gene), rows_in_group in grouped_data.items():
        i = sample_index[sample]
        j = gene_index[gene]
        for k, row in enumerate(rows_in_group):
             
            if k >= max_muts:
               break # stop storing beyond 5 -> not enough samples
            
            protein_val = row[head_mut['PROTEIN_POSITION']]
            if protein_val != '':
                protein_int = np.log(int(protein_val) + 1)
                protein_pos[i, j, k] = protein_int
            
            amino_acid = row[head_mut['HGVSP_SHORT']]
            aa_match = re.findall(r'([A-Z]|\*|splice|del)', str(amino_acid))
            if aa_match:
                aa_ref = aa_match[0]
                aa_alt = aa_match[-1]
                aa_ref_idx = amino_index[aa_ref]
                # Assign ref aa
                aa_sub[i, j, aa_ref_idx, k] = 1
                aa_alt_idx = amino_index[aa_alt]
                # Assign alt aa
                aa_sub[i, j, len(unique_aminos) + aa_alt_idx, k] = 1

            variant = row[head_mut['VARIANT_TYPE']]
            if variant != '':
                var_idx = var_type_index[variant]
                var_type_np[i, j, var_idx, k] = 1
            
            var_class = row[head_mut['VARIANT_CLASSIFICATION']]
            if var_class != '':
                var_class_idx = var_class_index[var_class]
                var_class_np[i, j, var_class_idx, k] = 1
            
            chr_num = row[head_mut['CHROMOSOME']]
            if chr_num != '':
                chr_idx = chromosome_index[chr_num]
                chromosome_np[i, j, chr_idx, k ] = 1

    # Min-max protien positions    


    return  {'amino_acid': aa_sub, 'protein_pos': protein_pos, 'variant_type_np': var_type_np, 
            'var_class_np': var_class_np,'chromosome_np': chromosome_np}

def process_sv(clin_dict: Dict[str, Dict[str, List[str]]], sample_index: Dict[str, int],
                      gene_index: Dict[str, int]):
    
    max_muts = 5 # Based on exploratory analysis    
    head_sv = clin_dict['sv']['header']
    rows_sv = clin_dict['sv']['rows']
   
   # Create unique variant class and chromosome sets
    class_set = set()
    chrom_set = set()
    region_set = set()
    for row in rows_sv:
        # Var class
        var_class = row[head_sv['CLASS']]
        if var_class is not None and var_class != '':
            class_set.add(var_class)
        # Chromosome number
        chrom1 = row[head_sv['SITE1_CHROMOSOME']]
        chrom2 = row[head_sv['SITE2_CHROMOSOME']]
        chrom_set.add(chrom1)
        chrom_set.add(chrom2)
        # Site Region
        region1 = row[head_sv['SITE1_REGION']]
        region2 = row[head_sv['SITE2_REGION']]
        if region1 and region2 != '':
            region_set.add(region1)
            region_set.add(region2)

    # Sort and List
    class_list = sorted(list(class_set))
    chrom_list = sorted(list(chrom_set))
    region_list = sorted(list(region_set))

    # Create index mappings
    class_index = {var: i for i, var in enumerate(class_list)}
    chrom_index = {chrom: i for i, chrom in enumerate(chrom_list)}
    region_index = {reg: i for i, reg in enumerate(region_list)}

    # Initialize output arrays
    chrom = np.zeros((len(sample_index), len(gene_index), len(chrom_list) * 2, max_muts)) # encode position 1 + 2
    var_class = np.zeros((len(sample_index), len(gene_index), len(class_list), max_muts)) 
    region_sites = np.zeros((len(sample_index), len(gene_index), len(region_list) * 2, max_muts))
    connection_type = np.zeros((len(sample_index), len(gene_index), max_muts))
    sv_length = np.zeros((len(sample_index), len(gene_index), max_muts))
                            
    # Some samples have multiple gene-sample pairs
    grouped_data = {} 
    for row in rows_sv:
        sample = row[head_sv['SAMPLE_ID']]
        gene = row[head_sv['SITE1_HUGO_SYMBOL']]
        if sample in sample_index and gene in gene_index:
            group_key = (sample, gene)
            grouped_data.setdefault(group_key, []).append(row)
    
    # Loop through grouped data
    for (sample,gene), rows_in_group in grouped_data.items():
        
        i = sample_index[sample]
        j = gene_index[gene]

        for k, row in enumerate(rows_in_group):
         
            if k >= max_muts:
                break
            # Add chomosome data
            chrom1 = row[head_sv['SITE1_CHROMOSOME']] # get value 
            chrom2 = row[head_sv['SITE2_CHROMOSOME']]
            chrom1_idx = chrom_index[chrom1] # assign index
            chrom2_idx = chrom_index[chrom2]
            # Update entries
            chrom[i, j, chrom1_idx, k] = 1
            chrom[i, j, len(chrom_list) + chrom2_idx, k] = 1            
            # Add Var class data
            var = row[head_sv['CLASS']]
            var_idx = class_index[var]
            # Update entries
            var_class[i, j, var_idx, k] = 1 
            # region info
            site1 = row[head_sv['SITE1_REGION']]
            site2 = row[head_sv['SITE2_REGION']]
            if site1 and site2 != '':
                site1_idx = region_index[site1]
                site2_idx = region_index[site2]
                region_sites[i, j, site1_idx , k] = 1
                region_sites[i, j, len(region_list) + site2_idx, k] = 1
            # connection type
            connection = row[head_sv['CONNECTION_TYPE']]
            if connection == '5to3':
                connection_val = 1
            elif connection == '3to5':
                connection_val = 0
            else: continue         
            connection_type[i, j, k] = connection_val 
            # sv length
            sv_len_val = row[head_sv['SV_LENGTH']]
            if sv_len_val != '':
                sv_length[i, j, k] = sv_len_val

    return {'chromosome': chrom, 'var_class': var_class, 'region_sites': region_sites,
            'connection_type': connection_type, 'sv_length': sv_length}

def read_files(path):

    # Store outputs
    data_output = {
        "gene_index": None,
        "max_muts": None,
        "patient": None,
        "os_array": None,
        "sample_index": None,
        "mutation": None,
        "sv": None
    }

    # Find all paths containing "data" in the name
    pattern = os.path.join(path, "**", "*data*")
    paths = glob.glob(pattern, recursive=True)
    path_dict = {Path(p).stem: p for p in paths}

    # Define required files
    required_files = [
        "data_mutations",
        "data_sv",
        "data_clinical_sample",
        "data_clinical_patient"
    ]

    # Check if all required files exist
    for file in required_files:
        if file not in path_dict:
            raise FileNotFoundError(f"Required file {file} not found")

    # Assign Files
    mut_file = path_dict["data_mutations"]
    sv_file = path_dict["data_sv"]
    patient_file = path_dict["data_clinical_patient"]
    sample_file = path_dict["data_clinical_sample"]
   
    # Read in files, store as lists, make consistent, remove samples with missing data
    data_unified = unify_files(mut_file, sv_file, patient_file, sample_file)
    os_array, clin_array, samples_index = process_clin(data_unified)
    gene_index = create_gene_list(data_unified)
    calc_gene_muts(data_unified) # output saved in figures -> exploratory
    mutation_dict = process_mutations(data_unified, samples_index, gene_index)
    sv_dict = process_sv(data_unified, samples_index, gene_index)

    data_output["sv"] = sv_dict
    data_output["mutation"] = mutation_dict
    data_output['gene_index'] = gene_index
    data_output['max_muts'] = 5
    data_output['os_array'] = os_array
    data_output['patient'] = clin_array
    data_output['sample_index'] = samples_index

    return data_output

#test = read_files("/home/degan/msk/temp/msk_impact_2017")
    # Store outputs
