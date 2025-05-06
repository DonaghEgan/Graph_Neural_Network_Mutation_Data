import glob
from pathlib import Path
import numpy as np
import pandas as pd
import re
import download_study as dd

def normalize_mut_pos(mut_pos, gene_list, genome):
    """
    Normalize mutation positions based on gene coordinates.
    Parameters:
    - mut_pos: 3D NumPy array (samples, genes, max_mutations) with absolute positions.
    - gene_list: List of gene names corresponding to the second dimension.
    - genome: String (e.g., "hg19") specifying the reference genome.
    Returns:
    - 3D NumPy array with normalized positions (0 to 1).
    """   

    gene_length_dict = dd.get_gene_lengths(genome)  # {gene: [start, end]}

    if not isinstance(gene_length_dict, dict):
        raise TypeError("gene_length_dict must be a dictionary")
    missing_values = [gene for gene in gene_list if gene not in gene_length_dict]
    print(f"Genes missing from length dict: {missing_values}")

    # Extract start positions and gene lengths
    start_positions = []
    gene_lengths = []
    for gene in gene_list:
        if gene in gene_length_dict:
            start, end = gene_length_dict[gene]
            start_positions.append(start)
            gene_lengths.append(end - start)

    start_positions = np.array(start_positions)[np.newaxis, :, np.newaxis]  # Shape: (1, genes, 1)
    gene_lengths = np.array(gene_lengths)[np.newaxis, :, np.newaxis]  # Shape: (1, genes, 1)
    
    # Initialize output array
    mut_pos_normalized = np.zeros_like(mut_pos, dtype=np.float32)
    
    # Normalize using broadcasting, preserving zeros
    nonzero_mask = mut_pos != 0
    mut_pos_normalized[nonzero_mask] = (
        (mut_pos[nonzero_mask] - start_positions[0, np.where(nonzero_mask)[1], 0]) /
        gene_lengths[0, np.where(nonzero_mask)[1], 0]
    )
    
    return mut_pos_normalized

def harmonize_dimensions(file1, file2):

    mut_df = pd.read_csv(file1, sep='\t', low_memory=False).drop_duplicates()
    sv_df = pd.read_csv(file2, sep='\t', low_memory=False).drop_duplicates()
    gene_list = sorted(set(mut_df['Hugo_Symbol']) | set(sv_df['Site1_Hugo_Symbol']))

    mut_count = pd.crosstab(
    mut_df['Tumor_Sample_Barcode'], mut_df['Hugo_Symbol']).values
    
    max_mut = mut_count.max()

    sv_count = pd.crosstab(
    sv_df['Sample_Id'], sv_df['Site1_Hugo_Symbol']).values
    max_sv = sv_count.max()

    combined_max =  max(max_sv, max_mut)

    return gene_list, combined_max

def process_clinical(file, patient_to_sample):

    # Read the tab-delimited file
    df = pd.read_csv(file, sep='\t', skiprows=4, low_memory=False).drop_duplicates()

    # Desired order from dictionary
    desired_order = list(patient_to_sample.keys())

    # Make 'patient_id' a categorical type with the specified order
    df['PATIENT_ID'] = pd.Categorical(df['PATIENT_ID'], categories=desired_order, ordered=True)

    # Sort the DataFrame by this new ordered categorical column
    df_sorted = df.sort_values('PATIENT_ID')
    # Format variables
    df_sorted['OS_MONTHS'] = df_sorted['OS_MONTHS'].astype(float)
    df_sorted['SEX'] = df_sorted['SEX'].map({'Male': 0, 'Female': 1})
    df_sorted['OS_STATUS'] = df_sorted['OS_STATUS'].map({'0:LIVING': 0, '1:DECEASED': 1})
    
    # Add dummies
    dummies = pd.get_dummies(df_sorted[['DRUG_TYPE']], dtype=int)
    df_sorted = pd.concat([df_sorted, dummies], axis=1)

    # Drop unwanted columns
    df_features = df_sorted.drop(columns=['PATIENT_ID', 'OS_MONTHS', 'OS_STATUS', 'AGE_GROUP', 'DRUG_TYPE'])
    df_os = df_sorted[['OS_MONTHS', 'OS_STATUS']]

    # Handle NaN values
    if df_features.isna().any().any():
        print("Warning: NaN values detected. Filling with 0.")
        df_features = df_features.fillna(0)

    # Ensure numeric data
    if not df_features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.equals(df_features.columns):
        raise ValueError("Non-numeric columns detected. Please encode all features.")

    # Convert to NumPy array
    patient_array = df_features.to_numpy()  # Shape: [patient_no, feature_no]
    os_array = df_os.to_numpy()

    return patient_array, os_array

def process_sample(file):

    # Read the file
    df = pd.read_csv(file, sep='\t', skiprows=4, low_memory=False).drop_duplicates()
    sample_list = sorted(df['SAMPLE_ID'])

    # Create mapping
    patient_to_samples = {}
    grouped = df.groupby('PATIENT_ID')
    for patient_id, group in grouped:
        # Extract SAMPLE_IDs for this patient
        samples = sorted(group['SAMPLE_ID'].unique())
        patient_to_samples[patient_id] = samples
    return patient_to_samples, sample_list

def process_mutations(file, sample_list, gene_list, max_muts):

    # Read the tab-delimited file directly into a DataFrame
    df = pd.read_csv(file, sep='\t', low_memory=False).drop_duplicates()

    sample_counts = len(set(df['Tumor_Sample_Barcode'].unique()) - set(sample_list))
    print(f"{sample_counts} Samples not in mutation data")

    # Get unique values using Pandas' built-in methods
    var_list = sorted(df['Variant_Type'].unique())
    
    # Extract amino acid patterns
    amino_list = df['HGVSp_Short'].fillna('').str.extractall(r'([A-Z]|\*|splice|del)').iloc[:, 0]
    unique_aminos = sorted(set(amino_list))
    
    # Create index mappings
    sample_index = {sample: i for i, sample in enumerate(sample_list)}
    gene_index = {gene: i for i, gene in enumerate(gene_list)}
    amino_index = {amino: i for i, amino in enumerate(unique_aminos)}
    var_index = {var: i for i, var in enumerate(var_list)}

      # Initialize output arrays
    mut_pos = np.zeros((len(sample_list), len(gene_list), max_muts))
    var_type = np.zeros((len(sample_list), len(gene_list), len(var_list), max_muts))
    aa_sub = np.zeros((len(sample_list), len(gene_list), len(unique_aminos) * 2, max_muts))
    ns = np.zeros((len(sample_list), len(gene_list), max_muts), dtype=int)
    
    # Process mutations using groupby
    grouped = df.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol'])

    for (sample, gene), group in grouped:

        i = sample_index[sample]
        j = gene_index[gene]
        
        # Process each mutation in the group
        for k, (_, row) in enumerate(group.iterrows()):

            if k >= max_muts:
                break
                
            # Add mutation position
            mut_pos[i, j, k] = float(row['Start_Position'])
            
            # Process amino acid sequence
            seq = row['HGVSp_Short']
            if pd.isna(seq): # skip na
                continue

            # find string matches
            matches = re.findall(r'[A-Z]|\*|splice|del', seq)
            if not matches:
                continue

            # add first and last aa -> ref and alt    
            aa_ref = matches[0]
            aa_alt = matches[-1]
            ref_idx = amino_index[aa_ref]
            alt_idx = amino_index[aa_alt]
            
            # index array
            aa_sub[i, j, ref_idx, k] = 1
            aa_sub[i, j, alt_idx + len(unique_aminos), k] = 1  

            var = row['Variant_Type']
            var_idx = var_index[var]
            var_type[i, j, var_idx, k] = 1

            var_class = row['Variant_Classification']
            if var_class != "Silent":
                ns[i, j, k] = 1
    
    # Normalize mut_pos before returning
    mut_pos = normalize_mut_pos(mut_pos, gene_list, genome = 'hg19')

    return mut_pos, var_type, aa_sub, ns

def process_sv(file, sample_list, gene_list, max_muts):
    
    # read df
    df = pd.read_csv(file, sep='\t', low_memory=False).drop_duplicates()
    
    # sample list read in to maintain order
    sample_count = len(set(df['Sample_Id'].unique()) - set(sample_list)) 
    print(f"{sample_count} Samples not in SV data")
     
    # Get unique values using Pandas' built-in methods
    class_list = sorted(df['Class'].unique())
    chrom_list = sorted(np.union1d(df['Site1_Chromosome'].unique(), df['Site2_Chromosome'].unique())) 

    # Create index mappings
    sample_index = {sample: i for i, sample in enumerate(sample_list)}
    gene_index = {gene: i for i, gene in enumerate(gene_list)}
    class_index = {var: i for i, var in enumerate(class_list)}
    chrom_index = {chrom: i for i, chrom in enumerate(chrom_list)}
    
    # Initialize output arrays
    chrom = np.zeros((len(sample_list), len(gene_list), len(chrom_list) * 2, max_muts)) # encode position 1 + 2
    var_class = np.zeros((len(sample_list), len(gene_list), len(class_list) + 2, max_muts))
    
    # Process mutations using groupby
    grouped = df.groupby(['Sample_Id', 'Site1_Hugo_Symbol'])

    for (sample, gene), group in grouped:

        i = sample_index[sample]
        j = gene_index[gene]
    
        for k, (_, row) in enumerate(group.iterrows()):

            if k >= max_muts:
                break

            # process chrom
            chrom1 = row['Site1_Chromosome']
            chrom2 = row['Site2_Chromosome']
            chrom1_idx = chrom_index[chrom1]
            chrom2_idx = chrom_index[chrom2]

            chrom[i, j, chrom1_idx, k] = 1
            chrom[i, j, len(chrom_list) + chrom2_idx, k] = 1

            # var class
            var = row['Class']
            var_idx = class_index[var]

            var_class[i, j, var_idx, k] = 1
            
            # 
            gene1 = row['Site1_Hugo_Symbol']
            gene2 = row['Site2_Hugo_Symbol']

            same_gene = (gene1 == gene2)
            site2_vec = [1,0] if same_gene else [0,1]
            var_class[i, j, len(class_list), k] = site2_vec[0]
            var_class[i, j, len(class_list) + 1, k] = site2_vec[1]
   
    return chrom, var_class

def read_files(path):

    data_output = {
        "gene_list": None,
        "max_muts": None,
        "patient_array": None,
        "os_array": None,
        "patient_to_samples": None,
        "sample_list": None,
        "mutations": None,
        "sv": None
    }

    # Find all paths containing "data" in the name
    paths = glob.glob("**/*data*", recursive=True)
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

    # Step 1: Process data_mutations and data_sv for gene_list and max_muts
    mut_file = path_dict["data_mutations"]
    sv_file = path_dict["data_sv"]
    data_output["gene_list"], data_output["max_muts"] = harmonize_dimensions(mut_file, sv_file)

    # Step 2: Process data_clinical_sample for patient_to_samples and sample_list
    sample_file = path_dict["data_clinical_sample"]
    data_output["patient_to_samples"], data_output["sample_list"] = process_sample(sample_file)

    # Step 3: Process data_clinical_patient for patient_array
    patient_file = path_dict["data_clinical_patient"]
    data_output["patient_array"], data_output['os_array'] = process_clinical(patient_file, data_output["patient_to_samples"])

    # Step 4: Process data_mutations for mutations
    mut_pos, var_type, aa_sub, ns = process_mutations(
        mut_file,
        data_output["sample_list"],
        data_output["gene_list"],
        data_output["max_muts"]
    )
    data_output["mutations"] = {
        "mut_pos": mut_pos,
        "var_type": var_type,
        "aa_sub": aa_sub,
        "ns": ns
    }

    # Step 5: Process data_sv for sv
    chrom, var_class = process_sv(
        sv_file,
        data_output["sample_list"],
        data_output["gene_list"],
        data_output["max_muts"]
    )
    data_output["sv"] = {"chrom": chrom, "var_class": var_class}

    return data_output

