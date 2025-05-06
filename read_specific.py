import os
import re
import torch
from torch_geometric.data import download_url, extract_zip, extract_gz
import numpy as np
from torch_geometric.data import Data
import csv
import read_folders
import download_study
import zipfile 
from typing import Optional, List  

def read_reactome_new(tokens: Optional[List[str]] = None, folder: str = 'temp/', url: str = 'https://reactome.org/download/tools/ReatomeFIs/FIsInGene_122921_with_annotations.txt.zip') -> torch.Tensor:
    """
    Function to read Reactome file and extract gene regulatory and protein-protein interaction networks.

    :param tokens: A list of gene symbols to filter the network. If None, all genes are included.
    :param folder: Directory where the file will be downloaded and extracted.
    :return: A dictionary with:
        - tokens: List of gene symbols.
    """
    # test url
    if url is None or not isinstance(url, str):
        raise ValueError('url must be provided, and in string format')

    # download
    try:
        path = download_url(url, folder) 
    except Exception as e:
        raise RuntimeError(f"Failed to download or extract file: {e}")
    
    # if it is a zip file extract
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder)
    
    # takes first filename found
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            extracted_path = os.path.join(folder, filename)
            break
    else:
        raise FileNotFoundError("No .txt file found in the folder.")

    # Read file and store parsed lines (exlcuding header)
    gene_index = {gene: i for i, gene in enumerate(tokens)}
    interaction_pairs = []

    with open(extracted_path, 'r') as fo:
        for i, line in enumerate(fo):
            if i == 0:
                continue
            parts = line.rstrip('\n').split('\t')
            interaction_pairs.append([parts[0], parts[1]]) 
 
    # Initialize adjacency matrix
    adj_matrix = torch.zeros((len(tokens), len(tokens)), dtype = torch.float32)
    for gene_a, gene_b in interaction_pairs:
        if gene_a in gene_index and gene_b in gene_index:
             idx_a = gene_index[gene_a]    
             idx_b = gene_index[gene_b]
             adj_matrix[idx_a, idx_b] = 1
             adj_matrix[idx_b, idx_a] = 1

    # Add genes in gene_list -> 0s if not in reactome
    for gene in tokens:
        idx = gene_index[gene]
        adj_matrix[idx, idx] = 0
    
    return adj_matrix

def read_cn_depmap(genelist = None, depmaps = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return: A numpy array with the copy number data.
    """
    url = 'https://ndownloader.figshare.com/files/34989937'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    gene_columns = re.split(',', lineList[0])
    gene_indexes = list()
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    if genelist == None:
        genelist = gene_columns.copy()
        genelist.pop(0)
        gene_indexes = list()
        for i in range(len(genelist)):
            gene_indexes.append(i+1)
    else:
        for i in range(len(genelist)):
            if genelist[i] != '':
                if gene_columns.count(genelist[i]) > 0:
                    gene_indexes.append(gene_columns.index(genelist[i]))
                else:
                    gene_indexes.append(None)
            else:
                gene_indexes.append(None)
    if depmaps == None:
        n = len(lineList) - 1
        depmaps = list()
        # pre-allocate memory for cn
        cn_dep = np.zeros((n, len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            depmaps.append(div[0])
            for j in range(len(genelist)):
                if gene_indexes[j] != None:
                    if div[gene_indexes[j]] != '':
                        cn_dep[i-1, j] = float(div[gene_indexes[j]])
    else:
        # pre-allocate memory for cn
        cn_dep = np.zeros((len(depmaps), len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[0]) > 0:
                idx = depmaps.index(div[0])
                for j in range(len(genelist)):
                    if gene_indexes[j] != None:
                        if div[gene_indexes[j]] != '':
                            cn_dep[idx, j] = float(div[gene_indexes[j]])
    out_dict = dict()
    out_dict['cna'] = cn_dep
    out_dict['sample_ids'] = depmaps
    out_dict['geneList'] = genelist
    return (out_dict)


def read_exp_depmap(genelist = None, depmaps = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return:
            A tupple with the following elements:
            A numpy array with the expression data.
            A list with the depmap ids for each row in the array.
    """
    url = 'https://ndownloader.figshare.com/files/34989919'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    gene_columns = re.split(',', lineList[0])
    gene_indexes = list()
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    if genelist == None:
        genelist = gene_columns.copy()
        genelist.pop(0)
        gene_indexes = list()
        for i in range(len(genelist)):
            gene_indexes.append(i+1)
    else:
        for i in range(len(genelist)):
            if genelist[i] != '':
                if gene_columns.count(genelist[i]) > 0:
                    gene_indexes.append(gene_columns.index(genelist[i]))
                else:
                    gene_indexes.append(None)
            else:
                gene_indexes.append(None)
    if depmaps == None:
        n = len(lineList) - 1
        depmaps = list()
        # pre-allocate memory for cn
        cn_dep = np.zeros((n, len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            depmaps.append(div[0])
            for j in range(len(genelist)):
                if gene_indexes[j] != None:
                    if div[gene_indexes[j]] != '':
                        cn_dep[i-1, j] = float(div[gene_indexes[j]])
    else:
        # pre-allocate memory for cn
        cn_dep = np.zeros((len(depmaps), len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[0]) > 0:
                idx = depmaps.index(div[0])
                for j in range(len(genelist)):
                    if gene_indexes[j] != None:
                        if div[gene_indexes[j]] != '':
                            cn_dep[idx, j] = float(div[gene_indexes[j]])
    out_dict = dict()
    out_dict['exp'] = cn_dep
    out_dict['sample_ids'] = depmaps
    out_dict['geneList'] = genelist
    return (out_dict)

def read_fusions_depmap(depmaps = None, geneList = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return:
            fusion:  numpy array with the one-hote encoded data of whether a gene is fused or not.
    """
    url = 'https://ndownloader.figshare.com/files/34989931'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    cols = re.split(',', lineList[0])
    depmap_id_col = cols.index('DepMap_ID')
    gene_col = cols.index('FusionName')
    gene_left = cols.index('LeftGene')
    gene_right = cols.index('RightGene')
    chrom_list = ['1', '2', '3', '4', '5', '6', '7','8' , '9', '10', '11', '12', '13', '14' , '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
    pos_left = cols.index('LeftBreakpoint')
    pos_right = cols.index('RightBreakpoint')
    if depmaps == None:
        depmaps = list()
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[depmap_id_col]) == 0:
                depmaps.append(div[depmap_id_col])
    if geneList == None:
        geneList = list()
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            g1 = div[gene_left]
            parenthesis = re.search(' ', g1)
            g1 = g1[0:parenthesis.start(0)]
            g2 = div[gene_right]
            parenthesis = re.search(' ', g2)
            g2 = g2[0:parenthesis.start(0)]
            if geneList.count(g1) ==0:
                geneList.append(g1)
            if geneList.count(g2) == 0:
                geneList.append(g2)
    pos = np.zeros((len(depmaps), len(geneList)))
    chrom = np.zeros((len(depmaps), 48))
    fusions = np.zeros((len(depmaps), len(geneList)))  # store fusions.
    for i in range(1, len(lineList)):
        # separate the columns
        div = re.split(',', lineList[i])
        # read Depmap ID
        depmap_i = div[depmap_id_col]
        gene_f = div[gene_col]
        gene_1, gene_2  = re.split('--', gene_f)
        chrom_left, p_left, strand_left =  re.split(':', div[pos_left])
        chrom_right, p_right, strand_right =  re.split(':', div[pos_right])
        if depmaps.count(depmap_i) > 0:
            I = depmaps.index(depmap_i)
            if geneList.count(gene_1)>0:
                J1 = geneList.index(gene_1)
                fusions[I,J1] = 1.
                chrom_left = re.split('r', chrom_left)[1]
                if chrom_list.count(chrom_left)>0:
                    chrom[I, chrom_list.index(chrom_left)] = 1
                pos[I, J1] = float(p_left)
            if geneList.count(gene_2)>0:
                J2 = geneList.index(gene_2)
                fusions[I,J2] = 1.
                chrom_right = re.split('r', chrom_right)[1]
                if chrom_list.count(chrom_right)>0:
                    chrom[I, chrom_list.index(chrom_right)+23] = 1
                pos[I, J2] = float(p_right)
    out_dict = dict()
    out_dict['sv_pos'] = pos
    out_dict['chrom'] = chrom
    out_dict['oh_fusions'] = fusions
    out_dict['sample_ids'] = depmaps
    out_dict['geneList'] = geneList
    return (out_dict)

def read_mut_depmap(depmaps = None, geneList = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return:
            A tupple with the following elements:
            pos_muts: A numpy array with the mutation position in the genome.
            vtp: A numpy array with the type of mutations with a 1 or 0 with the columns ordered as 'SNP', 'DNP',
                 'INS', 'DEL'and 'TNP'.
            subs: A numpy array with the base substitution of mutations with a 1 or 0 with the columns ordered as
                  'T', 'C', 'G' and 'A'. The first 4 columns correspond to the baseline base and the second 4
                  to the substitution.
            ns: A numpy array with the one-hote encoded data of whether a mutation is non-silent or not.
            pos_muts, vtp, bases, ns
    """
    url = 'https://ndownloader.figshare.com/files/34989940'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    cols = re.split(',', lineList[0])
    depmap_id_col = cols.index('DepMap_ID')
    sp_col = cols.index('Start_position')
    vt_col = cols.index('Variant_Type')
    vc_col = cols.index('Variant_Classification')
    ra_col = cols.index('Reference_Allele')
    aa_col = cols.index('Alternate_Allele')
    if depmaps == None:
        depmaps = list()
        for i in range(1,len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[depmap_id_col])==0:
                depmaps.append(div[depmap_id_col])
    if geneList == None:
        geneList = list()
        for i in range(1,len(lineList)):
            div = re.split(',', lineList[i])
            if geneList.count(div[0])==0:
                geneList.append(div[0])
    pos_muts = np.zeros((len(depmaps), len(geneList)))  # store mutation position.
    vtp = np.zeros((len(depmaps), len(geneList), 5))  # variant type.
    subs = np.zeros((len(depmaps), len(geneList), 8))
    ns = np.zeros((len(depmaps), len(geneList)))
    vts = ['SNP', 'DNP', 'INS', 'DEL', 'TNP']
    bases = ['T', 'C', 'G', 'A']
    for i in range(1, len(lineList)):
        # separate the columns.
        div = re.split(',', lineList[i])
        # read Depmap ID
        depmap_i = div[depmap_id_col]
        gene_i = div[0]
        if depmaps.count(depmap_i) > 0 and geneList.count(gene_i) > 0:
            I = depmaps.index(depmap_i)
            J = geneList.index(gene_i)
            pos_muts[I, J] = float(div[sp_col])  # store start position.
            if vts.count(div[vt_col]) > 0:
                vtp[I, J, vts.index(div[vt_col])] = 1.
            if bases.count(div[ra_col]) > 0:
                subs[I, J, bases.index(div[ra_col])] = 1.
            if bases.count(div[aa_col]) > 0:
                subs[I, J, bases.index(div[aa_col]) + 4] = 1.
            if div[vc_col] != 'Silent':
                ns[I, J] = 1.
    out_dict = dict()
    out_dict['pos'] = pos_muts
    out_dict['vtp'] = vtp
    out_dict['subs'] = subs
    out_dict['ns'] = ns
    out_dict['sample_ids'] = depmaps
    out_dict['geneList']  = geneList
    return (out_dict)


def read_mut_nd_depmap(depmaps = None, genelist = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return:
            A tupple with the following elements:
            nd: A numpy array with entries of 1 if a gene has a dammaging mutation. 
            depmaps: A list with the cell-lines Ids.
            geneList: A list with the gene Symbols
    """
    url = 'https://ndownloader.figshare.com/files/38357498'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    gene_columns = re.split(',', lineList[0])
    gene_indexes = list()
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    if genelist == None:
        genelist = gene_columns.copy()
        genelist.pop(0)
        gene_indexes = list()
        for i in range(len(genelist)):
            gene_indexes.append(i+1)
    else:
        for i in range(len(genelist)):
            if genelist[i] != '':
                if gene_columns.count(genelist[i]) > 0:
                    gene_indexes.append(gene_columns.index(genelist[i]))
                else:
                    gene_indexes.append(None)
            else:
                gene_indexes.append(None)
    if depmaps == None:
        n = len(lineList) - 1
        depmaps = list()
        # pre-allocate memory for cn
        cn_dep = np.zeros((n, len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            depmaps.append(div[0])
            for j in range(len(genelist)):
                if gene_indexes[j] != None:
                    if div[gene_indexes[j]] != '':
                        cn_dep[i-1, j] = float(div[gene_indexes[j]])
    else:
        # pre-allocate memory for cn
        cn_dep = np.zeros((len(depmaps), len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[0]) > 0:
                idx = depmaps.index(div[0])
                for j in range(len(genelist)):
                    if gene_indexes[j] != None:
                        if div[gene_indexes[j]] != '':
                            cn_dep[idx, j] = float(div[gene_indexes[j]])
    out_dict = dict()
    out_dict['nd'] = cn_dep
    out_dict['sample_ids'] = depmaps
    out_dict['geneList'] = genelist
    return (out_dict)


def read_mut_hotspot_depmap(depmaps = None, genelist = None, folder='temp/'):
    """   Function to read copy number from Depmap file.
        :param:
            genelist: The depmap ids whose data are going to be retrieved.
            depmaps: A list with the depmap ids that are going to determine the order of the CNV array.
            folder: A string with the folder where the file will be downloaded to.
        :return:
            A tupple with the following elements:
            nd: A numpy array with entries of 1 if a gene has a dammaging mutation. 
            depmaps: A list with the cell-lines Ids.
            geneList: A list with the gene Symbols
    """
    url = 'https://figshare.com/ndownloader/files/38357504'
    path = download_url(url, folder)
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # find gene in first line
    gene_columns = re.split(',', lineList[0])
    gene_indexes = list()
    # Brilliant shtuff lads
    for i in range(len(gene_columns)):
        # Our genes_list has SYMBOLs so we will re-arrange.
        parenthesis = re.search(' ', gene_columns[i])
        if parenthesis != None:
            gene_columns[i] = gene_columns[i][0:parenthesis.start(0)]
    if genelist == None:
        genelist = gene_columns.copy()
        genelist.pop(0)
        gene_indexes = list()
        for i in range(len(genelist)):
            gene_indexes.append(i+1)
    else:
        for i in range(len(genelist)):
            if genelist[i] != '':
                if gene_columns.count(genelist[i]) > 0:
                    gene_indexes.append(gene_columns.index(genelist[i]))
                else:
                    gene_indexes.append(None)
            else:
                gene_indexes.append(None)
    if depmaps == None:
        n = len(lineList) - 1
        depmaps = list()
        # pre-allocate memory for cn
        cn_dep = np.zeros((n, len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            depmaps.append(div[0])
            for j in range(len(genelist)):
                if gene_indexes[j] != None:
                    if div[gene_indexes[j]] != '':
                        cn_dep[i-1, j] = float(div[gene_indexes[j]])
    else:
        # pre-allocate memory for cn
        cn_dep = np.zeros((len(depmaps), len(genelist)))
        for i in range(1, len(lineList)):
            div = re.split(',', lineList[i])
            if depmaps.count(div[0]) > 0:
                idx = depmaps.index(div[0])
                for j in range(len(genelist)):
                    if gene_indexes[j] != None:
                        if div[gene_indexes[j]] != '':
                            cn_dep[idx, j] = float(div[gene_indexes[j]])
    out_dict = dict()
    out_dict['hotspot'] = cn_dep
    out_dict['sample_ids'] = depmaps
    out_dict['geneList'] = genelist
    return (out_dict)


# make a dictionary with each drug and dataset.
def read_lincs(files_txt='D://Kinomescan_LINCS.txt', lincs_prots_path =  'D://lincs_prots.txt', folder='temp/'):
    """   Function to read the LINCS datasets.
            :param:
                files: A text file with the Kinomscan LINCS studies number to read.
            :return:
                A dictionary with keys corresponding to drug names and values corresponding to the data.
    """
    fo = open(lincs_prots_path)
    lineList  = [line.rstrip('\n') for line in fo]
    fo.close()
    lincs_prots = dict()
    vals = list()
    for i in lineList:
        L = re.split('\t', i)
        res = list()
        res.append(L[1])
        vals.append(L[1])
        res.append(vals.count(L[1])-1)
        lincs_prots[L[0]] = res.copy()
    # see how many times the same gene is used.
    fo = open(files_txt)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    lincs_dict = dict()
    url_p1 = 'https://lincs.hms.harvard.edu/db/datasets/'
    url_p2 = '/results?search=&output_type=.csv'
    for i in range(2,len(lineList)):
        j = re.split('\t', lineList[i])
        # download file from URL.
        url = url_p1 + j[4]+ url_p2
        path = download_url(url=url, folder=folder)
        fo = open(path)
        lineList_2 = [line.rstrip('\n') for line in fo]
        fo.close()
        os.remove(path)
        data_list = list()
        print(f'Read: {(i-1)/(len(lineList)-2)*100:.4f}%')
        if len(lineList_2) > 1:
            oo = re.split(',', lineList_2[0])
            if len(oo) > 4:
                if oo[4] == 'Kd':
                    for k in range(1,len(lineList_2)):
                        oo = re.split(',', lineList_2[k])
                        if len(oo) == 6:
                            if oo[4] != '':
                                # make dictionary.
                                oo_dict = dict()
                                if oo[3][0] == '"':
                                   oo[3] = oo[3][1:]
                                oo_dict['pseudo'] = oo[3]
                                oo_dict['symbol'] = lincs_prots[oo[3]][0]
                                oo_dict['index'] = lincs_prots[oo[3]][1]
                                oo_dict['Kd'] = float(oo[4])
                                data_list.append(oo_dict)
                        else:
                            if oo[5] != '':
                                pseudo = oo[3] + ','+ oo[4]
                                if pseudo[0] == '"':
                                    pseudo = pseudo[1:]
                                if pseudo[len(pseudo)-1] == '"':
                                    pseudo = pseudo[:len(pseudo)-1]
                                oo_dict['pseudo'] = pseudo
                                oo_dict['symbol'] = lincs_prots[pseudo][0]
                                oo_dict['index'] = lincs_prots[pseudo][1]
                                oo_dict['Kd'] = float(oo[5])
                                data_list.append(oo_dict)
                    # this happens when there is a comma in the name.
                lincs_dict[str.lower(j[0])] = data_list.copy()
    return (lincs_dict, lincs_prots)


def read_crtp(depmaps: list, path = 'D://Drug_sensitivity_replicate-level_dose_(CTD^2).csv'):
    """ Read the CRTP Replicate Level dose data from:
        https://depmap.org/portal/download/custom/
        click on Use all cell lines and Use all genes/compounds
        then click: Drug sensitivity replicate-level dose (CTD^2)
        then Download file.
        Return: A lists of torch Data type,
        one  with fields: depmap_id, cell_line, drug, dose, and lineage.
    """
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    L =  lineList[0]
    columns = list()
    # first columns should be depmap_id,cell_line_display_name,lineage_1,lineage_2,lineage_3,lineage_4,
    count = 0
    stj = L[0]
    j =0
    while count < 6:
        if L[j] == ',':
            columns.append(stj)
            j += 1
            stj = L[j]
            j += 1
            count += 1
        else:
            stj += L[j]
            j += 1
    # The rest of columns are finish at reps. (we're not using the , separation because of conflicts with drug name)
    L = L[j-1:]
    columns2 = re.split(' rep\d+', L)
    for i in columns2:
        if len(i) > 1:
            if i[0] == ',':
                i = i[1:]
            columns.append(i)
    assert len(re.split(',', lineList[1])) == len(columns)
    prysm_ids = list()
    count = 0
    row_indexes = list()
    for i in range(1, len(lineList)):
        L = re.split(',', lineList[i])
        prysm_ids.append(L[0])
        if depmaps.count(L[0]) > 0:
            count += 1
            row_indexes.append(i)
    data_points = list()
    for i in row_indexes:
        L = re.split(',', lineList[i])
        # Now go through each column of L.
        for j in range(6,len(L)):
            viability = L[j]
            if viability != '':
                # turn into float.
                via_float = torch.tensor(float(viability), dtype = torch.float)
                depmap_id = L[0]
                cell_line = L[1]
                lineage = L[2]
                exp_char = re.split(' ',columns[j])
                drug = exp_char[0]
                dose = float(re.split('Î', exp_char[len(exp_char)-1])[0])
                data_points.append(Data(depmap_id = depmap_id, cell_line = cell_line, lineage = lineage, drug = drug, dose = dose, viability = via_float))
    return(data_points)




def read_crtp_auc( drugs_list: list, depmaps_ids: list, path = 'D://Drug_sensitivity_AUC_(CTD^2).csv'):
    """ Read the Area Under the Curve Data for CRTP data from:
        https://depmap.org/portal/download/custom/
        click on Use all cell lines and Use all genes/compounds
        then click: Drug sensitivity AUC (CTD^2)
        then Download file.
        Return: A lists of torch Data type,
        one  with fields: depmap_id, cell_line, drug, dose, and lineage.
    """
    # Read CSV file. 
    lineList = list()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            lineList.append(row)
    data_auc = list()
    # The first row has the drug names. First element is empty. 
    # first column should be depmap_id.
    # This is a comma separated file. However some commas might be in the compound names (yikes!)
    # Let's count columns using the second rows?
    drugs = lineList[0]
    keep_index = list()
    for i in range(len(drugs)):
        # get string before the CRT
        d  = re.split(' \(CTRP',drugs[i])
        drugs[i] = d[0].lower()
        if drugs_list.count(drugs[i])>0:
            keep_index.append(i)
    # The rest of columns are finish at reps. (we're not using the , separation because of conflicts with drug name)
    # Let's get the cell-line. 
    for i in range(1,len(lineList)):
        line = lineList[i]
        # get cell-line.
        cell_i = line[0] # this is the DepMap Id.
        if depmaps_ids.count(cell_i)>0:
            for j in keep_index:
                if line[j] != '':
                    auc_j = float(line[j])
                    # create a torch data object.
                    data_auc.append(Data(depmap_id = cell_i, drug = drugs[j], auc = torch.tensor((auc_j-7)/14, dtype = torch.float)))
    return(data_auc, drugs)



def prepare_datasets_depmap_lincs(gene_list:list, sin_cos_exp =50, add_kinomescan_prots = True, norm_x = True):
    lincs_dict, lincs_prots = read_lincs()
    # if add_kinomescan_prots then complete gene_list
    if add_kinomescan_prots:
        for i in lincs_prots.values():
            if gene_list.count(i[0])==0:
                gene_list.append(i[0])
    # now read exp_dep.
    # find max cols
    max_cols = 1
    for i in lincs_prots.values():
        if i[1]>max_cols:
            max_cols = i[1]
    # Now format the Kd dataset.
    Kd_format = dict()
    for i in lincs_dict:
        kd = lincs_dict[i]
        if len(kd)>0:
            # initialize vector
            Kd_vector = torch.zeros((len(gene_list),max_cols),dtype = torch.float)-1.0
            for j in kd:
                gene = j['symbol']
                col_indx = j['index']
                if gene_list.count(gene)>0:
                    row_indx = gene_list.index(gene)
                    Kd_vector[row_indx, col_indx] = j['Kd']
            Kd_format[i] = Kd_vector
    exp_dict = read_exp_depmap(gene_list)
    # now read the copy numbers.
    cn_dict = read_cn_depmap(gene_list, exp_dict['sample_ids'])
    # now the mutations.
    mut_dict = read_mut_depmap(exp_dict['sample_ids'], gene_list)
    fusions_dict = read_fusions_depmap(exp_dict['sample_ids'], gene_list)
    # use the data we need.
    exp_dep = exp_dict['exp']
    cn_dep = cn_dict['cna']
    fusions = fusions_dict['oh_fusions']
    ns = mut_dict['ns']
    pos_muts = mut_dict['pos']
    # Read the CTRP Dataset
    data_points = read_crtp(exp_dict['sample_ids'])
    baseline = dict()
    weights = np.arange(1, sin_cos_exp+1)
    for i in range(len(exp_dict['sample_ids'])):
        exp_i =  np.reshape(exp_dep[i, :], (exp_dep.shape[1], 1))
        if norm_x:
            exp_i = (exp_i - np.mean(exp_i))/np.var(exp_i)**.5
        cn_i = np.reshape(cn_dep[i, :], (exp_dep.shape[1], 1))
        ns_i = np.reshape(ns[i, :], (ns.shape[1], 1))
        fus_i = np.reshape(fusions[i, :], (fusions.shape[1], 1))
        pos_i = pos_muts[i, :]
        pos_smat = np.zeros((ns.shape[1], sin_cos_exp))
        pos_cmat = np.zeros((ns.shape[1], sin_cos_exp))
        w = np.where(pos_i > 0.0)[0]
        for j in range(sin_cos_exp):
            pos_smat[w, j] = np.sin(pos_i[w] / weights[j])
            pos_cmat[w, j] = np.cos(pos_i[w] / weights[j])
        x_ttr = np.concatenate((exp_i, cn_i, ns_i,fus_i, pos_smat, pos_cmat), axis=1)
        baseline[exp_dict['sample_ids'][i]] = torch.tensor(x_ttr, dtype=torch.float)
    # Now Read the Reactome Dataset
    react_dict =  read_reactome(gene_list)
    edges_index_grn_act = react_dict['edges_index_grn_act']
    edges_index_grn_rep = react_dict['edges_index_grn_rep']
    edges_index_ppi_act = react_dict['edges_index_ppi_act']
    edges_index_ppi_inh = react_dict['edges_index_ppi_inh']
    edges_index_ppi_bin = react_dict['edges_index_ppi_bin']
    edge_index_gact = torch.zeros((2, len(edges_index_grn_act)), dtype = torch.long)
    for i in range(len(edges_index_grn_act)):
        edge_index_gact[0,i] = edges_index_grn_act[i][0]
        edge_index_gact[1,i] = edges_index_grn_act[i][1]
    edge_index_grep = torch.zeros((2, len(edges_index_grn_rep)), dtype = torch.long)
    for i in range(len(edges_index_grn_rep)):
        edge_index_grep[0,i] = edges_index_grn_rep[i][0]
        edge_index_grep[1,i] = edges_index_grn_rep[i][1]
    edge_index_pact = torch.zeros((2, len(edges_index_ppi_act)), dtype = torch.long)
    for i in range(len(edges_index_ppi_act)):
        edge_index_pact[0,i] = edges_index_ppi_act[i][0]
        edge_index_pact[1,i] = edges_index_ppi_act[i][1]
    edge_index_pinh = torch.zeros((2, len(edges_index_ppi_inh)), dtype = torch.long)
    for i in range(len(edges_index_ppi_inh)):
        edge_index_pinh[0,i] = edges_index_ppi_inh[i][0]
        edge_index_pinh[1,i] = edges_index_ppi_inh[i][1]
    edge_index_pinh = torch.zeros((2, len(edges_index_ppi_inh)), dtype = torch.long)
    for i in range(len(edges_index_ppi_inh)):
        edge_index_pinh[0,i] = edges_index_ppi_inh[i][0]
        edge_index_pinh[1,i] = edges_index_ppi_inh[i][1]
    edge_index_pbin = torch.zeros((2, len(edges_index_ppi_bin)), dtype = torch.long)
    for i in range(len(edges_index_ppi_bin)):
        edge_index_pbin[0,i] = edges_index_ppi_bin[i][0]
        edge_index_pbin[1,i] = edges_index_ppi_bin[i][1]
    # Create a Data object for the reactome dataset.
    tokens_index = np.arange(0, len(gene_list))
    Reactome_Data = Data(num_nodes=len(gene_list), tokens=torch.tensor(tokens_index, dtype=torch.int),
                         edge_index_pbin =edge_index_pbin, edge_index_pact = edge_index_pact,
                         edge_index_pinh = edge_index_pinh, edge_index_gact = edge_index_gact,
                         edge_index_grep = edge_index_grep)
    # Make a dictionary for the outputs.
    out_dict = dict()
    out_dict['geneList'] = gene_list
    out_dict['Reactome_Data'] = Reactome_Data
    out_dict['baseline'] = baseline
    out_dict['Kd_format'] = Kd_format
    out_dict['data_points'] = data_points
    out_dict['baseline_columns'] = ['exp', 'cn', 'ns', 'fusions', 'pos_sine', 'pos_cosine']
    return out_dict




def average_lincs(lincs_dict):
    drugs_list = list()
    average_lincs_dict = dict()
    for i in list(lincs_dict.keys()):
        if len(lincs_dict[i])>0:
            drugs_list.append(i)
    for i in drugs_list:
        gene_averages = dict()
        for j in lincs_dict[i]:
            if list(gene_averages.keys()).count(j['symbol']) == 0:
                gene_averages[j['symbol']] = {'Kd':j['Kd'], 'count':1}
            else:
                gene_averages[j['symbol']]['Kd'] += j['Kd']
                gene_averages[j['symbol']]['count'] +=1
        for k in gene_averages:
            gene_averages[k]['Kd'] = gene_averages[k]['Kd']/gene_averages[k]['count']
        average_lincs_dict[i] = gene_averages
    return(average_lincs_dict, drugs_list)



def drug_list_edges(average_lincs_dict, drugs_list, edges, gene_list):
    # Pre-allocate memory for output.
    drugs_edges = list()
    edges_weights = list()
    for i in drugs_list:
    # get targets
        for j in average_lincs_dict[i]:
            gene_s = j
            if gene_list.count(gene_s)>0:
                # go through edge to find what edges are part of the edges.
                idx = gene_list.index(gene_s)
                for t in edges:
                    if t[0] == idx or t[1] == idx:
                        drugs_edges.append([drugs_list.index(i), edges.index(t)+len(drugs_list)])
                        edges_weights.append(average_lincs_dict[i][j]['Kd'])
    return(drugs_edges, edges_weights)


 

def lincs_edges_baseline(lincs_prots, gene_list):
    # We will use this function to get a sample of the proteins in lincs.
    uni_genes = list()
    for i in lincs_prots:
        if uni_genes.count(lincs_prots[i][0]) == 0 and gene_list.count(lincs_prots[i][0]) > 0:
            uni_genes.append(lincs_prots[i][0])
    return(uni_genes)




def drug_vector_edges(average_lincs_dict, gene_list, drugs_list, edges):
    # Pre-allocate memory for output.
    drugs_edges = dict()
    for i in drugs_list:
    # get targets
        edges_weights = np.zeros((len(edges),))
        mask = np.zeros((len(edges),))
        for j in average_lincs_dict[i]:
            gene_s = j
            if gene_list.count(gene_s)>0:
                # go through edge to find what edges are part of the edges.
                idx = gene_list.index(gene_s)
                for t in range(len(edges)):
                    if edges[t][0] == idx:
                        edges_weights[t] = (average_lincs_dict[i][j]['Kd'])
                        mask[t] = 1.
        edges_weights_un = edges_weights +0.
        edges_weights[mask==1] =-1*( edges_weights[mask == 1] - np.mean(edges_weights[mask == 1]))
        edges_weights[mask==1] = edges_weights[mask==1]/(np.var(edges_weights[mask==1])**.5)
        drugs_edges[i] = {'weight': edges_weights, 'mask': mask, 'weight_unnorm' : edges_weights_un}
    return(drugs_edges)






def download_pathway_sets(folder = 'temp/'):
    url = 'https://reactome.org/download/current/ReactomePathways.gmt.zip'
    path = download_url(url, folder)
    extract_zip(path, folder)
    path = path[0:len(path)-4]
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # read what genes belong 
    # First column is pathway name, then code, then gene set.
    gene_set_dict = dict()
    # go through limelist object in a for loop and split.
    for i in lineList:
        div = re.split('\t', i)
        descrip = div[0]
        code = div[1]
        set = div[2:]
        gene_set_dict[code] = {'description':descrip, 'set': set}
    return(gene_set_dict)


def download_read_gos(folder = 'temp/'):
    ontology_description = 'http://current.geneontology.org/ontology/go-basic.obo' 
    path = download_url(ontology_description, folder)
    lineList = [line.rstrip('\n') for line in open(path)]
    # we'll return three outputs, 
    # 1 tree of ontologies.
    onto_tree = dict()
    # 2 dict of ontologies.
    onto_dict = dict()
    # 3 ontology by ctype
    onto_type = dict() # for simplicity, we could use the previous one. 
    j = 0
    # Do a while loop. 
    while j < len(lineList):
        # see if we find a line that starts with id then store ontology id.
        if lineList[j] == '[Term]':
            j += 1
            onto_id = re.split(' ',lineList[j])[1]
            j +=1
            onto_desc = re.split(':', lineList[j])[1]
            j += 1
            onto_namespace = re.split(':', lineList[j])[1][1:]
            onto_sub = list()
            while lineList[j]!= '':
                 j += 1
                 div = re.split(' ',  lineList[j])
                 if div[0] == 'is_a:':
                     onto_sub.append(div[1])
            # add elements to onto trees
            onto_tree[onto_id] = onto_sub
            # add elements to onto_dict
            onto_dict[onto_id] = {'name':onto_desc, 'name space':onto_namespace, 'part of':onto_sub}
            # add to onto_type
            if list(onto_type.keys()).count(onto_namespace)>0:
                onto_type[onto_namespace].append(onto_id)
            else:
                onto_type[onto_namespace] = [onto_id]
        else:
            j+= 1
    return(onto_type, onto_dict, onto_tree)



def download_gene_gos(folder = 'temp/'):
    ontology_annot = 'http://geneontology.org/gene-associations/goa_human.gaf.gz'
    path = download_url(ontology_annot, folder)
    extract_gz(path, folder)
    path = path[0:len(path)-3]
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    gene_onto = dict()
    for i in lineList:
        # these file has multiple lines starting with ! to discard
        if i[0] != '!':
            div = re.split('\t', i) # Tab separate the entries.
            gene_id  = div[2] # get SYMBOL
            onto_i = div[4] # Ontology code. 
            if list(gene_onto.keys()).count(gene_id) >0 :
                if gene_onto[gene_id].count(onto_i)==0:
                    gene_onto[gene_id].append(onto_i)
            else:
                gene_onto[gene_id] = [onto_i]
    return(gene_onto)




def download_onco_tree(folder = 'temp/'):
    onco_url = 'http://oncotree.mskcc.org/api/tumor_types.txt'
    path = download_url(onco_url, folder)
    lineList = [line.rstrip('\n') for line in open(path)]
    onco_tree = dict()
    onco_dict = dict()
    # do a for loop and store in dictioanry with two fields. 
    # field 1, directed graph. 
    # field 2 the main name. 
    for i in range(1, len(lineList)):
        div = re.split('\t', lineList[i])
        # first field is the level 1. get the code from within brackets.
        onco_codes = list()
        j =0
        while div[j]!= '':
            splt = re.split(r'[\(\)]', div[j])
            # check if term is in dictionary already. 
            if list(onco_dict.keys()).count(splt[1]) ==0:
                onco_dict[splt[1]] = splt[0]
            onco_codes.append(splt[1])
            j += 1
        # add to onco_tree dictionary.
        onco_tree[onco_codes[len(onco_codes)-1]] = onco_codes
    return(onco_tree, onco_dict)



def download_read_msk_2017(folder = 'temp/', top = 15):
    path, sources, urls = download_study(name = 'msk_pan_2017', folder = folder)
    ret_dict = read_folders.read_cbioportal_folder(path)
    # read the onco_tree and onco_dict.
    onco_tree, onco_dict = download_onco_tree(folder)
    # we need to put the outputs in the following formats.
    sample_proc = msk_2017_sample_info(ret_dict['sample']['sample_dict'])
    # and process then the patient data.
    pat_proc = msk_2017_patient_sample(sample_dict,patient_dict)
     


                
def msk_2017_sample_info(sample_dict, top = 15):
    # Finally, let's go through the sample and clinical datasets.
    # From the sample dictionary we are interested in a couple of things.
    # First: What patient it came from? We will use this for survival and clinical variables. Key: 'Patient Identifier'
    pats = list()
    # Second: What oncotree code is the sample from? Key; 'Oncotree Code'
    onco_codes = dict()
    # Third: Is it primary or metastatic sample. Key: 'Sample Type'
    sample_type = dict()
    # Fourth: Primary tumour Site. Key: 'Primary Tumor Site'.
    tumor_sites = dict()
    # Fifth: Metastatic site. Key: 'Metastatic Site'.
    metas_sites = dict()
    # This is a bit of an odd procedure. We will first create a list
    for i in sample_dict:
        p = sample_dict[i]['Patient Identifier']
        if pats.count(p) == 0:
            pats.append(p)
        # add unique onco codes
        onc = sample_dict[i]['Oncotree Code']
        onc_keys = list(onco_codes.keys())
        if onc_keys.count(onc) == 0:
            onco_codes[onc] = 1
        else:
            onco_codes[onc] += 1
        s = sample_dict[i]['Sample Type']
        samp_keys = list(sample_type.keys())
        if samp_keys.count(s) == 0:
            sample_type[s] = 1
        else:
            sample_type[s] += 1
        ts = sample_dict[i]['Primary Tumor Site']
        tum_keys = list(tumor_sites.keys())
        if tum_keys.count(ts) == 0:
            tumor_sites[ts] = 1
        else:
            tumor_sites[ts] += 1
        ms = sample_dict[i]['Metastatic Site']
        metas_keys = list(metas_sites.keys())
        if metas_keys.count(ms) == 0:
            metas_sites[ms] = 1
        else:
            metas_sites[ms] += 1
    # Now we wish to encode each into a nice matrix. 
    # A couple of situations have risen.
    # we need to encode everything in nummerical form.
    # We will use the onco trees to encode into nummerical matrixes.
    summary = {'patients': pats, 'onco_codes': onco_codes, 'sample_types': sample_type, 'tumor_sites': tumor_sites, 'metas_sites': metas_sites}
    n = len(sample_dict.keys())
    metas_site_bool = np.zeros((n,1))# Matrix for whether there was a metastasic site associated. 
    sample_type_bool = np.zeros((n,2)) # Matrix to indiciate if a sample is from a metastasis or not. 
    # we will select the top common sites (to avoid cancer sites with only 1 or 2 samples for example.)
    metas_site = np.zeros((n, top))
    top_meta = np.argsort(-1*np.array(list(metas_sites.values())))[0:top]
    top_meta_list = list()
    meta_keys = list(metas_sites.keys())
    for i in range(top_meta.shape[0]):
        top_meta_list.append(meta_keys[top_meta[i]])
    # same with tumour type. 
    tumour_site = np.zeros((n, top)) 
    top_tumor = np.argsort(-1*np.array(list(tumor_sites.values())))[0:top]
    top_tumor_list = list()
    tum_keys = list(tumor_sites.keys())
    for i in range(top_tumor.shape[0]):
        top_tumor_list.append(tum_keys[top_tumor[i]])
    # we will download the onco tree dictionary and make a onco encoding? # yeap that sounds correct I supose. 
    oncotree = np.zeros((n, 6)) # there's a maximum of 6 levels for this things. we will take the integers from the oncotree associated.
    # download and read the oncotree.
    onco_tree, onco_dict = download_onco_tree()
    # make a dictionary where we encode each onco code into an integer.
    onco_keys = list(onco_dict.keys())
    j = 0
    for i in sample_dict:
        # Fill in meta_site_pool
        if sample_dict[i]['Metastatic Site'] != 'Not Applicable':
            metas_site_bool[j,0] = 1
        # Next sample type
        if sample_dict[i]['Sample Type'] == 'Primary':
            sample_type_bool[j,0] = 1
        else:
            sample_type_bool[j,1] = 1
        # Write the top metastatic sites. 
        if top_meta_list.count(sample_dict[i]['Metastatic Site'])>0 :
            idx = top_meta_list.index(sample_dict[i]['Metastatic Site'])
            metas_site[j,idx] = 1
        if top_tumor_list.count(sample_dict[i]['Primary Tumor Site'])>0 :
            idx = top_tumor_list.index(sample_dict[i]['Primary Tumor Site'])
            tumour_site[j,idx] = 1
        # Encode the oncotree sitch!
        onco_code = sample_dict[i]['Oncotree Code']
        if list(onco_tree.keys()).count(onco_code)>0:
            onco_situation = onco_tree[onco_code]
            for k in range(len(onco_situation)):
                no = onco_keys.index(onco_situation[k])
                oncotree[j,k] = no 
        j += 1
    output = dict()
    output['summary'] = summary
    output['metas_site_bool']  = metas_site_bool
    output['sample_type_bool'] = sample_type_bool
    output['metas_site'] = metas_site
    output['tumour_site'] = tumour_site
    output['oncotree']  = oncotree
    return(output)



def msk_2017_patient_sample(sample_dict,patient_dict):
    # get keys in sample_dict and put the survival and age into numpy arrays.
    sample_keys = list(sample_dict.keys())
    # get number of obvs. 
    n = len(sample_keys)
    # Pre-allocate memory for Overall Survival, Age, Gender, Smoking
    smoking = np.zeros((n,2)) # Column 1 Never, Column 2 Prev/Curr Smoker, (other value is unknown)
    osurv = np.zeros((n,2)) # First column is overall survival in days 
    gender = np.zeros((n,1)) # We will for simplicity just use one gender as baseline. 
    status = np.zeros((n,1))
    # Do a for loop.
    p_inf= dict()
    for i in patient_dict:
        info = patient_dict[i]
        # make a dictionary with the data 
        pat_info = dict()
        pat_info['gender'] = 1*(info[0]=='Female')
        pat_info['status'] = 1*(info[1]=='DECEASED')
        if info[2] == 'Never':
            pat_info['smoking'] = 1
        elif info[2] == 'Prev/Curr Smoker':
            pat_info['smoking'] = 2
        else:
            pat_info['smoking'] = 0
        if info[3] != '':
            pat_info['survival_time'] = float(info[3])
        else:
            pat_info['survival_time'] = 0
        if info[4] == '1:DECEASED':
            pat_info['censored'] =1
        else:
            pat_info['censored'] = 0.
        p_inf[i] = pat_info
    j = 0
    for i in sample_dict:
        # get patient id. 
        pat_id = sample_dict[i]['Patient Identifier']
        # get info.
        info = p_inf[pat_id]
        # gender
        gender[j,0] = info['gender']
        # smoking
        if info['smoking'] >0 :
            smoking[j,info['smoking']-1] = 1.
        # survival
        osurv[j,0] = info['survival_time']
        osurv[j,1] = info['censored']
        # status
        status[j,0] = info['status']
        j += 1
    out = dict()
    out['pat_info'] = p_inf
    out['status'] = status
    out['gender'] = gender
    out['smoking'] = smoking
    out['osurv'] = osurv
    return(out)

