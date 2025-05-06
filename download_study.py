from torch_geometric.data import download_url, extract_zip, extract_gz, extract_tar
import mysql.connector
from collections import defaultdict

def url_search(name = None, keywords = None):
    """   Function with dictionary with url guides from selected studies. There is also a cbio package for python but we are using this as a learning tool. 
    :param
        name: A string with The Name of the study or a list of strings with the names. 
        keywords: A list with strings that correspond to keywords.
    :return: 
        urls: a list with urls according to the keywords search. 
        sources: a list with the origins (or sources) of the 
    """
    # Start dictionary for each study name.
    # Append each study to dictionary. 
    study_dict = dict()
    msk_immuno_2019 = {'url':'https://cbioportal-datahub.s3.amazonaws.com/tmb_mskcc_2018.tar.gz', 'source':'cbioportal','keywords':['immunotheray','pan-cancer', 'msk', 'mutations', 'fusions', 'muts', 'inmuno', 'pan', 'thousands', 'patient', 'large', '2019']}
    study_dict['msk_immuno_2019'] = msk_immuno_2019 
    msk_pan_2017 = { 'url':'https://cbioportal-datahub.s3.amazonaws.com/msk_impact_2017.tar.gz', 'source':'cbioportal','keywords':['treatments','pan-cancer', 'msk', 'pan', 'mutations', 'fusions','muts', 'thousands', 'patient', 'large', '2017']}
    study_dict['msk_pan_2017'] = msk_pan_2017 
    ontology_annotations = {'url':'http://geneontology.org/gene-associations/goa_human.gaf.gz', 'source':'GO', 'keywords':['annotation', 'annot', 'ontology', 'go', 'nodes']}
    study_dict['ontology_anotations'] = ontology_annotations
    ontology_description = {'url': 'http://current.geneontology.org/ontology/go-basic.obo' ,'source':'GO', 'keywords':['annotation', 'annot', 'ontology', 'go', 'nodes', 'names', 'ontology description']}
    study_dict['ontology_description'] = ontology_description
    reactome_pathways_gene_set = {'url':'https://reactome.org/download/current/ReactomePathways.gmt.zip', 'source':'Reactome', 'keywords':['genesets' 'sets','annotation', 'annot', 'pathways', 'reactome','go', 'nodes']} 
    study_dict['reactome_pathways_gene_set'] = reactome_pathways_gene_set
    reactome_pathways = {'url':'https://reactome.org/download/current/gene_association.reactome.gz','source':'Reactome', 'keywords':['annotation', 'annot', 'pathways', 'reactome','go', 'nodes']}
    study_dict['reactome_pathways'] = reactome_pathways 
    reactome_pathways_names =  {'url':'https://reactome.org/download/current/ReactomePathways.txt','source':'Reactome', 'keywords':['annotation', 'pathway names', 'pathways', 'reactome','go', 'nodes']}
    study_dict['reactome_pathways_names'] = reactome_pathways_names
    reactome_reactions = {'url':'https://reactome.org/download/tools/ReatomeFIs/FIsInGene_122921_with_annotations.txt.zip','source':'Reactome', 'keywords':['annotation', 'reactions', 'pathways', 'reactome','go', 'nodes', 'graph']}
    study_dict['reactome_reactions'] = reactome_reactions
    target_all = {'url':'https://cbioportal-datahub.s3.amazonaws.com/aml_target_2018_pub.tar.gz', 'source':'cbioportal', 'keywords':['leukimia', 'target', 'TARGET', 'leu', 'childhood', 'patient', 'muts', 'meth', 'methylation', 'exp',  'large', '2018']}
    study_dict['target_all'] = target_all
    target_all_II = {'url':'https://cbioportal-datahub.s3.amazonaws.com/all_phase2_target_2018_pub.tar.gz', 'source':'cbioportal', 'keywords':['leukimia', 'target', 'TARGET', 'leu', 'childhood', 'patient', 'muts', 'exp', 'meth', 'methylation', 'large', '2018']}
    study_dict['target_all_II'] = target_all_II
    onco_tree = {'url': 'http://oncotree.mskcc.org/api/tumor_types.txt', 'source':'msk', 'keywords':['annotation', 'msk', 'onco_tree', 'tree', 'onco', 'cancer types', 'clinical']}
    study_dict['onco_tree'] = onco_tree
    keyword_dictionary = dict()
    urls = list()
    sources = list()
    # do a for loop aross keys
    for i in study_dict:
        key_old = list(keyword_dictionary.keys())
        key_new = study_dict[i]['keywords']
        for j in key_new:
            if key_old.count(j)>0:
                # if the key already exists simply append a new entry.
                keyword_dictionary[j].append(i)
            else:
                # if the key is not in then add. 
                keyword_dictionary[j] = [i]
    if name != None:
        key_study = list(study_dict.keys())
        if str(type(keywords)) != "<class 'list'>":
            if key_study.count(name)>0:
                urls.append(study_dict[name]['url'])
                sources.append(study_dict[name]['source'])
            else:
                raise Exception("Study name not found")
        else:
            for i in name:
                if key_study.count(i)>0:
                    urls.append(study_dict[i]['url'])
                    sources.append(study_dict[i]['source'])
            if len(urls)==0:
                raise Exception("Study name not found")
    elif keywords != None:
        if str(type(keywords)) != "<class 'list'>":
            keywords = [keywords] # convert into list. 
        keys_dict = list(keyword_dictionary.keys())
        for i in keywords:
            if keys_dict.count(i) > 0:
                # get studies 
                studies_keywords = keyword_dictionary[i]
                for j in studies_keywords:
                    if urls.count(study_dict[j]['url'])==0:
                        urls.append(study_dict[j]['url'])
                        sources.append(study_dict[j]['source'])
        if len(urls)==0:
            raise Exception("Keyword name not found")
    else:
        for i in study_dict:
            urls.append(study_dict[i]['url'])
            sources.append(study_dict[i]['source'])
    return(urls, sources)

def download_study(name = None, keywords = None, folder = 'temp/'):
    
    """   Function to download file from url and extract.
    :param
        name: A string with The Name of the study or a list of strings with the names. 
        keywords: A list with strings that correspond to keywords.
        folder: A string with the folder where the file will be downloaded to. 
    :return:
        paths: A list with the folders in the local machine where the file is. 
    """
    # Search for url
    urls, sources = url_search(name, keywords)
    paths = list()
    # From url determine what kind of file it is.
    for i in urls:
        nchar = len(i)
        if i[len(i)-4:len(i)] == '.zip':
            paths.append(download_extract_zip(i, folder))
        elif i[len(i)-6:len(i)] == 'tar.gz':
            paths.append(download_tar_gz(i, folder))
        elif i[len(i)-3:len(i)] == '.gz' and i[len(i)-6:len(i)]  != '.tar.gz':
            path = download_url(i, folder)
            extract_gz(path, folder)
            paths.append(path[0:len(path)-3])
        elif i[len(i)-4:len(i)] == '.txt':
            path = download_url(i, folder)
            paths.append(path)
    return(paths, sources, urls)

def download_extract_zip(url,folder='temp/'):
    """   Function to download file from url and extract zip.
    :param
        url: the url with the zip folder to download. 
        folder: A string with the folder where the file will be downloaded to.
    :return: 
        path: the folder in the local machine where the file is. 
    """
    path = download_url(url, folder)
    # extract zip 
    extract_zip(path, folder)
    # remove .zip from path.
    path = path[0:len(path)-4]
    return(path)

def download_tar_gz(url,folder='temp/'):
    """   Function to download file from url and extract zip.
    :param
        url: the url with the zip folder to download. 
        folder: A string with the folder where the file will be downloaded to.
    :return: 
        path: the folder in the local machine where the file is. 
    """
    path = download_url(url, folder)
    # extract tar 
    extract_tar(path, folder)
    # extract gz
    extract_gz(path, folder)
    # remove .tar.gz from path.
    path = path[0:len(path)-7]
    return(path)

def get_gene_lengths(genome="hg19"):

    """
    Outputs dictionary: Gene -> [txStart, txEnd] for the transcript with the maximum length.
    MSK = hg19
    """
    
    if not genome:
        genome = "hg19"  # Default to hg19 if None

    # Connect to UCSC MySQL server
    try:
        conn = mysql.connector.connect(
            host="genome-mysql.soe.ucsc.edu",
            user="genome",
            password="",
            database=genome
        )
    except mysql.connector.Error as e:
        raise ValueError(f"Failed to connect to UCSC database {genome}: {e}")

    cursor = conn.cursor(dictionary=True)
    
    # Query refGene for gene symbol, txStart, txEnd
    query = "SELECT name2, txStart, txEnd FROM refGene"
    cursor.execute(query)

    # Collect coordinates and find the transcript with max length per gene
    gene_coord_temp = defaultdict(list)
    for row in cursor:
        gene_name = row["name2"]
        txStart = row["txStart"]
        txEnd = row["txEnd"]
        length = txEnd - txStart
        gene_coord_temp[gene_name].append((txStart, txEnd, length))
    
    # For each gene, select the transcript with the maximum length
    gene_lengths = {}
    for gene, transcripts in gene_coord_temp.items():
        # Find the transcript with the maximum length
        max_transcript = max(transcripts, key=lambda x: x[2])
        gene_lengths[gene] = [max_transcript[0], max_transcript[1]]  # [txStart, txEnd]
    
    cursor.close()
    conn.close()
    
        # Map known aliases to match mutation data
    aliases = {
        'FAM175A': 'ABRAXAS1',
        'FAM46C': 'TENT5C',
        'FAM58A': 'CCNQ',
        'MRE11A': 'MRE11',
        'PAK7': 'PAK5',
        'PARK2': 'PRKN',
        'TCEB1': 'ELOC',
        'WHSC1': 'NSD2',
        'WHSC1L1': 'NSD3',
        'HIST1H1C': 'H1-2', 
        'HIST1H2BD': 'H2BC5',
        'HIST1H3A': 'H3C1',
        'HIST1H3B': 'H3C2',
        'HIST1H3C': 'H3C3',
        'HIST1H3D': 'H3C6',
        'HIST1H3E': 'H3C7',
        'HIST1H3F': 'H3C8',
        'HIST1H3G': 'H3C10',
        'HIST1H3H': 'H3C11',
        'HIST1H3I': 'H3C12',
        'HIST1H3J': 'H3C13',
        'HIST3H3': 'H3-4',
        'H3F3A': 'H3-3A',
        'H3F3B': 'H3-3B',
        'H3F3C': 'H3-5',
        'RFWD2': 'COP1',
        'SETD8': 'KMT5A',
        'C10orf76': 'ARMH3',   
        'GPR1-AS1': 'CMKLR2-AS',
        'CXorf26': 'MRPL57',    
        'DEPDC6': 'DEPTOR',     
        'MUT': 'MMUT'           
    }

    for old, new in aliases.items():
        if new in gene_lengths and old not in gene_lengths:
            gene_lengths[old] = gene_lengths[new]

    # Add manual coordinates for RP11-211G3.3
    # Manually add coordinates for unresolved genes
    gene_manual = {
        'AATK-AS1': [79074314, 79106851],    # chr17:79,074,314-79,106,851
        'CMKLR2-AS': [206373351, 206412263], # chr2:206,373,351-206,412,263
        'LINC02915': [75297403, 75319189],   # chr15:75,297,403-75,319,189
        'RP11-211G3.3': [187702313, 187733849]  
    } 

    for gene, coord in gene_manual.items():
        if gene not in gene_lengths.keys():
            gene_lengths[gene] = coord

    return gene_lengths
