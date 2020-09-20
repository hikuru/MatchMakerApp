import json
import numpy as np
import pandas as pd
import argparse
from matchmaker.helper_funcs import normalize

def argument_parser():
    parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')
    
    parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv',
                        help="Name of the drug combination data")

    parser.add_argument('--drug-info-name', default='data/drugs_info.json',
                        help="Name of the drug chemical information data")

    parser.add_argument('--cell_line-gex-name', default='data/untreated_gex.csv',
                        help="Name of the cell line gene expression data")

    parser.add_argument('--cell-line-metadata-name', default='data/cell_line_metadata.txt',
                        help="Name of the cell line metadata")

    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--train-test-mode', default=1, type = int,
                        help="Test of train mode (0: test, 1: train)")

    parser.add_argument('--train-ind', default='data/train_inds.txt',
                        help="Data indices that will be used for training")

    parser.add_argument('--val-ind', default='data/val_inds.txt',
                        help="Data indices that will be used for validation")

    parser.add_argument('--test-ind', default='data/train_inds.txt',
                        help="Data indices that will be used for test")

    parser.add_argument('--arch', default='data/architecture.txt',
                        help="Architecute file to construct MatchMaker layers")

    parser.add_argument('--gpu-support', default=False,
                        help='Use GPU support or not')

    parser.add_argument('--saved-model-name', default="matchmaker_saved",
                        help='Model name to save weights')
    args = parser.parse_args()
    return(args)

def print_drug_info(cid,cids):
    #print(cids.keys())
    if cid in cids.keys():
        print("Name: {}".format(cids[cid]['name']))
        print("SMILES: {}".format(cids[cid]['canonical_smiles']))
        print()
        return(True)
    else:
        print("Could not find drug with CID: {}".format(cid))
        return(False)

def read_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return(data)

def get_user_input():
    drugs = read_json('data/drugs_info.json')
    cell_line_data = pd.read_csv('data/cell_line_metadata.txt')
    cell_line_exp = pd.read_csv('data/untreated_gex.csv')
    first_drug = False
    while first_drug is False:
        cid1 = input("Enter PubChem CID of first drug: ") 
        print("Your first drug's info")
        first_drug = print_drug_info(cid1,drugs)
    
    second_drug = False
    while second_drug is False:
        cid2 = input("Enter PubChem CID of second drug: ") 
        print("Your second drug's info")
        second_drug = print_drug_info(cid2,drugs)

    tissues = np.unique(cell_line_data['Tissue'])
    print("TISSUE TYPES")
    for i in range(tissues.shape[0]):
        print("{}. {}".format(i, tissues[i]))

    tissue_type = input("Select the tissue type by typing its number from 0 to {}: ".format(tissues.shape[0]))
    tis = tissues[int(tissue_type)]
    print("You selected {} tissue".format(tis))
    print()
    filtered_cl = np.array(cell_line_data['Cell_line'][cell_line_data['Tissue'] == tis])
    gdsc = np.array(cell_line_data['GDSC_id'][cell_line_data['Tissue'] == tis])
    print("CELL LINE TYPES")
    for i in range(filtered_cl.shape[0]):
        print("{}. {}".format(i, filtered_cl[i]))
    cl_type = input("Select the cell line by typing its number from 0 to {}: ".format(filtered_cl.shape[0]))
    cl = filtered_cl[int(cl_type)]
    cl_id = gdsc[int(cl_type)]
    print("You selected {} cell line".format(cl))
    chem1 = np.asarray(drugs[cid1]['chemicals'])
    chem2 = np.asarray(drugs[cid2]['chemicals'])
    cl_exp = np.asarray(cell_line_exp[cl_id])
    chem1 = chem1.reshape((1,chem1.shape[0]))
    chem2 = chem2.reshape((1,chem2.shape[0]))
    cl_exp = cl_exp.reshape((1,cl_exp.shape[0]))
    return(cid1, cid2, cl, cl_id, chem1, chem2, cl_exp)

def prep_pair(chem1, chem2, cl_exp):
    mean1 = np.loadtxt('data/normalization_params/drug1_mean1.txt')
    mean2 = np.loadtxt('data/normalization_params/drug1_mean2.txt')
    std1 = np.loadtxt('data/normalization_params/drug1_std1.txt')
    std2 = np.loadtxt('data/normalization_params/drug1_std2.txt')
    feat_filt = np.loadtxt('data/normalization_params/drug1_feat_filt.txt',dtype=np.bool)
    pair = {}
    pair['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1,
                                                                        mean1, 
                                                                        std1, 
                                                                        mean2, 
                                                                        std2, 
                                                                        feat_filt=feat_filt, 
                                                                        norm='tanh_norm')

    mean1 = np.loadtxt('data/normalization_params/drug2_mean1.txt')
    mean2 = np.loadtxt('data/normalization_params/drug2_mean2.txt')
    std1 = np.loadtxt('data/normalization_params/drug2_std1.txt')
    std2 = np.loadtxt('data/normalization_params/drug2_std2.txt')
    feat_filt = np.loadtxt('data/normalization_params/drug2_feat_filt.txt',dtype=np.bool)
    pair['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2,
                                                                        mean1, 
                                                                        std1, 
                                                                        mean2, 
                                                                        std2, 
                                                                        feat_filt=feat_filt, 
                                                                        norm='tanh_norm')
    
    mean1 = np.loadtxt('data/normalization_params/cl_mean1.txt')
    mean2 = np.loadtxt('data/normalization_params/cl_mean2.txt')
    std1 = np.loadtxt('data/normalization_params/cl_std1.txt')
    std2 = np.loadtxt('data/normalization_params/cl_std2.txt')
    feat_filt = np.loadtxt('data/normalization_params/cl_feat_filt.txt',dtype=np.bool)
    cl_exp, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(cl_exp,
                                                                mean1, 
                                                                std1, 
                                                                mean2, 
                                                                std2, 
                                                                feat_filt=feat_filt, 
                                                                norm='tanh_norm')
    pair['drug1'] = np.concatenate((pair['drug1'],cl_exp),axis=1)
    pair['drug2'] = np.concatenate((pair['drug2'],cl_exp),axis=1)
    return(pair)

def read_architecture():
    architecture = pd.read_csv('matchmaker/architecture.txt')
    # prepare layers of the model and the model name
    layers = {}
    layers['layers1'] = architecture['DSN_1'][0] # layers of Drug Synergy Network 1
    layers['layers2'] = architecture['DSN_2'][0] # layers of Drug Synergy Network 2
    layers['concatLayer'] = architecture['SPN'][0] # layers of Synergy Prediction Network
    return(layers)

def read_norm_params():
    mean1 = np.loadtxt('data/normalization_params/mean1.txt')
    mean2 = np.loadtxt('data/normalization_params/mean2.txt')
    std1 = np.loadtxt('data/normalization_params/std1.txt')
    std2 = np.loadtxt('data/normalization_params/std2.txt')
    feat_filt = np.loadtxt('data/normalization_params/feat_filt.txt',dtype=np.bool)
    return(mean1, mean2, std1, std2, feat_filt)

'''with open('data/drugs_cid.json') as json_file:
    cids = json.load(json_file)

#val = input("Enter your value: ") 
print("DONE") 
cid1 = input("Enter PubChem CID of first drug: ") 
print("Your first drug's info")
print_drug_info(cid1,cids)
cid2 = input("Enter PubChem CID of second drug: ") 
print("Your second drug's info")
print_drug_info(cid2,cids)
print(cids[int(cid2)]['chemicals'])

cid1 = input("Enter PubChem CID of first drug: ") 
print("Your first drug's info")
print_drug_info(cid1,cids)
cid2 = input("Enter PubChem CID of second drug: ") 
print("Your second drug's info")
print_drug_info(cid2,cids)

cell_line_data = pd.read_csv('data/cell_line_metadata.txt')
tissues = np.unique(cell_line_data['Tissue'])
print("TISSUE TYPES")
for i in range(tissues.shape[0]):
    print("{}. {}".format(i, tissues[i]))

tissue_type = input("Select the tissue type by typing its number from 0 to {}: ".format(tissues.shape[0]))
tis = tissues[int(tissue_type)]
print("You selected {} tissue".format(tis))
print()
filtered_cl = np.array(cell_line_data['Cell_line'][cell_line_data['Tissue'] == tis])
print("CELL LINE TYPES")
for i in range(filtered_cl.shape[0]):
    print("{}. {}".format(i, filtered_cl[i]))
cl_type = input("Select the cell line by typing its number from 0 to {}: ".format(filtered_cl.shape[0]))
cl = filtered_cl[int(cl_type)]
print("You selected {} cell line".format(cl))

chem1 = cids[int(cid1)]['chemicals']
chem2 = cids[int(cid2)]['chemicals']
cc = np.concatenate((chem1,chem2))'''