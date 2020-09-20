import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matchmaker.helper_funcs import normalize

# comb_data_name: 'data/DrugCombinationData.tsv'
# drug_info_name: 'data/drugs_info.json'
# cell_line_gex_name: 'data/untreated_gex.csv'
# cell_line_metadata_name: 'data/cell_line_metadata.txt'
def data_loader(comb_data_name, drug_info_name, cell_line_gex_name, cell_line_metadata_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")

    with open(drug_info_name) as json_file1:
        drug_info = json.load(json_file1)

    # drug_names = list(drug_info.keys())
    cl_gex = pd.read_csv(cell_line_gex_name)

    cl_gdsc = pd.read_csv(cell_line_metadata_name,delimiter=",")

    


    drug1_name = []
    drug2_name = []
    cell_name = []
    drug1_chem = []
    drug2_chem = []
    cell_gex = []
    synergy_scores = []

    nrow = comb_data.shape[0]

    print("Data parsing step has just started")
    progress(0, nrow, prefix = 'Progress:', suffix = 'Complete', length = 100)

    for i in range(nrow):
        # select drugs, cell line and synergy score from combination data
        d1_name = comb_data.iloc[[i]]["drug_row"].values[0]
        d2_name = comb_data.iloc[[i]]["drug_col"].values[0]
        cname = comb_data.iloc[[i]]["cell_line_name"].values[0]
        score = comb_data.iloc[[i]]["synergy_loewe"].values[0]
        gdsc_id = cl_gdsc["GDSC_id"][cl_gdsc["Cell_line"]==cname]
        cell_gex.append(np.array(cl_gex[gdsc_id]).flatten())
        drug1_chem.append(list(drug_info[d1_name]["chemicals"]))
        drug2_chem.append(list(drug_info[d2_name]["chemicals"]))
        synergy_scores.append(score)
        # print the prorgess status
        progress(i, nrow, prefix = 'Progress:', suffix = 'Complete', length = 100)

    print("Data parsing step ended")
    return np.array(drug1_chem), np.array(drug1_chem), np.array(cell_gex)



def prepare_data(chem1, chem2, cell_line, norm, train_ind, val_ind, test_ind):
    print("Data normalization and preparation of train/validation/test data")
    train_data = {}
    val_data = {}
    test_data = {}
    train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0)
    train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
    val_data['val1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['test1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
    train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
    val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)

    train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
    train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
    val_cell_line, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(cell_line[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
    train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
    val_data_rev = {}
    val_data_rev['drug2'] = np.concatenate((val_data['drug1'],val_data["val3"]),axis=1)
    val_data_rev['drug1'] = np.concatenate((val_data['drug2'],val_data["val3"]),axis=1)

    val_data['drug1'] = np.concatenate((val_data['drug1'],val_data["val3"]),axis=1)
    val_data['drug2'] = np.concatenate((val_data['drug2'],val_data["val3"]),axis=1)

    test_data_rev = {}
    test_data_rev['drug2'] = np.concatenate((test_data['drug1'],test_data["test3"]),axis=1)
    test_data_rev['drug1'] = np.concatenate((test_data['drug2'],test_data["test3"]),axis=1)

    test_data['drug1'] = np.concatenate((test_data['drug1'],test_data["test3"]),axis=1)
    test_data['drug2'] = np.concatenate((test_data['drug2'],test_data["test3"]),axis=1)

    train_data['y'] = np.concatenate((synergies[train_ind],synergies[train_ind]),axis=0)
    val_data['y'] = synergies[val_ind]
    test_data['y'] = synergies[test_ind]
    return train_data, val_data, test_data

def generate_network(train, layers, modelName, inDrop, drop):
    # fill the architecture params from dict
    dsn1_layers = layers["layers1"].split("-")
    dsn2_layers = layers["layers2"].split("-")
    snp_layers = layers["concatLayer"].split("-")
    # contruct two parallel networks
    for l in range(len(dsn1_layers)):
        if l == 0:
            input_drug1    = Input(shape=(train["drug1"].shape[1],))
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug1)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn1_layers)-1):
            dsn1_output = Dense(int(dsn1_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn1_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)

    for l in range(len(dsn2_layers)):
        if l == 0:
            input_drug2    = Input(shape=(train["drug2"].shape[1],))
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug2)
            middle_layer = Dropout(float(inDrop))(middle_layer)
        elif l == (len(dsn2_layers)-1):
            dsn2_output = Dense(int(dsn2_layers[l]), activation='linear')(middle_layer)
        else:
            middle_layer = Dense(int(dsn2_layers[l]), activation='relu')(middle_layer)
            middle_layer = Dropout(float(drop))(middle_layer)
    
    concatModel = concatenate([dsn1_output, dsn2_output])
    
    for snp_layer in range(len(snp_layers)):
        if len(snp_layers) == 1:
            snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
            snp_output = Dense(1, activation='linear')(snpFC)
        else:
            # more than one FC layer at concat
            if snp_layer == 0:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
                snpFC = Dropout(float(drop))(snpFC)
            elif snp_layer == (len(snp_layers)-1):
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snp_output = Dense(1, activation='linear')(snpFC)
            else:
                snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
                snpFC = Dropout(float(drop))(snpFC)

    model = Model([input_drug1, input_drug2], snp_output)
    return model

def trainer(model, l_rate, train, val, epo, batch_size, earlyStop, modelName,weights):
    cb_check = ModelCheckpoint(('trained_model/'+modelName+'.h5'), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=float(l_rate), beta_1=0.9, beta_2=0.999, amsgrad=False))
    model.fit([train["drug1"], train["drug2"]], train["y"], epochs=epo, shuffle=True, batch_size=batch_size,verbose=1, 
                   validation_data=([val["drug1"], val["drug2"]], val["y"]),sample_weight=weights,
                   callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = earlyStop),cb_check])

    return model

def predict(model, data):
    pred = model.predict(data)
    return pred.flatten()




'''for trio in unique_trios:
        inds = np.where(all_list == trio)
        iy = np.array(inds[0])
        ii = int(iy[0])
        d1_name = drug1[ii]
        d2_name = drug2[ii]
        cname = cl[ii]

        dd = comb_data.loc[comb_data["cell_line_name"]==(cname)]
        dd = dd.loc[dd["drug_row"]==(d1_name)]
        dd = dd.loc[dd["drug_col"]==(d2_name)]
        score = float(dd["synergy_loewe"])
        
        gdsc_id = cl_gdsc["GDSC_id"][cl_gdsc["Cell_line"]==cname]
        cell_gex.append(np.array(cl_gex[gdsc_id]).flatten())
        drug1_chem.append(list(drug_info[d1_name]["chemicals"]))
        drug2_chem.append(list(drug_info[d2_name]["chemicals"]))
        synergy_scores.append(score)'''