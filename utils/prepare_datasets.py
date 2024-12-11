
import os
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env== 'Mol':
    import torch
    from fast_transformers.masking import LengthMask as LM
    import deepchem as dc


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import ast
from constants import *
# from util_alignment import *

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size


def embed(model, smiles, tokenizer, batch_size=64):
    # print(len(model.blocks.layers))
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.blocks.layers[0].register_forward_hook(get_activation('0'))
    model.blocks.layers[1].register_forward_hook(get_activation('1'))
    model.blocks.layers[2].register_forward_hook(get_activation('2'))
    model.blocks.layers[3].register_forward_hook(get_activation('3'))
    model.blocks.layers[4].register_forward_hook(get_activation('4'))
    model.blocks.layers[5].register_forward_hook(get_activation('5'))
    model.blocks.layers[6].register_forward_hook(get_activation('6'))
    model.blocks.layers[7].register_forward_hook(get_activation('7'))
    model.blocks.layers[8].register_forward_hook(get_activation('8'))
    model.blocks.layers[9].register_forward_hook(get_activation('9'))
    model.blocks.layers[10].register_forward_hook(get_activation('10'))
    model.blocks.layers[11].register_forward_hook(get_activation('11'))
    model.eval()
    embeddings = []
    keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    activations_embeddings = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            
            token_embeddings = model.blocks(model.tok_emb(torch.as_tensor(idx)), length_mask=LM(mask.sum(-1)))
            
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())
            
            for i,key in enumerate(keys):
                transformer_output= activation[key]
                input_mask_expanded = mask.unsqueeze(-1).expand(transformer_output.size()).float()
                sum_embeddings = torch.sum(transformer_output * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                activations_embeddings[i].append(embedding.detach().cpu())
    return embeddings, activations_embeddings
def postproce_molembeddings(embeddings,index):

    molecules_embeddings_penultimate = torch.cat(embeddings)
    columns_size= int(molecules_embeddings_penultimate.size()[1])
    if index.ndim>1:
        molecules_embeddings_penultimate = torch.cat((  torch.from_numpy( index.to_numpy()),molecules_embeddings_penultimate), dim=1)
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=['CID','subject']+[str(i) for i in range(columns_size)])
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID','subject'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()

        
    else:
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=[str(i) for i in range(columns_size)])
        df_molecules_embeddings['CID']=index
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()
    df_molecules_embeddings=df_molecules_embeddings.reset_index()
    return df_molecules_embeddings


def prepare_mols_helper(lm,tokenizer,df_mols,mol_type="nonStereoSMILES",index="CID",modeldeepchem=None):
    df_mols_layers=[]
    df_mols_layers_zscored=[]
    
    #inference on molecules
    df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_mols[mol_type], tokenizer, batch_size=64)
    df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])

     #z-score embeddings
    df_mols_embeddings_zscored = zscore_embeddings(df_mols_embeddings,dim=768)

    for df_mols_layer in df_mols_layers_original:
        df_mols_layer = postproce_molembeddings(df_mols_layer, df_mols[index])
        df_mols_layers.append(df_mols_layer)

        # z-score embeddings
        df_mols_embeddings_zscored = zscore_embeddings(df_mols_layer)
        df_mols_layers_zscored.append(df_mols_embeddings_zscored)

    #linear transformation of embeddings
    #
    if modeldeepchem is not None:
        df_mols_embeddings_linear = linear_transformation_embeddings(df_mols, df_mols_embeddings, index, modeldeepchem)
        #z-score linear embeddings
        df_mols_embeddings_linear_zscored = zscore_embeddings(df_mols_embeddings_linear,dim=256)

        return df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored
    else:
        return df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored


def linear_transformation_embeddings(df_mols, df_mols_embeddings, index, modeldeepchem):
    df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
    df_mols_embeddings_linear = modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
    df_mols_embeddings_linear_torch = [torch.from_numpy(x.reshape(1, -1)) for x in df_mols_embeddings_linear]
    df_mols_embeddings_linear = postproce_molembeddings(df_mols_embeddings_linear_torch, df_mols[index])
    return df_mols_embeddings_linear


def zscore_embeddings(df_mols_embeddings,dim=768):
    df_mols_embeddings_zscored = df_mols_embeddings.copy()
    scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':str(dim-1)].values.tolist())
    df_mols_embeddings_zscored.loc[:, '0':str(dim-1)] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index,
                                                                columns=[str(i) for i in range(dim)])
    df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':str(dim-1)].values.tolist()
    return df_mols_embeddings_zscored


def prepare_mols_helper_mixture(df_mols_embeddings_original,df_mols,start,end,mol_type="nonStereoSMILES",index="CID",modeldeepchem=None):
    df_mols_layers=[]
    df_mols_layers_zscored=[]
    
 
    
        
    df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])

    
    
    # df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
    # df_mols_embeddings_linear=modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
    # df_mols_embeddings_linear_torch=[torch.from_numpy(x.reshape(1,-1)) for x in df_mols_embeddings_linear]
    # df_mols_embeddings_linear=postproce_molembeddings(df_mols_embeddings_linear_torch,df_mols[index])
    
    
     #z-score embeddings
    df_mols_embeddings_zscored = df_mols_embeddings.copy()
    scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, start:end].values.tolist())
    df_mols_embeddings_zscored.loc[:, start:end] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(int(end)+1)])
    df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, start:end].values.tolist()
    
    
    
    #z-score linear embeddings
    # df_mols_embeddings_linear_zscored = df_mols_embeddings_linear.copy()
    # scaled_features = StandardScaler().fit_transform(df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist())
    # df_mols_embeddings_linear_zscored.loc[:, '0':'255'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_linear_zscored.index, columns=[str(i) for i in range(256)])
    # df_mols_embeddings_linear_zscored['Combined'] = df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist()


    

        
    
    # ÷return df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored

    return df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored,


def prepare_keller():
    # input_file_keller = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_keller2016.csv'
    input_file_keller = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/curated_keller2016_nona.csv'
    df_keller=pd.read_csv(input_file_keller)
    df_keller=df_keller.replace(-1000.0, np.NaN)
    df_keller=df_keller.dropna(subset=['Acid', 'Ammonia',
       'Bakery', 'Burnt', 'Chemical', 'Cold', 'Decayed', 'Familiarity', 'Fish',
       'Flower', 'Fruit', 'Garlic', 'Grass', 'Intensity', 'Musky',
       'Pleasantness', 'Sour', 'Spices', 'Sweaty', 'Sweet', 'Warm', 'Wood'])
    n_components=5
    print(df_keller.columns)
    
    #Average of ratings per Molecule
    df_keller_mean =df_keller.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    df_keller_mean['Combined'] = df_keller_mean.loc[:, 'Acid':'Wood'].values.tolist()
    
    #Z-score Keller dataset
    df_keller_zscored = df_keller_mean.copy()
    df_keller_zscored=df_keller_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_keller_zscored.loc[:, 'Acid':'Wood'].values.tolist())
    df_keller_zscored.loc[:, 'Acid':'Wood'] = pd.DataFrame(scaled_features, index=df_keller_zscored.index, columns=df_keller_zscored.columns[5:])
    
    
    # print("df_keller_zscored.columns[5:]",df_keller_zscored.columns[5:])
    
    #Mean over z-score keller
    df_keller_zscored_mean =df_keller_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_keller_zscored['Combined'] = df_keller_zscored.loc[:, 'Acid':'Wood'].values.tolist()
    df_keller_zscored_mean['Combined'] = df_keller_zscored_mean.loc[:, 'Acid':'Wood'].values.tolist()
    
    
    #PCA on z-scored Keller
    df_keller_zscored_cid_combined = df_keller_zscored[['CID', 'Combined']]
    df_keller_zscored_pca=PCA_df(df_keller_zscored_cid_combined,'Combined' )
    
    #PCA on z-scored_mean Keller
    df_keller_zscored_mean_cid_combined = df_keller_zscored_mean[['CID', 'Combined']]
    df_keller_zscored_mean_pca=PCA_df(df_keller_zscored_mean_cid_combined,'Combined',n_components=n_components )
    
    #Mean on z_scored_PCA
    df_keller_zscored_pca_mean=df_keller_zscored_pca.drop('Combined',axis=1)
    df_keller_zscored_pca_mean =df_keller_zscored_pca_mean.groupby(['CID']).mean().reset_index()
    
    # df_mean_reduced_keller_zscored_cid_combined =df_keller_zscored_pca.groupby(['CID']).mean().reset_index()
    df_keller_zscored_pca_mean['Combined']=df_keller_zscored_pca_mean.loc[:, 0:n_components-1].values.tolist()
    df_keller_zscored_pca_mean=df_keller_zscored_pca_mean.drop([0,1,2,3,4],axis=1)
    
    
    return df_keller, df_keller_mean, df_keller_zscored, df_keller_zscored_mean, df_keller_zscored_pca,df_keller_zscored_mean_pca,df_keller_zscored_pca_mean


# def prepare_keller_mols(modeldeepchem_gslf,lm,tokenizer):
#     df_keller_mols = df_keller.drop_duplicates('CID')
#     print(df_keller_mols.columns)
#     df_keller_mols_embeddings_original,df_keller_mols_layers_original,df_keller_mols_embeddings,df_keller_mols_embeddings_zscored,df_keller_mols_layers,df_keller_mols_layers_zscored,df_keller_mols_embeddings_linear,df_keller_mols_embeddings_linear_zscored=prepare_mols_helper(lm,tokenizer,df_keller_mols,mol_type="nonStereoSMILES",modeldeepchem=modeldeepchem_gslf)
#     return df_keller_mols,df_keller_mols_embeddings_original,df_keller_mols_layers_original,df_keller_mols_embeddings,df_keller_mols_embeddings_zscored,df_keller_mols_layers,df_keller_mols_layers_zscored,df_keller_mols_embeddings_linear,df_keller_mols_embeddings_linear_zscored

    
    
# def prepare_ravia_backup():
#     # input_file = '/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_ravia2020_behavior_similairity.csv'
# #     pd.read_csv('/local_storage/datasets/farzaneh/openpom/data/curated_datasets/curated_ravia2020_alvaa.csv')
#     input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/ravia_molecules_alva_17Apr.csv'
#     df_ravia_original=pd.read_csv(input_file)
#     df_ravia=df_ravia_original.copy()
#     print(df_ravia.columns)
#     # 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
#        # 'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'
#
#     features= ['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity']
#     agg_functions={}
    
    
    # chemical_features_r=["nCIR",
    #                      "ZM1",
    #                      "GNar",
    #                      "S1K",
    #                      "piPC08",
    #                      "MATS1v",
    #                      "MATS7v",
    #                      "GATS1v",
    #                      "Eig05_AEA(bo)",
    #                      "SM02_AEA(bo)",
    #                      "SM03_AEA(dm)",
    #                      "SM10_AEA(dm)",
    #                      "SM13_AEA(dm)",
    #                       "SpMin3_Bh(v)",
    #                      "RDF035v",
    #                      "G1m",
    #                      "G1v",
    #                      "G1e",
    #                      "G3s",
    #                      "R8u+",
    #                      "nRCOSR"]

    
    nonStereoSMILE1 = list(map(lambda x: "Stimulus 1-nonStereoSMILES___" + x, chemical_features_r))
    nonStereoSMILE2 = list(map(lambda x: "Stimulus 2-nonStereoSMILES___" + x, chemical_features_r))
    IsomericSMILES1 = list(map(lambda x: "Stimulus 1-IsomericSMILES___" + x, chemical_features_r))
    IsomericSMILES2 = list(map(lambda x: "Stimulus 2-IsomericSMILES___" + x, chemical_features_r))
   
    chemical_features = nonStereoSMILE1+nonStereoSMILE2+IsomericSMILES1+IsomericSMILES2
    keys = chemical_features.copy()
    values = [chemical_aggregator]*len(chemical_features)

    # Create the dictionary using a dictionary comprehension
    agg_functions = {key: value for key, value in zip(keys, values)}        
        
    features_all = features + chemical_features

    df_ravia=df_ravia.reindex(columns=features_all)
        
    agg_functions['RatedSimilarity'] = 'mean'
    # print(agg_functions,"agg_functions")
    # print(features_all)
    
    
    df_ravia = df_ravia[ features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES', 'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES', 'CID Stimulus 1': 'CID Stimulus 2', 'CID Stimulus 2': 'CID Stimulus 1','Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_ravia_copy['RatedSimilarity']=np.nan
    df_ravia_concatenated= pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia=df_ravia_concatenated.drop_duplicates(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    
    # df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).mean().reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    
    df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()
    # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
    
    # result_df = df_ravia.groupby('category').agg(agg_functions)
    
    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values='RatedSimilarity')
    # df_ravia_mean_pivoted.head(5)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted=df_ravia_mean_pivoted.sort_index(ascending=True)
    
    
    return  df_ravia_original,df_ravia_mean,df_ravia_mean_pivoted





def prepare_ravia_or_snitz(dataset,base_path='/local_storage/datasets/farzaneh/alignment_olfaction_datasets/'):
    # generate docstrings for this function with a brief description of the function and the parameters and return values
    """
    Prepare the similarity dataset for the alignment task
    :param base_path:   (str) path to the base directory where the datasets are stored
    :return:         (tuple) a tuple containing the original  dataset, the mean  dataset, and the pivoted mean  dataset
    """

    input_file = base_path + dataset
    df_ravia_original = pd.read_csv(input_file)
    df_ravia = df_ravia_original.copy()

    features = ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
                'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity']
    agg_functions = {}
    features_all = features
    df_ravia = df_ravia.reindex(columns=features_all)

    agg_functions['RatedSimilarity'] = 'mean'

    df_ravia = df_ravia[features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES',
                                                  'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                                  'CID Stimulus 1': 'CID Stimulus 2',
                                                  'CID Stimulus 2': 'CID Stimulus 1',
                                                  'Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES',
                                                  'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_ravia_copy['RatedSimilarity'] = np.nan
    df_ravia_concatenated = pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia = df_ravia_concatenated.drop_duplicates(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    df_ravia_mean = df_ravia.groupby(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()

    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2',
                                                values='RatedSimilarity')

    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.sort_index(ascending=True)

    return df_ravia_original, df_ravia_mean, df_ravia_mean_pivoted

def extract_set_idxs(base_path, indices_path):
    input_file_indices = base_path + indices_path  # or new downloaded file path
    indices = pd.read_csv(input_file_indices)
    indices_train = indices.loc[indices['split'] == 'train']['main_idx'].values.tolist()
    indices_valid = indices.loc[indices['split'] == 'valid']['main_idx'].values.tolist()
    indices_test = indices.loc[indices['split'] == 'test']['main_idx'].values.tolist()
    return indices_train, indices_valid, indices_test

#extract dataframe from indices
def extract_set_from_indices_df(base_path, ds_path, indices_train, indices_valid, indices_test):
    input_file_pom = base_path + ds_path
    gs_lf_pom = pd.read_csv(input_file_pom)
    gs_lf_pom = gs_lf_pom.reset_index()
    gs_lf_pom = gs_lf_pom.rename(columns={'index': 'CID'})

    gs_lf_pom_train = gs_lf_pom.loc[gs_lf_pom['CID'].isin(indices_train)]
    gs_lf_pom_valid = gs_lf_pom.loc[gs_lf_pom['CID'].isin(indices_valid)]
    gs_lf_pom_test = gs_lf_pom.loc[gs_lf_pom['CID'].isin(indices_test)]
    return gs_lf_pom_train, gs_lf_pom_valid, gs_lf_pom_test


def extract_set_from_indices(base_path, ds_path,x_att,y_att, indices_train, indices_valid, indices_test):
    input_file_pom = base_path + ds_path
    gs_lf = pd.read_csv(input_file_pom)
    gs_lf = prepare_dataset(gs_lf, x_att, y_att)

    gs_lf_np = np.asarray(gs_lf[x_att].tolist())
    gs_lf_y = np.asarray(gs_lf[y_att].tolist())

    gs_lf_proba_train = gs_lf_np[indices_train]
    gs_lf_y_train = gs_lf_y[indices_train]

    gs_lf_proba_test = gs_lf_np[indices_test]
    gs_lf_y_test = gs_lf_y[indices_test]

    gs_lf_proba_valid = gs_lf_np[indices_valid]
    gs_lf_y_valid = gs_lf_y[indices_valid]





    return gs_lf, gs_lf_np,gs_lf_y,gs_lf_proba_train,gs_lf_y_train,gs_lf_proba_valid,gs_lf_y_valid,gs_lf_proba_test,gs_lf_y_test

def prepare_dataset(ds,x_att,y_att):
    ds[y_att] = ds[y_att].apply(ast.literal_eval)
    ds[x_att] = ds[x_att].apply(ast.literal_eval)
    return ds

# def prepare_ravia_sep():
#
#     input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/mols_datasets/curated_ravia2020_behavior_similairity.csv'
#     df_ravia_original=pd.read_csv(input_file)
#     df_ravia=df_ravia_original.copy()
#     print(df_ravia.columns)
#     # 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
#        # 'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'
#
#     features= ['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep', 'RatedSimilarity']
#     agg_functions={}
#
#     features_all = features
#     df_ravia=df_ravia.reindex(columns=features_all)
#
#     agg_functions['RatedSimilarity'] = 'mean'
#     # print(agg_functions,"agg_functions")
#     # print(features_all)
#
#
#     df_ravia = df_ravia[ features_all]
#     df_ravia_copy = df_ravia.copy()
#     df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES_sep': 'Stimulus 2-IsomericSMILES_sep', 'Stimulus 2-IsomericSMILES_sep': 'Stimulus 1-IsomericSMILES_sep', 'CID Stimulus 1': 'CID Stimulus 2', 'CID Stimulus 2': 'CID Stimulus 1','Stimulus 1-nonStereoSMILES_sep': 'Stimulus 2-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep': 'Stimulus 1-nonStereoSMILES_sep'})
#     df_ravia_copy['RatedSimilarity']=np.nan
#     df_ravia_concatenated= pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
#     df_ravia=df_ravia_concatenated.drop_duplicates(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep'])
#
#
#     # df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).mean().reset_index()
#     # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
#
#
#     df_ravia_mean =df_ravia.groupby(['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES_sep','Stimulus 2-IsomericSMILES_sep','Stimulus 1-nonStereoSMILES_sep', 'Stimulus 2-nonStereoSMILES_sep']).agg(agg_functions).reset_index()
#     # df_ravia_mean=df_ravia_mean.drop(columns=['CID Stimulus 1','CID Stimulus 2','Stimulus 1-IsomericSMILES','Stimulus 2-IsomericSMILES','Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])
#
#     # result_df = df_ravia.groupby('category').agg(agg_functions)
#
#     df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values='RatedSimilarity')
#     # df_ravia_mean_pivoted.head(5)
#     df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
#     df_ravia_mean_pivoted=df_ravia_mean_pivoted.sort_index(ascending=True)
#
#
#     return  df_ravia_original,df_ravia_mean,df_ravia_mean_pivoted

def prepare_ravia_similarity_mols_mix_on_smiles(df_ravia_similarity_mean, lm, tokenizer, modeldeepchem_gslf=None):
    # df_ravia_mean_mols1 = df_ravia_similarity_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    # df_ravia_mean_mols2 = df_ravia_similarity_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    # df_ravia_mols= pd.concat([df_ravia_mean_mols1, df_ravia_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    # df_ravia_mols=df_ravia_mols.drop_duplicates().reset_index(drop=True)
    # df_ravia_mols = df_ravia_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })

    df_ravia_mols = create_pairs(df_ravia_similarity_mean)

    res=prepare_mols_helper(lm,tokenizer,df_ravia_mols,modeldeepchem=modeldeepchem_gslf)

    df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored=res
    return df_ravia_mols,df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored

def sum_embeddings(cid_list, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
    return embedding_sum


def average_embeddings(cid_list, df_embeddings):
    print(cid_list)
    print(df_embeddings['CID'].values)

    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    n_cid = 0
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
            n_cid +=1
    return embedding_sum/n_cid

# def extract_embeddings(cid, df_embeddings):
#     embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
#     # for cid in cid_list:
#     if cid in df_embeddings['CID'].values:
#         embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
#     return embedding_sum



def prepare_ravia_similarity_mols_mix_on_representations(input_file_embeddings, df_ravia_similarity_mean, modeldeepchem_gslf=None,mixing_type='sum',sep=';',start='0',end='255'):
    df_ravia_mols = create_pairs(df_ravia_similarity_mean)

    df_embeddigs = pd.read_csv(input_file_embeddings)[['embeddings','CID']]
    df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))

    if mixing_type == 'sum':

        df_ravia_mols['Stimulus Embedding Sum'] = df_ravia_mols['CID'].apply(lambda x: sum_embeddings(list(map(int, x.split(sep))), df_embeddigs))
    elif mixing_type == 'average':
        df_ravia_mols['Stimulus Embedding Sum'] = df_ravia_mols['CID'].apply(lambda x: average_embeddings(list(map(int, x.split(sep))), df_embeddigs))

    df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_ravia_mols['Stimulus Embedding Sum'].values.tolist()))]

    df_ravia_mols_embeddings_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored=prepare_mols_helper_mixture(df_mols_embeddings_original,df_ravia_mols, start,end, modeldeepchem_gslf)
    
    return df_ravia_mols,df_ravia_mols_embeddings_original,df_ravia_mols_embeddings,df_ravia_mols_embeddings_zscored


def create_pairs(df_ravia_similarity_mean):
    df_ravia_mean_mols1 = df_ravia_similarity_mean[
        ['Stimulus 1-IsomericSMILES', 'Stimulus 1-nonStereoSMILES', 'CID Stimulus 1']].drop_duplicates().reset_index(
        drop=True)
    df_ravia_mean_mols2 = df_ravia_similarity_mean[
        ['Stimulus 2-IsomericSMILES', 'Stimulus 2-nonStereoSMILES', 'CID Stimulus 2']].drop_duplicates().reset_index(
        drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES',
                                   'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                   'CID Stimulus 2': 'CID Stimulus 1'})
    df_ravia_mols = pd.concat([df_ravia_mean_mols1, df_ravia_mean_mols2], ignore_index=True, axis=0).reset_index(
        drop=True)
    df_ravia_mols = df_ravia_mols.drop_duplicates().reset_index(drop=True)
    df_ravia_mols = df_ravia_mols.rename(
        columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES', 'Stimulus 1-nonStereoSMILES': 'nonStereoSMILES',
                 'CID Stimulus 1': 'CID'})
    return df_ravia_mols



def prepare_sagar():
    
    input_file_sagar = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/alva/sagar_molecules_alva_17Apr.csv'
    df_sagar=pd.read_csv(input_file_sagar)
    df_sagar = df_sagar.rename(columns={"cid":"CID"})
    
    columns_list = ['Intensity', 'Pleasantness', 'Fishy', 'Burnt', 'Sour', 'Decayed',
       'Musky', 'Fruity', 'Sweaty', 'Cool', 'Chemical', 'Floral', 'Sweet',
       'Warm', 'Bakery', 'Garlic', 'Spicy', 'Acidic', 'Ammonia', 'Edible','Familiar']
    
    
    df_sagar_common=df_sagar.copy()
    
      # Specify your list of columns

    # Find columns with NaN values
    columns_with_nan = df_sagar_common.columns[df_sagar_common.isna().any()].tolist()
    
    # Find columns that are both in the list and contain NaN values
    columns_to_drop = list(set(columns_list) & set(columns_with_nan))
    
    # Drop columns from DataFrame
    df_sagar_common = df_sagar_common.drop(columns=columns_to_drop)
        
    
    
    
    
    # df_sagar = df_sagar.dropna(axis=1)
    
    # df_sagar_mean =df_sagar.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    df_sagar_mean =df_sagar.groupby(['IsomericSMILES','nonStereoSMILES']).apply(lambda x: x.iloc[:, :-4].mean()).reset_index()
    
    
    df_sagar_mean['Combined'] = df_sagar_mean.loc[:, columns_list].values.tolist()
    df_sagar['Combined'] = df_sagar.loc[:, columns_list].values.tolist()
    # return df_sagar_mean
    
#     #Z-score sagar dataset
    df_sagar_zscored = df_sagar_mean.copy()
    df_sagar_zscored=df_sagar_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_sagar_zscored.loc[:,columns_list].values.tolist())
    df_sagar_zscored.loc[:, columns_list] = pd.DataFrame(scaled_features, index=df_sagar_zscored.index, columns=columns_list)
    
#     #Mean over z-score sagar
    df_sagar_zscored_mean =df_sagar_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_sagar_zscored['Combined'] = df_sagar_zscored.loc[:, columns_list].values.tolist()
    df_sagar_zscored_mean['Combined'] = df_sagar_zscored_mean.loc[:, columns_list].values.tolist()
    
    
    #PCA on z-scored sagar
    df_sagar_zscored_cid_combined = df_sagar_zscored[['CID', 'Combined']]
    # df_sagar_zscored_pca=PCA_df(df_sagar_zscored_cid_combined,'Combined' )
    
    #PCA on z-scored_mean sagar
    df_sagar_zscored_mean_cid_combined = df_sagar_zscored_mean[['CID', 'Combined']]
    # df_sagar_zscored_mean_pca=PCA_df(df_sagar_zscored_mean_cid_combined,'Combined',n_components=n_components )
    
    #Mean on z_scored_PCA
    # df_sagar_zscored_pca_mean=df_sagar_zscored_pca.drop('Combined',axis=1)
    # df_sagar_zscored_pca_mean =df_sagar_zscored_pca_mean.groupby(['CID']).mean().reset_index()
    
    # df_mean_reduced_sagar_zscored_cid_combined =df_sagar_zscored_pca.groupby(['CID']).mean().reset_index()
#     df_sagar_zscored_pca_mean['Combined']=df_sagar_zscored_pca_mean.loc[:, 0:n_components-1].values.tolist()
#     df_sagar_zscored_pca_mean=df_sagar_zscored_pca_mean.drop([0,1,2,3,4],axis=1)
    
    
    # return df_sagar, df_sagar_mean, df_sagar_zscored, df_sagar_zscored_mean, df_sagar_zscored_pca,df_sagar_zscored_mean_pca,df_sagar_zscored_pca_mean
    
    

    df_sagar_common_mean =df_sagar_common.groupby(['IsomericSMILES','nonStereoSMILES']).apply(lambda x: x.iloc[:, :-4].mean()).reset_index()
    
    
    
    
    columns_list_common = ['Intensity', 'Pleasantness', 'Fishy', 'Burnt', 'Sour', 'Decayed',
       'Musky', 'Fruity', 'Sweaty', 'Cool', 'Floral', 'Sweet',
       'Warm', 'Bakery', 'Spicy']
    
    df_sagar_common_mean['Combined'] = df_sagar_common_mean.loc[:, columns_list_common].values.tolist()
    df_sagar_common['Combined'] = df_sagar_common.loc[:, columns_list_common].values.tolist()
    # return df_sagar_mean
    
#     #Z-score sagar dataset
    df_sagar_common_zscored = df_sagar_common_mean.copy()
    df_sagar_common_zscored=df_sagar_common_zscored.drop('Combined',axis=1)
    scaled_features = StandardScaler().fit_transform(df_sagar_common_zscored.loc[:,columns_list_common].values.tolist())
    df_sagar_common_zscored.loc[:, columns_list_common] = pd.DataFrame(scaled_features, index=df_sagar_common_zscored.index, columns=columns_list_common)
    
#     #Mean over z-score sagar
    df_sagar_common_zscored_mean =df_sagar_common_zscored.groupby(['IsomericSMILES','nonStereoSMILES']).mean().reset_index()
    
    #combine columns
    df_sagar_common_zscored['Combined'] = df_sagar_common_zscored.loc[:, columns_list_common].values.tolist()
    df_sagar_common_zscored_mean['Combined'] = df_sagar_common_zscored_mean.loc[:, columns_list_common].values.tolist()
    
    
    

    return df_sagar, df_sagar_mean, df_sagar_zscored, df_sagar_zscored_mean, df_sagar_common,df_sagar_common_mean, df_sagar_common_zscored,df_sagar_common_zscored_mean


# def prepare_sagar_mols(modeldeepchem_gslf,lm,tokenizer):
#     # df_sagar=df_sagar.rename(columns={"cid":"CID"})
#     df_sagar_mols = df_sagar.drop_duplicates("CID")
#     print(df_sagar_mols.columns)
#     df_sagar_mols_embeddings_original,df_sagar_mols_layers_original,df_sagar_mols_embeddings,df_sagar_mols_embeddings_zscored,df_sagar_mols_layers,df_sagar_mols_layers_zscored,df_sagar_mols_embeddings_linear,df_sagar_mols_embeddings_linear_zscored=prepare_mols_helper(lm,tokenizer,df_sagar_mols,mol_type="nonStereoSMILES",modeldeepchem=modeldeepchem_gslf)
#     return df_sagar_mols,df_sagar_mols_embeddings_original,df_sagar_mols_layers_original,df_sagar_mols_embeddings,df_sagar_mols_embeddings_zscored,df_sagar_mols_layers,df_sagar_mols_layers_zscored,df_sagar_mols_embeddings_linear,df_sagar_mols_embeddings_linear_zscored
#





def prepare_snitz_mols(df_snitz_mean,modeldeepchem_gslf,lm,tokenizer):
    df_snitz_mean_mols1 = df_snitz_mean[['Stimulus 1-IsomericSMILES','Stimulus 1-nonStereoSMILES','CID Stimulus 1']].drop_duplicates().reset_index(drop=True)
    df_snitz_mean_mols2 = df_snitz_mean[['Stimulus 2-IsomericSMILES','Stimulus 2-nonStereoSMILES','CID Stimulus 2']].drop_duplicates().reset_index(drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES','Stimulus 2-IsomericSMILES':'Stimulus 1-IsomericSMILES', 'CID Stimulus 2': 'CID Stimulus 1' })
    df_snitz_mols= pd.concat([df_snitz_mean_mols1, df_snitz_mean_mols2], ignore_index=True, axis=0).reset_index(drop=True)
    df_snitz_mols = df_snitz_mols.rename(columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES','Stimulus 1-nonStereoSMILES':'nonStereoSMILES', 'CID Stimulus 1': 'CID' })

    df_snitz_mols=df_snitz_mols.drop_duplicates().reset_index(drop=True)
    # df_snitz_mols.to_csv('df_snitz_mols.csv')  
    # mol_type="nonStereoSMILES"
    
    
    df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,\
    df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,\
    df_snitz_mols_layers_zscored=prepare_mols_helper(lm,tokenizer,df_snitz_mols,modeldeepchem=modeldeepchem_gslf)
        
    
    return df_snitz_mols,df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored

def select_features(input_file):
    ds_alva = pd.read_csv(input_file)
    nonStereoSMILE = list(map(lambda x: "nonStereoSMILES___" + x, chemical_features_r))
    # IsomericSMILES = list(map(lambda x: "IsomericSMILES___" + x, chemical_features_r))
    selected_features = nonStereoSMILE
    features= ['CID','nonStereoSMILES']+selected_features
    ds_alva= ds_alva.rename(columns={"cid":"CID"})
    ds_alva_selected = ds_alva[features]
    # ds_alva_selected = ds_alva_selected.fillna(0)
    #drop columns with all na values
    ds_alva_selected = ds_alva_selected.dropna(axis=1, how='all')
    ds_alva_selected = ds_alva_selected.fillna(0)
    print(ds_alva_selected.shape)

    ds_alva_selected['embeddings'] = ds_alva_selected[selected_features].values.tolist()
    return ds_alva_selected

# def prepare_mols_other(input_file_embeddings, df_mean,modeldeepchem_gslf):
#     df_mols = create_pairs(df_mean)
#
#     df_embeddigs = pd.read_csv(input_file_embeddings)[['embeddings','CID']]
#     df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))
#
#
#     df_mols['Stimulus Embedding Sum'] = df_mols['CID'].apply(lambda x: sum_embeddings(list(map(int, x.split(','))), df_embeddigs))
#     df_mols_embeddings_original =[torch.from_numpy(np.asarray(df_mols['Stimulus Embedding Sum'].values.tolist()))]
#
#     df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored=prepare_mols_helper_mixture(df_mols_embeddings_original,df_mols)
#
#     return df_mols,df_mols_embeddings_original,df_mols_embeddings,df_mols_embeddings_zscored




def prepare_goodscentleffignwell_mols(modeldeepchem_gslf,lm,tokenizer):
    goodscentleffignwell_input_file = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/curated_datasets/mols_datasets/curated_GS_LF_merged_4983.csv' # or new downloaded file path
    df_goodscentleffignwell=pd.read_csv(goodscentleffignwell_input_file)
    df_goodscentleffignwell.index.names = ['CID']
    df_goodscentleffignwell= df_goodscentleffignwell.reset_index()
    df_goodscentleffignwell['y'] = df_goodscentleffignwell.loc[:,'alcoholic':'woody'].values.tolist()
    df_gslf_mols_embeddings_original,df_gslf_mols_layers_original,df_gslf_mols_embeddings,df_gslf_mols_embeddings_zscored,df_gslf_mols_layers,df_gslf_mols_layers_zscored=prepare_mols_helper(lm,tokenizer,df_goodscentleffignwell,modeldeepchem=modeldeepchem_gslf)
    return df_goodscentleffignwell, df_gslf_mols_embeddings_original,df_gslf_mols_layers_original,df_gslf_mols_embeddings,df_gslf_mols_embeddings_zscored,df_gslf_mols_layers,df_gslf_mols_layers_zscored


    
#     return df_snitz_mols,df_snitz_mols_embeddings_original,df_snitz_mols_layers_original,df_snitz_mols_embeddings,df_snitz_mols_embeddings_zscored,df_snitz_mols_layers,df_snitz_mols_layers_zscored

