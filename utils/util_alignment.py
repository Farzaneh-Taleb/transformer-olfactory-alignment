import os
conda_env = os.environ.get('CONDA_DEFAULT_ENV')

if conda_env== 'Mol':
    from fast_transformers.masking import LengthMask as LM
if conda_env== 'Mol' or conda_env== 'open_pom':
    import deepchem as dc
    import torch

print(conda_env)

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.stats import pearsonr  
from sklearn.metrics import r2_score
import random
from constants import *
def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size


#Freche distance
# __all__ = ['frdist']
def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]
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



def cosine_similarity_df(df,col_name):
    """
    cosine_similarity_df function calculated cosine_similarity  for df
    :param p1: df is a dataframe with 'Combined' column as a column which all the features are combined in a list.
    It does have one entry per CID. It also contains a 'CID' column 
    :return: a square dataframe which is pair-wise cosine similarity for each pair of cids.
    """
    df = df.dropna(subset=[col_name])
    df_cid_combined = df[['CID', col_name]]
    list_cid_combined = df_cid_combined[col_name].to_list()
    df_cosine_sim_matrix = cosine_similarity(list_cid_combined)
    df_cosine_sim_df = pd.DataFrame(df_cosine_sim_matrix, index=df_cid_combined['CID'], columns=df_cid_combined['CID'])
    df_cosine_sim_df = df_cosine_sim_df.reindex(sorted(df_cosine_sim_df.columns), axis=1)
    df_cosine_sim_df=df_cosine_sim_df.sort_index(ascending=True)
    return df_cosine_sim_df






def cosine_sim_helper(df_mols_embeddings, df_mols_embeddings_zscored):
    
    cosine_sim_df_mols_embeddings=cosine_similarity_df(df_mols_embeddings,'Combined')
    cosine_sim_df_mols_embeddings_zscored=cosine_similarity_df(df_mols_embeddings_zscored,'Combined')
    return cosine_sim_df_mols_embeddings, cosine_sim_df_mols_embeddings_zscored



def correlation_helper_mixture(df_all,df_mols_all,value_type="r"):
    # layers=[]
    # layers_pvalue=[]
    data_flattered=flattening_data_helper(df_all,df_mols_all,equalize_size=True)
    
    
    if value_type=="R2":
        last=r2_score(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
    else:       
        last=pearsonr(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
    if value_type=="R2":
        return last
    else:
        print("change",abs(last.statistic))
        return abs(last.statistic),last.pvalue


def flattening_data_helper(out_original,out_mols,equalize_size=True):

    out_mols=out_mols.to_numpy().flatten()
    out=out_original.to_numpy().flatten()
    if equalize_size:
        to_be_deleted=np.argwhere(out!=out).flatten().tolist()
        out=np.delete(out,to_be_deleted)
        out_mols=np.delete(out_mols,to_be_deleted)
        
    data = {"Peceptual Similarity": out, "Model Similarity": out_mols}
    return data






def extract_molformer_representations(lm, tokenizer, Tasks, input_file, smiles_field):
    featurizer = dc.feat.DummyFeaturizer()
    loader = dc.data.CSVLoader(tasks=Tasks,
                               feature_field=smiles_field,
                               featurizer=featurizer
                               )
    dataset = loader.create_dataset(inputs=[input_file])
    embeddings_original, activations_embeddings_original = embed(lm, dataset.X, tokenizer, batch_size=64)
    embeddings_original = torch.cat(embeddings_original).numpy()
    X = torch.from_numpy(embeddings_original)
    if len(Tasks) != 0:
        y = dataset.y
    else:
        y = None
    X_layers = []
    y_layers = []
    for df_mols_layer in activations_embeddings_original:
        embeddings_original = torch.cat(df_mols_layer).numpy()
        X = torch.from_numpy(embeddings_original)
        X_layers.append(X)
        if len(Tasks) != 0:
            # y=torch.from_numpy(y)
            y_layers.append(y)
        else:
            y_layers.append(None)
    return X, y, X_layers, y_layers


def compute_statistics(df_ravia_similarity_mols):
    ravia_mean=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).mean()
    ravia_std=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).std()
    ravia_max=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).max()
    ravia_min=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).min()
    print(ravia_min, ravia_max, ravia_mean, ravia_std)


def set_seeds(seed):
    if conda_env=='Mol' or conda_env=='open_pom':
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def grand_average(df, ds):
    if ds == "keller":
        descriptors = keller_descriptors
    elif ds == "keller2":
        descriptors = keller_descriptors2

    elif ds == "sagar":
        descriptors = sagar_descriptors

    elif ds == "sagar2":
        descriptors = sagar_descriptors2
    else:
        raise ValueError("Invalid dataset")

    df_groupbyCID = df.groupby('CID')[descriptors].mean().reset_index()

    df_groupbyCID['y'] = df_groupbyCID.loc[:, descriptors[0]:descriptors[-1]].values.tolist()
    df_embeddings = df.drop_duplicates(subset=['CID'])
    df_embeddings = df_embeddings[['CID', 'embeddings']]
    df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
    return df_groupbyCID



def average_over_subject(df, ds):
    if ds == "keller":
        descriptors = keller_descriptors
    elif ds == "keller2":
        descriptors = keller_descriptors2

    elif ds == "sagar":
        descriptors = sagar_descriptors

    elif ds == "sagar2":
        descriptors = sagar_descriptors2

    else:
        raise ValueError("Invalid dataset")

    df_groupbyCID = df.groupby(['CID', 'subject'])[descriptors].mean().reset_index()

    df_groupbyCID['y'] = df_groupbyCID.loc[:, descriptors[0]:descriptors[-1]].values.tolist()
    df_embeddings = df.drop_duplicates(subset=['CID'])
    df_embeddings = df_embeddings[['CID', 'embeddings']]
    df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
    return df_groupbyCID


def post_process_results_df(mserrorrs_corssvalidated, correlations_corssvalidated):
    mserrorrs_corssvalidated_array = np.asarray(mserrorrs_corssvalidated)
    if len(mserrorrs_corssvalidated_array.shape) == 3:
        mserrorrs_corssvalidated_array = np.squeeze(mserrorrs_corssvalidated_array, -1)
        mserrorrs_corssvalidated_array = np.moveaxis(mserrorrs_corssvalidated_array, 0, 1)
    # print(mserrorrs_corssvalidated_array.shape,"shapeeee1")

    correlations_corssvalidated = np.asarray(correlations_corssvalidated)
    if len(correlations_corssvalidated.shape) == 4:
        correlations_corssvalidated = np.moveaxis(correlations_corssvalidated, 0, 1)
        # print("correlations_corssvalidateds",correlations_corssvalidated.shape)
        correlations_corssvalidated = np.squeeze(correlations_corssvalidated, 2)
    # print(correlations_corssvalidated.shape,"shapeeee2")
    statistics_correlations_corssvalidated_array = correlations_corssvalidated[:, :, 0]
    pvalues_correlations_corssvalidated_array = correlations_corssvalidated[:, :, 1]

    return mserrorrs_corssvalidated_array, statistics_correlations_corssvalidated_array, pvalues_correlations_corssvalidated_array

