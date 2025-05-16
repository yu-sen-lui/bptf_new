# handling parameter arguments ====================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_components_combined",
    type=int,
    default=150,
    help="number of combined components"
)
args = parser.parse_args()

# packages ========================================================================================

import bptf
from own_implementation import BPTF
import numpy as np
import pandas as pd
import sparse
import os
import shutil
from tqdm import tqdm
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
import torch

import tensorly
import cupy

import multiprocessing
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map

print(os.getcwd())

import gc
gc.collect()

# Global variables and settings ===================================================================
parallel = False
tol = 1e-6
max_iter = 1000
device = 'cuda'
end_year = 18
if parallel:
    print(multiprocessing.cpu_count())

n_components = {
    "combined": args.n_components_combined,
    "icews": int(round(args.n_components_combined / 3)),
    "gdelt": int(round(args.n_components_combined / 3)),
    "terrier": int(round(args.n_components_combined / 3)),
}

print(f'Components for each model: {n_components}')

def dataframe_to_sparse_tensor(data, country_indices, date_indices, database_indices, cameo_col='CAMEO_Code', events_col='Num_Events'):
    """
    Converts a pandas DataFrame into an sptensor.
    
    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        country_indices (pd.DataFrame): DataFrame with country indices.
        date_indices (pd.DataFrame): DataFrame with date indices.
        database_indices (pd.DataFrame): DataFrame with database indices.
        cameo_col (str): Name of the CAMEO code column in 'data'. Defaults to 'CAMEO_Code'.
        events_col (str): Name of the column containing event counts in 'data'. Defaults to 'Num_Events'.
    
    Returns:
        sptensor: The resulting sparse tensor.
    """
    
    # Define the shape of the tensor (V, V, A, T, D)
    V = len(country_indices)
    A = 20  # Assuming the CAMEO code has 20 distinct values
    T = len(date_indices)
    D = len(database_indices)
    
    shape = (V, V, A, T, D)
    
    # Initialize empty subs and vals
    subs = ([], [], [], [], [])
    vals = []

    # Iterate through the DataFrame to populate the tensor
    for i in range(len(data)):

        source_country = country_indices.loc[country_indices['country'] == data['Source_Country_Code'].iloc[i]]
        source_country_index = int(source_country.iloc[0, 1])

        target_country = country_indices.loc[country_indices['country'] == data['Target_Country_Code'].iloc[i]]
        target_country_index = int(target_country.iloc[0, 1])

        action_index = int(data[cameo_col].iloc[i] - 1)  # Adjust CAMEO code to 0-based index

        date = date_indices.loc[date_indices['date'] == data['formatteddate'].iloc[i]]
        date_index = int(date.iloc[0, 1])

        database = database_indices.loc[database_indices['database'] == data['Database'].iloc[i]]
        database_index = int(database.iloc[0, 1])

        # Append indices and values
        subs[0].append(source_country_index)
        subs[1].append(target_country_index)
        subs[2].append(action_index)
        subs[3].append(date_index)
        subs[4].append(database_index)
        vals = np.append(vals, data[events_col].iloc[i])

    # Convert subs to a tuple of numpy arrays and vals to a numpy array
    subs = tuple(np.array(s, dtype=int) for s in subs)
    vals = np.array(vals)

    # Create the sparse COO tensor
    Y = sparse.COO(coords=subs, data=vals, shape=shape)
    
    return Y

def add_sparse_tensors(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape to be added."
    
    # Directly add the two tensors
    result_tensor = tensor1 + tensor2
    
    return result_tensor


def sum_sparse_tensor_list(tensor_list):
    if not tensor_list:
        raise ValueError("The list of tensors is empty.")
    
    # Use sum with a starting tensor of 0 in the shape of the tensors
    result_tensor = sum(tensor_list[1:], start=tensor_list[0])
    
    return result_tensor

def component_analysis_plot(component, path_to_save, entropy_rank = None, database = None):
    """
    Plots bptf factor vectors
    Args:
        component: column of factor matrix
        entropy_rank: if ranking by entropy, use this. You can leave this as a blank str
        path_to_save: path to save png plot, relative to working directory
        database: name of database if your bptf was fit to only 1 database
    """

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    time_vector = G_DK_M[3][:, component]
    time_vector = pd.DataFrame({
        'time steps' : time_vector,
        'index' : range(time_vector.shape[0])
    })
    time_vector = pd.merge(
        left=time_vector, right=date_indices,
        how='left', on='index'
    )
    ax1.plot(time_vector['date'], time_vector['time steps'], color='b')
    ticks = ax1.get_xticks()[::12]
    ax1.set_xticks(ticks)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_title('Time steps')

    ax2 = fig.add_subplot(gs[1, 0])
    sender_vector = G_DK_M[0][:, component]
    sender_vector = pd.DataFrame({
        'sender' : sender_vector,
        'index' : range(sender_vector.shape[0])})
    sender_vector = pd.merge(
        left=sender_vector, right=country_indices,
        how='left', on='index'
    ).sort_values(
        by='sender', 
        ascending=False).iloc[:10, :]
    sender_vector['country'] = sender_vector['country'].astype(str)
    ax2.vlines(x=sender_vector['country'], ymin=0, ymax=sender_vector['sender'], color='r')
    ax2.set_title('Sender')

    ax3 = fig.add_subplot(gs[1, 1])
    receiver_vector = G_DK_M[1][:, component]
    receiver_vector = pd.DataFrame({
        'receiver' : receiver_vector,
        'index' : range(receiver_vector.shape[0])})
    receiver_vector = pd.merge(
        left=receiver_vector, right=country_indices,
        how='left', on='index'
    ).sort_values(
        by='receiver', ascending=False
    ).iloc[:10, :]
    ax3.vlines(x=receiver_vector['country'], ymin=0, ymax=receiver_vector['receiver'], color='g')
    ax3.set_title('Receiver')

    ax4 = fig.add_subplot(gs[2, 0]) if database is None else fig.add_subplot(gs[2, :])
    action_vector = G_DK_M[2][:, component]
    action_vector = pd.DataFrame({
        'action' : action_vector,
        'action type' : range(1, 21)
    })
    ax4.vlines(x=action_vector['action type'], ymin=0, ymax=action_vector['action'], color='black')
    ax4.set_title('Action types')
    
    if database is None:
        ax5 = fig.add_subplot(gs[2, 1])
        database_vector = G_DK_M[4][:, component]
        database_vector = pd.DataFrame({
            'factor' : database_vector,
            'index' : range(database_vector.shape[0])
        })
        database_vector = pd.merge(
            left=database_vector, right=database_indices,
            how='left', on='index'
        )
        ax5.vlines(x=database_vector['database'], ymin=0, ymax=database_vector['factor'], color='purple')
        ax5.set_title('Database factors')

    if database is None:
        fig.suptitle(f"Entropy rank {entropy_rank}. Component {component}")
    else:
        fig.suptitle(f"Database {database}. Component {component}")

    plt.savefig(path_to_save)
    plt.close()

# folder management ===============================================================================

model_list = ['combined', 'icews', 'gdelt', 'terrier']

folder_path = f"running_data_separately_and_combined_plots_{n_components['combined']}_components"

for model_name in model_list:
    plot_folder_path = os.path.join(folder_path, model_name)
    if os.path.isdir(plot_folder_path):
        print(f'{plot_folder_path} exists')
    else:
        os.makedirs(plot_folder_path)
        print(f'{plot_folder_path} created')
    for entry in tqdm(os.listdir(plot_folder_path), desc=f'Deleting existing plot pngs for {model_name}'):
        path = os.path.join(plot_folder_path, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

# get data ========================================================================================
folder = "/icews_gdelt_terrier/"
data_filepath = os.getcwd() + folder
files = os.listdir(data_filepath)
data = [pd.read_csv(data_filepath + filepath) for filepath in tqdm(files, desc = 'Reading data')]
data = pd.concat(data)
data['formatteddate'] = pd.to_datetime(data['formatteddate'], format='%Y-%m-%d')
cutoff = pd.to_datetime('2015-02-18')
mask = ~((data['Database'] == 'GDELT') & (data['formatteddate'] <= cutoff))
data = data.loc[mask]
data['formatteddate'] = data['formatteddate'].dt.strftime('%Y-%m-%d')

def read_gdeltv1(gdelt_filepath):
    gdelt_chunk = pd.read_csv(
        gdelt_filepath, dtype={
            'Actor1CountryCode': str,
            'Actor2countryCode': str,
            'EventRootCode': np.int64,
            'SQLDATE': str,
            'NumMentions': np.int64
        },
        na_values='--'
    )
    gdelt_chunk = gdelt_chunk.rename(
        columns={
            'Actor1CountryCode' : 'Source_Country_Code',
            'Actor2countryCode' : 'Target_Country_Code',
            'EventRootCode' : 'CAMEO_Code',
            'SQLDATE' : 'formatteddate',
            'NumMentions' : 'Num_Events'
        }
    )
    gdelt_chunk['formatteddate'] = pd.to_datetime(gdelt_chunk['formatteddate'].astype(str), format='%Y%m%d')
    gdelt_chunk['formatteddate'] = gdelt_chunk['formatteddate'].dt.strftime('%Y-%m-%d')

    return gdelt_chunk

gdelt_filepaths = [os.path.join('gdeltv1', f) for f in os.listdir('gdeltv1')]
gdelt1 = [read_gdeltv1(gdelt_filepath) for gdelt_filepath in tqdm(gdelt_filepaths, desc='Reading GDELTv1')]
gdelt1 = pd.concat(gdelt1)
gdelt1['Database'] = 'GDELT'

data = pd.concat([data, gdelt1])

del gdelt1

years = [str(2000 + x) for x in range(0, end_year)]
data = data[data['formatteddate'].str.startswith(tuple(years))]
data = data.dropna()

data['formatteddate'] = data['formatteddate'].str[:7]
data = data.groupby(['Source_Country_Code', 'Target_Country_Code', 'CAMEO_Code', 'formatteddate', 'Database'])
data = data.sum().reset_index()
data.sort_values(by=['Source_Country_Code', 'Target_Country_Code', 'Database', 'CAMEO_Code', 'formatteddate'], ascending=False, inplace=True)

# Make the list of countries, actions, times and databases ========================================
# get unique list of countries
countries = pd.concat([data['Source_Country_Code'], data['Target_Country_Code']])
countries = pd.unique(countries)
country_indices = pd.DataFrame({
    'country' : countries,
    'index' : range(len(countries))
})
del countries

# there's no need to get a list of actions since it's just 1 to 20

# get unique list of dates
# Make sure to change the date format and frequency of the date range if you change the time unit
date_indices = pd.date_range(
    start=pd.to_datetime(data['formatteddate'], format='%Y-%m').min().strftime('%Y-%m'), 
    end=pd.to_datetime(data['formatteddate'], format='%Y-%m').max().strftime('%Y-%m'),
    freq='MS')
date_indices = date_indices.strftime('%Y-%m').to_list()
date_indices = pd.DataFrame({
    'date' : date_indices,
    'index' : range(len(date_indices))
})
date_indices['date'] = date_indices['date'].str[:7]

# get unique list of databases
databases = pd.unique(data['Database'])
databases = pd.unique(databases)
database_indices = pd.DataFrame({
    'database' : databases,
    'index' : range(len(databases))
})
print(database_indices)
del databases

# action names
actions = [
    'Make public statement', 'Appeal', 'Express intent to cooperate', 'Consult', 'Engage in diplomatic cooperation',
    'Engage in material cooperation', 'Provide aid', 'Yield', 'Investigate', 'Demand',
    'Disapprove', 'Reject', 'Threaten', 'Protest', 'Exhibit military posture',
    'Reduce relations', 'Coerce', 'Assault', 'Fight', 'Engage in unconventional mass violence'
]
action_indices = pd.DataFrame({
    'action' : actions,
    'index' : range(len(actions))
})

# Convert to sparse tensor ========================================================================
n_batches = 1000
n = 10

data = np.array_split(data, n_batches)

if parallel:
    num_cores = multiprocessing.cpu_count()

    if __name__ == '__main__':
        data = Parallel(n_jobs = num_cores)(
            delayed(dataframe_to_sparse_tensor)(batch, country_indices, date_indices, database_indices)
            for batch in tqdm(data, desc='Converting to sparse tensors')
        )
else:
    data = [dataframe_to_sparse_tensor(batch, country_indices, date_indices, database_indices) for batch in tqdm(data, desc='Converting to sparse tensors')]

while len(data) > 10:
    data = [data[i:i + n] for i in range(0, len(data), n)]
    data = [sum(batch) for batch in data]
Y = sum(data)
del data

# Fit BPTF ========================================================================================
# run for combined
database_mapping = dict(zip(database_indices['database'], database_indices['index']))
for model_name in model_list:
    print(f'Fitting for {model_name}')
    if model_name == 'combined': 
        Y_ = Y.todense()
        diagonal_indices = np.arange(0, Y_.shape[0])
        mask = np.ones(Y_.shape)
        mask[diagonal_indices, diagonal_indices, :, :, :] = 1
        mask = mask.astype(np.int64)
        mask = torch.tensor(mask, dtype=torch.float64, device=device)

        Y_ = torch.tensor(Y_, dtype=torch.float64, device = device)
    else:
        database_index = database_mapping[model_name.upper()]
        Y_ = Y[:, :, :, :, database_index].todense()
        diagonal_indices = np.arange(0, Y_.shape[0])
        mask = np.ones(Y_.shape)
        mask[diagonal_indices, diagonal_indices, :, :] = 1
        mask = mask.astype(np.int64)
        mask = torch.tensor(mask, dtype=torch.float64, device=device)

        Y_ = torch.tensor(Y_, dtype=torch.float64, device = device)

    bptf_model = BPTF(data_shape=Y_.shape, n_components=n_components[model_name], device=device)
    bptf_model.fit(Y_, mask=mask, max_iter = max_iter, tol=tol, verbose=True)

    # plot for combined
    G_DK_M = [factor_matrix.cpu().numpy() for factor_matrix in bptf_model.G_DK_M]
    G_DK_M = [factor_matrix / factor_matrix.sum(axis=0) for factor_matrix in G_DK_M]
    if model_name == 'combined':
        database_factor_matrix = G_DK_M[4]
        database_factor_matrix = database_factor_matrix/database_factor_matrix.sum(axis=0,keepdims=1)
        database_entropy = st.entropy(database_factor_matrix, axis=0)
        database_components = pd.DataFrame({
            'ICEWS' : database_factor_matrix[0, :],
            'TERRIER' : database_factor_matrix[1, :],
            'GDELT' : database_factor_matrix[2, :],
            'entropy' : database_entropy,
            'index' : range(n_components['combined'])
        }).sort_values(by='entropy', ascending=False)

        entropy_rank = 1
        for component in tqdm(database_components['index'], desc=f'Plotting components for {model_name}'):
            path_to_save_plot = os.path.join(os.getcwd(), folder_path, model_name, f"entropy_rank_{entropy_rank}_component_{component}.png")
            component_analysis_plot(component, path_to_save_plot, entropy_rank)
            entropy_rank += 1
    else:
        for component in tqdm(list(range(n_components[model_name])), desc=f'Plotting components for {model_name}'):
            path_to_save_plot = os.path.join(os.getcwd(), folder_path, model_name, f"model_name_{model_name}_component_{component}.png")
            component_analysis_plot(component, path_to_save_plot, entropy_rank = None, database = model_name)