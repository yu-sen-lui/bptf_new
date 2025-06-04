# packages ========================================================================================
import pandas as pd
import torch
import sparse
from tqdm import tqdm
import os
import pickle
from itertools import permutations
import numpy as np

# handle folders ==================================================================================
csv_filepath = 'results_analysis.csv'

# helper function(s) ==============================================================================
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

def get_counts(data, country_list, year_list):
    """
    Gets the event counts for all country pairings possible in country_list from data
    """
    country_pairings = list(permutations(country_list, 2))
    data = data.copy()
    data['formatteddate'] = pd.to_datetime(data['formatteddate'], format='%Y-%m')
    mask = data['formatteddate'].dt.year.isin(year_list)
    data = data.loc[mask]
    event_count = 0
    for country_pairing in country_pairings:
        sender, receiver = country_pairing
        country_pairing_data = data[
            (data['Source_Country_Code'] == sender) &
            (data['Target_Country_Code'] == receiver)
        ]
        event_count += country_pairing_data['Num_Events'].sum()

    return event_count

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

gdelt_filepaths = [os.path.join('gdeltv1', f) for f in os.listdir('gdeltv1')]
gdelt1 = [read_gdeltv1(gdelt_filepath) for gdelt_filepath in tqdm(gdelt_filepaths, desc='Reading GDELTv1')]
gdelt1 = pd.concat(gdelt1)
gdelt1['Database'] = 'GDELT'

data = pd.concat([data, gdelt1])

del gdelt1

years = [str(2000 + x) for x in range(0, 25)]
data = data[data['formatteddate'].str.startswith(tuple(years))]
data = data.dropna()

data['formatteddate'] = data['formatteddate'].str[:7]
data = data.groupby(['Source_Country_Code', 'Target_Country_Code', 'CAMEO_Code', 'formatteddate', 'Database'])
data = data.sum().reset_index()
data.sort_values(by=['Source_Country_Code', 'Target_Country_Code', 'Database', 'CAMEO_Code', 'formatteddate'], ascending=False, inplace=True)

# list out countries for each named component =====================================================
ranks = []; components = []; countries = []; years = []; actions = []; counts = []

# entropy_rank_1_component_127_IND_PAK_standoff
ranks.append(1); components.append(127); countries.append(['IND', 'PAK'])
year_list = [str(2000 + x) for x in range(8, 16)]
years.append(year_list)

# entropy_rank_3_component_128_ISR_PSE_conflict
ranks.append(3); components.append(128); countries.append(['ISR', 'PSE'])
year_list = ['2008', '2009', '2012', '2014']
years.append(year_list)

# entropy_rank_4_component_26_LBN_ISR_conflict
ranks.append(4); components.append(26); countries.append(['LBN', 'ISR'])
year_list = ['2006', '2010', '2014', '2015']
years.append(year_list)

# entropy_rank_7_component_96_AUS_IDN_spying

# entropy_rank_8_component_41_USA_AFG_war

# entropy_rank_9_component_136_Snowden_discussions

# entropy_rank_10_component_81_SAU_UAE_QAT_sever_diplo_relations

# entropy_rank_13_component_94_Andean_diplomatic_crisis

# entropy_rank_14_component_48_Iraqi_freedom

# entropy_rank_15_component_71_2nd_swat_and_USA_increased_forces_in_PAK

# entropy_rank_16_component_110_SDN_intl_crisis

# entropy_rank_17_component_113_CHN_JPN_KOR_summit

# entropy_rank_22_component_52_falklands_dispute

# entropy_rank_27_component_91_IRQ_USA_conflict

# entropy_rank_28_component_66_SOM_KEN_conflict

# entropy_rank_29_component_33_ETH_ERI_border_conflict

# entropy_rank_30_component_17_ARM_AZE_conflict_and_armistice_mediated_by_RUS

# entropy_rank_31_component_54_RUS_in_SYR_civil_war

# entropy_rank_32_component_57_SAU_interv_YEM_civil_war

# entropy_rank_33_component_119_SOM_civil_war_US_interv

# entropy_rank_35_component_11_senkaku_islands_dispute

# entropy_rank_37_component_148_US_aid_for_HTI_earthquake

# entropy_rank_38_component_147_US_kills_bin_Laden

# entropy_rank_39_component_19_IRN_ISR_proxy_war

# entropy_rank_41_component_3_RUS_UKR_war

# entropy_rank_43_component_88_PRK

# entropy_rank_47_component_67_PRK_USA_relations

# entropy_rank_48_component_83_South_Sudanese_civil_war

# entropy_rank_49_component_100_EGY_protests_EGY_closes_path_to_Gaza

# get the event counts and form a dataframe =======================================================
for i in tqdm(range(len(ranks))):
    counts.append(get_counts(data, countries[i], years[i]))

total_counts = pd.DataFrame({
    'rank': ranks,
    'component': components,
    'actors': countries,
    'event_count': counts
})

print(total_counts)