import sys
from IPython.display import display, Javascript

def restart_kernel():
    """Restart the Jupyter Notebook kernel to reflect changes in modules and packages."""
    display(Javascript("Jupyter.notebook.kernel.restart()"))
    print("Kernel is restarting...")

restart_kernel()

import bptf
from bptf import BPTF
import numpy as np
import pandas as pd
import sparse
import os
import shutil
from tqdm import tqdm
import pickle
import gdelt
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
import tensorly
import cupy

from datetime import datetime, timedelta

import multiprocessing
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map

import requests
from bs4 import BeautifulSoup

import copy

print(os.getcwd())

import gc
gc.collect()

def make_date_list(start_date: str, end_date: str):
    """
    Generate a list of dates as "YYYY MM DD" between start_date and end_date (inclusive).
    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
    Returns:
        List of strings ["YYYY MM DD", ...]
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    delta = timedelta(days=1)

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y %m %d"))
        current += delta

    return dates

def process_date(year, verbose = False):
    """
    Queries the GDELT database and returns all events with a complete token (i.e. source, target country, action and time) by day.
    GDELT 2 only supports 2015 02 18 and onwards.
    Args:
        year: string. YYYY
        verbose: boolean. True if you need a progress bar
    Returns:
        pandas dataframe with source, target country codes, action and day
    """
    try:
        ver = 1 if int(year) <= 2016 else 2
        gd = gdelt.gdelt(version=ver)
        range_ = [year + '-01-01', year + '-12-31']
        range_ = make_date_list(range_[0], range_[1])
        # print(range_[0], type(range_[0]))
        # results = gd.Search([date], table='events', coverage=True)
        # progress_bar = tqdm(range_, desc=f'Downloading GDELT {year}') if verbose else range_
        # results = [gd.Search(date=date, table='events', coverage=True) for date in progress_bar]
        # results = pd.concat(results)
        results = gd.Search(range_, table='events', coverage=True)
        results = results[['Actor1CountryCode', 'Actor2CountryCode', 'EventBaseCode', 'SQLDATE', 'NumMentions']]
        results.loc[:, 'EventBaseCode'] = results['EventBaseCode'].str[:2]
        results = results.dropna()
        # results = results.groupby(['Actor1CountryCode', 'Actor2CountryCode', 'EventBaseCode', 'SQLDATE'])
        # results = results.sum().reset_index()
        results = results[results['EventBaseCode'].str.isnumeric()]

        # print(f"Processed: {date} - {len(results)} rows")
        return results
    except Exception as e:
        print(f'Failed to process {year}: {e}')
        return pd.DataFrame(columns=['Actor1CountryCode', 'Actor2CountryCode', 'EventBaseCode', 'SQLDATE', 'NumMentions'])
    
filepath = 'gdelt1.csv'
# if os.path.exists(filepath):
if False:
    gdelt1 = pd.read_csv(filepath)
    print(f'{filepath} exists')
else:
    "At present the GDELT 2.0 data streams only stretch back to late morning February 19, 2015"
    "From the gdelt documentation"
    print('Downloading GDELT1')
    list_of_missing_years = list(range(2000, 2016))
    gdelt1 = []
    while len(list_of_missing_years) > 0:
        years_that_failed_to_download = []
        download_list = tqdm(list_of_missing_years.copy())
        for year in download_list:
            year_df = process_date(str(year))
            if len(year_df) > 0:
                gdelt1.append(year_df)
                download_list.set_description(f'{year} successfully downloaded')
            else:
                years_that_failed_to_download.append(year)
                download_list.set_description(f'{year} failed to download')
        list_of_missing_years = years_that_failed_to_download.copy()
        print(f'Missing years: {list_of_missing_years}')
        gc.collect()
    # gdelt1 = [process_date(str(year)) for year in tqdm(range(2000, 2016))]
    gdelt1 = pd.concat(gdelt1)
    gdelt1 = gdelt1[gdelt1['SQLDATE'] <= 20150218]
    # gdelt1.to_csv(filepath, index=False)

print(f'Earliest date is {pd.to_datetime(gdelt1["SQLDATE"].astype(str), format="%Y%m%d").min()}')
print(f'Latest date is {pd.to_datetime(gdelt1["SQLDATE"].astype(str), format="%Y%m%d").max()}')