import os
import pandas as pd
import numpy as np
from tqdm import tqdm

CWD = os.getcwd()
AB_DIR = CWD + '/datasets/Abt_Buy/'
BEERS_DIR = CWD + '/datasets/Beers/'
PAPERS_DIR = CWD + '/datasets/DBLP_ACM/'

### ################# ###
### SUPPORT FUNCTIONS ###
### ################# ###

def clean_text(text):
    tokens = text.split()
    tokens = [word.lower() for word in tokens]
    punctuations = ["'", ',', '.', "\\", '/', '(', ')', '[', ']', '{', '}', ':', ';', '?', '!', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '~', '`', '<', '>', '|', '"', 'n/a']
    tokens = [t for t in tokens if t not in punctuations]
    text = ' '.join(tokens)
    return text

def prepare_source(dataset):
    match dataset:
        case 'Abt_Buy':
            ab_instances = pd.read_csv(AB_DIR + 'instances.csv', index_col=0, header=0, dtype=str)
            prepare_abt_buy(ab_instances)
        case 'Beers':
            beers_instances = pd.read_csv(BEERS_DIR + 'instances.csv', index_col=0, header=0, dtype=str)
            prepare_beers(beers_instances)
        case 'DBLP_ACM':
            dblp_acm_instances = pd.read_csv(PAPERS_DIR + 'instances.csv', index_col=0, header=0, dtype=str)
            prepare_dblp_acm(dblp_acm_instances)
        case _:
            raise ValueError("Invalid Dataset: choose among the Datasets in './datasets/'")

### ############# ###
### SUB-FUNCTIONS ###
### ############# ###
def prepare_abt_buy(instances):
    instances['name'] = instances['name'].astype(str).apply(clean_text)
    instances['price'] = instances['price'].astype(str).apply(clean_text)
    instances['description'] = instances['description'].astype(str).apply(clean_text)

    word_weights = dict()
    for idx in tqdm(instances.index, desc="Computing Word Weights for each Record in 'Abt-Buy'"):
        name, description, price = instances.loc[idx]
        words = set()
        if name != 'nan': words.update(name.split())
        if description != 'nan': words.update(description.split())
        if price != 'nan': words.update(price.split())

        for word in words:
            if word not in word_weights:
                word_weights[word] = 0
            word_weights[word] += 1
        
    for word, weight in word_weights.items():
        word_weights[word] = np.log(len(instances) / weight)
    
    weights_df = pd.DataFrame(columns=['word', 'weight'])
    weights_df['word'] = word_weights.keys()
    weights_df['weight'] = word_weights.values()

    weights_df.to_csv(AB_DIR + 'word_weights.csv', index=False, header=True)

    instances.to_csv(AB_DIR + 'instances_refined.csv', index=True, header=True)

def prepare_beers(instances):
    instances['Beer_Name'] = instances['Beer_Name'].astype(str).apply(clean_text)
    instances['Brew_Factory_Name'] = instances['Brew_Factory_Name'].astype(str).apply(clean_text)
    instances['Style'] = instances['Style'].astype(str).apply(clean_text)
    instances['ABV'] = instances['ABV'].astype(str).apply(clean_text)

    word_weights = dict()
    for idx in tqdm(instances.index, desc="Computing Word Weights for each Record in 'Beers'"):
        beer_name, factory_name, style, abv = instances.loc[idx]
        words = set()
        if beer_name != 'nan' and beer_name != '': words.update(beer_name.split())
        if factory_name != 'nan' and factory_name != '': words.update(factory_name.split())
        if style != 'nan' and style != '': words.update(style.split())
        if abv != 'nan' and abv != '': words.update(abv.split())

        for word in words:
            if word not in word_weights:
                word_weights[word] = 0
            word_weights[word] += 1
        
    for word, weight in word_weights.items():
        word_weights[word] = np.log(len(instances) / weight)
    
    weights_df = pd.DataFrame(columns=['word', 'weight'])
    weights_df['word'] = word_weights.keys()
    weights_df['weight'] = word_weights.values()

    weights_df.to_csv(BEERS_DIR + 'word_weights.csv', index=False, header=True)

    instances.to_csv(BEERS_DIR + 'instances_refined.csv', index=True, header=True)

def prepare_dblp_acm(instances):
    instances['title'] = instances['title'].astype(str).apply(clean_text)
    instances['authors'] = instances['authors'].astype(str).apply(clean_text)
    instances['venue'] = instances['venue'].astype(str).apply(clean_text)
    instances['year'] = instances['year'].astype(str).apply(clean_text)

    word_weights = dict()
    for idx in tqdm(instances.index, desc="Computing Word Weights for each Record in 'DBLP-ACM"):
        title, authors, venue, year = instances.loc[idx]
        words = set()
        if title != 'nan': words.update(title.split())
        if authors != 'nan': words.update(authors.split())
        if venue != 'nan': words.update(venue.split())
        if year != 'nan': words.update(year.split())

        for word in words:
            if word not in word_weights:
                word_weights[word] = 0
            word_weights[word] += 1
        
    for word, weight in word_weights.items():
        word_weights[word] = np.log(len(instances) / weight)
    
    weights_df = pd.DataFrame(columns=['word', 'weight'])
    weights_df['word'] = word_weights.keys()
    weights_df['weight'] = word_weights.values()

    weights_df.to_csv(PAPERS_DIR + 'word_weights.csv', index=False, header=True)

    instances.to_csv(PAPERS_DIR + 'instances_refined.csv', index=True, header=True)