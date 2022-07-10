import sys
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import functools
import string 

from tqdm import tqdm
import pandas as pd
from nltk.stem import PorterStemmer

porter = PorterStemmer()


def clean_text(text):
    new_string = text.translate(text.maketrans('', '', string.punctuation))
    cleaned = " ".join(new_string.split()).lower().strip()
    return " ".join([porter.stem(token) for token in cleaned.split()])

def get_child_to_parent():
    categoriesFilename = '/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
    root_category_id = 'cat00000'
    tree = ET.parse(categoriesFilename)
    root = tree.getroot()

    categories = []
    parents = []
    for child in root:
        id = child.find('id').text
        cat_path = child.find('path')
        cat_path_ids = [cat.find('id').text for cat in cat_path]
        leaf_id = cat_path_ids[-1]
        if leaf_id != root_category_id:
            categories.append(leaf_id)
            parents.append(cat_path_ids[-2])
    parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])
    child_to_parent = parents_df.set_index('category')
    return child_to_parent


def get_cat_lookup(max_depth=10):
    categoriesFilename = '/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
    tree = ET.parse(categoriesFilename)
    root = tree.getroot()    
    catDict = {}
    for child in root:
        catPath = child.find('path')
        leafCat = catPath[-1].find('id').text
        catPathStr = ''
        depth = 0
        for cat in catPath:
            if catPathStr != '':
                catPathStr = catPathStr + ' > '
            catPathStr = catPathStr + cat.find('name').text
            depth = depth + 1
            if max_depth > 0 and depth == max_depth:
                break
        catDict[leafCat] = catPathStr
    return catDict
            
def get_node_parent(cat_id: str, mapping_df: pd.Series):
    """
        Used to grab a node's parent category
    """
    try:
        parent_category = mapping_df.loc[cat_id].parent
        return parent_category
    except KeyError:
        # must already be top of the food chain
        return cat_id
    
@functools.lru_cache(maxsize=None)
def get_ultimate_parent(child_category):
    """
        walk back up to the first ancestor that's not the taxonomy root
    """
    parent_found = False
    parent = child_category
    while not parent_found:
        try:
            # Don't walk back up to the root
            if child_to_parent.loc[parent].parent == 'cat00000':
                return parent
            else:
                parent = child_to_parent.loc[parent].parent
        except KeyError:
            return parent    


if __name__ == '__main__':
    MIN_QUERY = 1000
    STRATIFY = False
    NBR_ROWS = 100_000

    child_to_parent = get_child_to_parent()    
    cat_lookup = get_cat_lookup()
    
    # Read our raw data in and get some category counts and other information
    queries = pd.read_csv('/workspace/datasets/train.csv')
    queries['path'] = queries['category'].map(cat_lookup)
    queries = queries.set_index('category').drop(['sku', 'user', 'query_time'], axis=1)
    # Drop a few rows that don't have a path
    queries = queries[~queries.path.isna()].copy()
    # get the length of the path, so we can prune from the most terminal nodes upward, later on
    queries['path_length'] = queries['path'].map(lambda x: len(x.split('>')))
    # Compute the existing frequency per Category
    sizes = queries.groupby('category').size()
    queries['leaf_counts'] = queries.index.map(sizes)
    queries['ultimate_parent'] = queries.index.map(get_ultimate_parent)

    revised_queries = queries.copy() # to avoid reloading the data!
    solved = False
    while not solved:
        print(f'solving for {MIN_QUERY} minimum query threshold')
        print(revised_queries.index.nunique(), 'is the unique remaining categories')
        # Get the rows having leaf_counts associated with their category lower than the threshold
        leaves_to_rollup = revised_queries.query('leaf_counts < @MIN_QUERY').copy() 
        if leaves_to_rollup.shape[0] == 0:
            solved = True
            break
        # get the other which consist of queries mapped to categories with enough labels already
        leaves_with_enough = revised_queries.query('leaf_counts >= @MIN_QUERY').copy() 

        # for our leaves in need of rolling upward, find the true terminal nodes using their path length
        longest_path_length = leaves_to_rollup.path_length.max()

        # Get the terminal leaves
        to_prune = leaves_to_rollup.query('path_length == @longest_path_length').copy()
        # Don't forget the non-termal leaves, too
        to_not_prune = leaves_to_rollup.query('path_length != @longest_path_length').copy()

        # grab the parent category using the current and a mapping we made in the first cell "child_to_parent"
        to_prune['parent_category'] = to_prune.index.map(lambda x: get_node_parent(x, child_to_parent))

        # update our index
        to_prune.set_index('parent_category', inplace=True)
        to_prune.index.names = ['category']

        # Update our path and path lengths
        to_prune['updated_path'] = to_prune.index.map(lambda x: cat_lookup[x] )
        to_prune['updated_path_length'] = to_prune['updated_path'].map(lambda x: len(x.split('>')))

        # drop the old path lengths and path columns and rename the updated ones in their place
        to_prune.drop(['path_length', 'path'], axis=1, inplace=True)
        to_prune.rename(columns={'updated_path':'path', 'updated_path_length':'path_length'}, inplace=True)

        # combine our updated rows with our unaffected rows
        revised_queries = pd.concat([leaves_with_enough, to_prune, to_not_prune], axis=0, sort=False)

        # Recompute category frequencies and reset the leaf_counts column on our combined data
        sizes = revised_queries.groupby(revised_queries.index).size()
        revised_queries['leaf_counts'] = revised_queries.index.map(sizes)
        print(revised_queries.index.nunique(), 'is the unique remaining categories')

    print(f'Writing the group with {MIN_QUERY} samples per leaf')

    with open(f'/workspace/datasets/fasttext/labeled_queries_stratified_{STRATIFY}.txt', 'w') as f:
        if STRATIFY:
            print(f'Writing Stratified Data - {NBR_ROWS} rows')
            sampled = revised_queries.groupby(revised_queries.index, group_keys=False)\
                .apply(lambda x: x.sample(min(len(x), 1000)))
            sampled['query'] = sampled['query'].map(clean_text)
            sampled = sampled.sample(frac=1).head(NBR_ROWS)
        else:
            print(f'Writing Non-Stratified Data - {NBR_ROWS} rows')
            sampled = revised_queries.sample(NBR_ROWS)
            sampled['query'] = sampled['query'].map(clean_text)

        for index, row in tqdm(sampled.iterrows()):
            if len(row['query']) > 2:
                f.write(f"__label__{index} {row['query']}\n")