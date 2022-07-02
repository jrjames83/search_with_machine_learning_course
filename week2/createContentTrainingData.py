import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import string

def transform_name(product_name):
    ''' basic string cleaning'''  
    return product_name.lower().strip().translate(str.maketrans('', '', string.punctuation))


# Directory for product data
directory = r'/workspace/datasets/product_data/products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--label", default="id", help="id is default and needed for downsteam use, but name is helpful for debugging")

# Consuming all of the product data, even excluding music and movies,
# takes a few minutes. We can speed that up by taking a representative
# random sample.
general.add_argument("--sample_rate", default=25.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=20, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
products_per_category = defaultdict(list)
min_products = args.min_products
sample_rate = args.sample_rate / 100
names_as_labels = False
if args.label == 'name':
    names_as_labels = True

# https://github.com/gitpod-io/gitpod/issues/758
print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > 1 - sample_rate:
                    continue
                # Check to make sure category name is valid and not in music or movies
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None and
                    child.find('categoryPath')[0][0].text == 'cat00000' and
                    child.find('categoryPath')[1][0].text != 'abcat0600000'):
                      # Choose last element in categoryPath as the leaf categoryId or name
                      if names_as_labels:
                          cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][1].text.replace(' ', '_')
                      else:
                          cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      products_per_category[cat].append(name)


    filtered_categories = {cat: names_list for cat, names_list in products_per_category.items() if len(names_list) >= min_products}
    print(f'found {len(list(filtered_categories.keys()))} Categories with more than {min_products} examples')
    for category, products in filtered_categories.items():
        for p in products:
            print(category, p)
            output.write(f"__label__{category} {transform_name(p)} \n")