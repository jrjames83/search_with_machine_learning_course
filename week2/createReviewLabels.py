import os
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import multiprocessing
import glob 
import time 
import string 

# Directory for review data
directory = r'/workspace/datasets/product_data/reviews/'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing reviews")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")


def clean_text(text):
    new_string = text.translate(text.maketrans('', '', string.punctuation))
    return " ".join(new_string.split()).lower().strip()

def analyze_file(_filename):
    records = []
    with open(_filename, 'r') as xml_file:
        soup = BeautifulSoup(xml_file, "html.parser")
        reviews = soup.find_all('review')
        for review in reviews:
            rating = review.find('rating').get_text()
            title = review.find('title').get_text()
            comment = review.find('comment').get_text()
            records.append({
                'rating': clean_text(rating), 
                'title_comment': clean_text(title + ' ' + comment)
            })
    return records


if __name__ == '__main__':
    args = parser.parse_args()
    output_file = args.output
    path = Path(output_file)
    output_dir = path.parent
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if args.input:
        directory = args.input
    files = glob.glob(f'{directory}/*.xml')
    with multiprocessing.Pool() as p:
        all_labels = tqdm(p.imap_unordered(analyze_file, files), total=len(files))    
        with open(output_file, 'w') as output:                    
            for record in all_labels:
                for row in record:
                    # print(row)
                    # time.sleep(1)
                    rating, data = row['rating'], row['title_comment']
                    output.write(f"__label__{rating} {data}\n")
