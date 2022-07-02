import fasttext
import pandas as pd 


model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')

def get_syns(word, threshold=.70, nbr=5):
    neighbors = model.get_nearest_neighbors(word, k=10 )
    filtered = [x[1] for x in neighbors if x[0] >= threshold][:nbr]
    if filtered:
        return f'{word},' + ','.join(filtered)

top_words = pd.read_csv('/workspace/datasets/fasttext/top_words.txt')
top_words.columns = ['word']
top_words['syns'] = top_words['word'].map(lambda x: get_syns(x, nbr=10))

print(top_words.sample(10))

with open('/workspace/datasets/fasttext/synonyms.csv', 'w') as f:
    for index, row in top_words.iterrows():
        if row['syns']:
            f.write(row['syns'] + '\n')


