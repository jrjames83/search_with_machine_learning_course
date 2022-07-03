import fasttext
import pandas as pd 

# shuf /workspace/datasets/reviews/output.fasttext > /workspace/datasets/reviews/shuffled_output.fasttext
# wc -l /workspace/datasets/reviews/shuffled_output.fasttext # 221,771

# head -100000 /workspace/datasets/reviews/shuffled_output.fasttext > /workspace/datasets/reviews/shuffled_output.train
# tail -10000 /workspace/datasets/reviews/shuffled_output.fasttext > /workspace/datasets/reviews/shuffled_output.test


df = pd.read_csv('/workspace/datasets/reviews/shuffled_output.fasttext')
df.columns = ['blob']
df['review'] = df['blob'].map(lambda x: x.split(' ', 1)[0])
df['comment'] = df['blob'].map(lambda x: x.split(' ', 1)[1])
print('label distribution')
print(df['review'].value_counts(normalize=True))
print('label counts')
print(df['review'].value_counts(normalize=False))

model = fasttext.train_supervised(input="/workspace/datasets/reviews/shuffled_output.train", 
                            lr=1.0, 
                            epoch=25, 
                            wordNgrams=2, 
                            bucket=200000, 
                            dim=50, 
                            loss='hs'
                            )
test_result = model.test("/workspace/datasets/reviews/shuffled_output.test")                            
print(test_result)

# Read 8M words
# Number of words:  100965
# Number of labels: 5
# Progress: 100.0% words/sec/thread:  996275 lr:  0.000000 avg.loss:  0.212272 ETA:   0h 0m 0s
# ^[[6;1R(10000, 0.6753, 0.6753)


# Finding Common Misclassifications

validation = pd.read_csv('/workspace/datasets/reviews/shuffled_output.test')
validation.columns = ['blob']
validation['review'] = validation['blob'].map(lambda x: x.split(' ', 1)[0])
validation['comment'] = df['blob'].map(lambda x: x.split(' ', 1)[1])
print('label distribution')
print(validation['review'].value_counts(normalize=True))
print('label counts')
print(validation['review'].value_counts(normalize=False))

validation['']
