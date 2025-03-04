# For classifying product names to categories:

## What precision (P@1) were you able to achieve?
After removing categories with fewer than 20 products, my `p@1` is noted below. This was on a 20% random sample 
of the entire corpus and evaluted on 5,000 records. 
I incorporated that minimum category count functionality into the `createContentTrainingData.py` script

```
# Final Read (7/2/22) - minimum 20 products per category, 20% sample
# Progress: 100.0% words/sec/thread:    1080 lr:  0.000000 avg.loss:  0.656024 ETA:   0h 0m 0s
# N       4998
# P@1     0.697
# R@1     0.697
```

## What fastText parameters did you use?

`~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data.txt -output $MODEL_NAME -lr 1.0 -epoch 25 -wordNgrams 2`

## How did you transform the product names?

Lowercased, stripped and removed punctuation, followed by converting >1 space to a single space using bash. 

```python
def transform_name(product_name): 
    return product_name.lower().strip().translate(str.maketrans('', '', string.punctuation))
```

## How did you prune infrequent category labels, and how did that affect your precision?

In the `createContentTrainingData.py` file, after looping through all the files and storing the product names in a default dictionary whose key was the category, I filtered out infrequent categories at the end, before writing to the training file. The core idea is illustrated below. 

```python
# products_per_category is a default dict whose key is 'category' and value is a list of products falling into said category
filtered_categories = {cat: names_list for cat, names_list in products_per_category.items() if len(names_list) >= min_products}
```

## How did you prune the category tree, and how did that affect your precision?

I may do this time permitting. 


# For deriving synonyms from content:

## What were the results for your best model in the tokens used for evaluation?

```
(search_with_ml) gitpod /workspace/search_with_machine_learning_course (main) $ ~/fastText-0.9.2/fasttext nn /workspace/datasets/fasttext/title_model.bin
Query word? iphone
4s 0.801497
apple 0.772327
3gs 0.712708
ipod 0.712024
ipad 0.705598
4thgeneration 0.625147
4 0.588033
3g 0.574658
6thgeneration 0.547428
ifrogz 0.543102

Query word? ipad
apple 0.77244
iphone 0.705597
3rd 0.656903
ipod 0.602837
generation 0.599774
sleeve 0.593675
4thgeneration 0.577304
4s 0.569899
6thgeneration 0.565073
5thgeneration 0.544465

Query word? headphones
earbud 0.878439
headphone 0.861802
overtheear 0.779598
onear 0.67021
noiseisolating 0.663647
2xl 0.66334
noisecanceling 0.657983
earphones 0.635765
bud 0.619621
inear 0.615925
```
Nothing very good, but you can see the model began picking-up basic semantic relationships. 

## What fastText parameters did you use?

```
~/fastText-0.9.2/fasttext skipgram -minCount 20 -epoch 25 -input /workspace/datasets/fasttext/normalized_titles.txt -output /workspace/datasets/fasttext/title_model
```

## How did you transform the product names?

I used the provided logic:
```
cut -d' ' -f2- /workspace/datasets/fasttext/shuffled_normalized_labeled_products.txt > /workspace/datasets/fasttext/titles.txt
cat /workspace/datasets/fasttext/titles.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_titles.txt
```

# For integrating synonyms with search:

## How did you transform the product names (if different than previously)?
Same as in the previous exercise. 

## What threshold score did you use?
I used a threshold of `.70` in `generate_synonyms.py`

## Were you able to find the additional results by matching synonyms?

Yeah, after modiying the `/utilities/query.py` file to optionally use `name.synonyms` using a command line argument, it does seem that we generally improve recall in terms of `nbr_hits`, but I'd need more exhaustive benchmarking or a handlabeled dataset to see if we improved recall at the expense of precision, or how often there's movement in the top N results with or without the use of the index-time synonym strategy on the name field. 


# For classifying reviews:

## What precision (P@1) were you able to achieve?


Here's the classification report, it wasn't very good. 

```
              precision    recall  f1-score   support

 __label__10       0.08      0.08      0.08      1767
 __label__20       0.05      0.04      0.05       936
 __label__30       0.07      0.07      0.07      1161
 __label__40       0.19      0.20      0.20      4025
 __label__50       0.60      0.60      0.60     12110

    accuracy                           0.42     19999
   macro avg       0.20      0.20      0.20     19999
weighted avg       0.42      0.42      0.42     19999
```

The distribution is dominated by 5 and 4 star reviews:

```
label distribution
__label__50    0.607954
__label__40    0.199256
__label__10    0.088443
__label__30    0.059494
__label__20    0.044848
__label__00    0.000005
Name: review, dtype: float64
label counts
__label__50    134826
__label__40     44189
__label__10     19614
__label__30     13194
__label__20      9946
__label__00         1
Name: review, dtype: int64
```

Another thing, I think some of the labels are either incorrect, or have changed:

https://github.com/jrjames83/search_with_machine_learning_course/blob/main/week2/Training%20FastText%20on%20Product%20Reviews.ipynb

If we inspect where the actual label is a 1.0 review and our model is predicting 5.0, e.g.

```python
(
    validation.query('review == "__label__10"')
    .query('predicted == "__label__50"')
    .sample(10, random_state=333)
    .drop('blob', axis=1)
)
```

You can Google some of them and actually see the 5-stars on best buy

```
chilidog9578
Rated 5 out of 5 stars
Better than Original Endless Summer
Posted 13 years ago.
Better shots and better likeability with the people in the movie. Unbelievable camera work and actually more clever dialogue than the original Endless Summer. My biggest problem is that there is NO BLU RAY verison available!! Hopefully that will come along soon because that would be amazing. This is a must buy!!

 I would recommend this to a friend
 ```

 So, I'd reckon that due to data quality issues, our classification metrics look worse. 


## What fastText parameters did you use?

```python
model = fasttext.train_supervised(input="/workspace/datasets/reviews/shuffled_output.train", 
                            lr=1.0, 
                            epoch=25, 
                            wordNgrams=2,
                            )
```

## How did you transform the review content?

Remove punctuation, normalizes spaces, no stemming. 
```python
def clean_text(text):
    new_string = text.translate(text.maketrans('', '', string.punctuation))
    return " ".join(new_string.split()).lower().strip()
```

## What else did you try and learn?

Getting jupyter set-up and running in gitpod https://github.com/jrjames83/search_with_machine_learning_course/blob/main/week2/getting_jupyter_running.txt

Also refactoring the `createReviewLabels.py` to use `Multiprocessing` and `BeautifulSoup` for structured data parsing. https://github.com/jrjames83/search_with_machine_learning_course/blob/main/week2/createReviewLabels.py



