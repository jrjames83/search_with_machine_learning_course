#!/bin/bash

# After running createContentTrainingData.py
# cat /workspace/datasets/fasttext/labeled_products.txt |sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_labeled_products.txt
# normalized_labeled_products
# shuf /workspace/datasets/fasttext/normalized_labeled_products.txt > /workspace/datasets/fasttext/shuffled_normalized_labeled_products.txt


WORKING_DIR=/workspace/datasets/fasttext
MODEL_NAME="category_model"
head -15000 $WORKING_DIR/shuffled_normalized_labeled_products.txt > $WORKING_DIR/training_data.txt
tail -5000 $WORKING_DIR/shuffled_normalized_labeled_products.txt > $WORKING_DIR/test_data.txt

# train_lines=$(ls *.log | grep "candump" | tail -n 1)

# ~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data.txt -output $MODEL_NAME
~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data.txt -output $MODEL_NAME -lr 1.0 -epoch 25 -wordNgrams 2
# Default
# Number of words:  11018
# Number of labels: 1390
# Progress: 100.0% words/sec/thread:     442 lr:  0.000000 avg.loss: 13.533681 ETA:   0h 0m 0s

~/fastText-0.9.2/fasttext test category_model.bin /workspace/datasets/fasttext/test_data.txt

# Read 0M words
# Number of words:  11079
# Number of labels: 1358
# Progress: 100.0% words/sec/thread:     486 lr:  0.000000 avg.loss: 13.267459 ETA:   0h 0m 0s
# Read 0M words
# Number of words:  11079
# Number of labels: 1358
# Progress: 100.0% words/sec/thread:     652 lr:  0.000000 avg.loss:  1.263036 ETA:   0h 0m 0s
# model_training.sh: line 17: 2: command not found
# N       4807
# P@1     0.604
# R@1     0.604


# Original , no nograms of additional training
# N       4854
# P@1     0.145
# R@1     0.145

# What about p@5
#  ~/fastText-0.9.2/fasttext test category_model.bin /workspace/datasets/fasttext/test_data.txt 5
# N       4854
# P@5     0.0434
# R@5     0.217


# Final Read (7/2/22) - minimum 20 products per category, 20% sample
# Progress: 100.0% words/sec/thread:    1080 lr:  0.000000 avg.loss:  0.656024 ETA:   0h 0m 0s
# N       4998
# P@1     0.697
# R@1     0.697

