WORKING_DIR=/workspace/datasets/fasttext
MODEL_NAME="category_model_week3"
head -50000 $WORKING_DIR/labeled_queries.txt > $WORKING_DIR/training_data_week3.txt
tail -5000 $WORKING_DIR/labeled_queries.txt > $WORKING_DIR/test_data_week3.txt

# ~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME -lr .90 -epoch 25 -wordNgrams 2
~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME 
~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 2
~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 3



