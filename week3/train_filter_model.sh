
#wc -l /workspace/datasets/fasttext/labeled_queries.txt 406k

shuf /workspace/datasets/fasttext/labeled_queries.txt > /workspace/datasets/fasttext/labeled_queries_shuffled_week3.txt

WORKING_DIR=/workspace/datasets/fasttext
MODEL_NAME="category_model_week3"
head -100000 $WORKING_DIR/labeled_queries_shuffled_week3.txt > $WORKING_DIR/training_data_week3.txt
tail -15000 $WORKING_DIR/labeled_queries_shuffled_week3.txt > $WORKING_DIR/test_data_week3.txt

~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME -lr 1.0 -epoch 25 -wordNgrams 2
# ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt

