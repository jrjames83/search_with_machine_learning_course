WORKING_DIR=/workspace/datasets/fasttext
MODEL_NAME="category_model_week3"
head -80000 $WORKING_DIR/labeled_queries_stratified_False.txt > $WORKING_DIR/training_data_week3.txt
tail -20000 $WORKING_DIR/labeled_queries_stratified_False.txt > $WORKING_DIR/test_data_week3.txt

~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME -epoch 25 -wordNgrams 2
# ~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME 
# ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 2
# ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 3

# /workspace/datasets/fasttext/test_data_week3.txt
# /workspace/datasets/fasttext/training_data_week3.txt
# /workspace/datasets/fasttext/labeled_queries_stratified_False.txt


# gitpod /workspace/search_with_machine_learning_course (main) $ ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 1
# N       20000
# P@1     0.549
# R@1     0.549
# gitpod /workspace/search_with_machine_learning_course (main) $ ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 2
# N       20000
# P@2     0.337
# R@2     0.675
# gitpod /workspace/search_with_machine_learning_course (main) $ ~/fastText-0.9.2/fasttext test category_model_week3.bin /workspace/datasets/fasttext/test_data_week3.txt 3
# N       20000
# P@3     0.245
# R@3     0.734






