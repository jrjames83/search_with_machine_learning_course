import fasttext
import pandas as pd
from prune_taxonomy import get_cat_lookup


def load_fasttext_model():
    model = fasttext.load_model('/workspace/search_with_machine_learning_course/category_model_week3.bin')
    return model 

def prediction_to_df(_prediction):
    preds = dict(zip(*_prediction))
    formatted = [
            {
                'category': k.replace('__label__', ''),
                'score': v 
            }
                for k,v in preds.items()
            ]
    return pd.DataFrame.from_records(formatted)


if __name__ == '__main__':
    cat_lookup_mapping = get_cat_lookup()
    model = load_fasttext_model()
    prediction = model.predict('iphone case', k=5)
    df = prediction_to_df(prediction)
    df['path'] = df['category'].map(lambda x: cat_lookup_mapping[x])
    print(df.head())