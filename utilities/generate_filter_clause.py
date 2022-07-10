import json 


def generate_filter(category_leafs: list):
    clause = f'''
    {
            "bool": {
                "should": [
                    {"term": {"categoryLeaf": "pcmcat214700050000"}},
                    {"term": {"categoryLeaf": "pcmcat171900050029"}}
                ]
            }
        }
    '''
    return clause
 