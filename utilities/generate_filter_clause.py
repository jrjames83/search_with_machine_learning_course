import json 

def generate_filter(categories: list):
    """
        Generate a filter clause statement from a list of categories
    """
    clause = dict()
    clause['bool'] = dict()
    clause['bool']['should'] = []
    for c in categories:
        clause['bool']['should'].append({"term": {"categoryLeaf": c}})
    return clause
 