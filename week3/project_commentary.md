# Week 3 project commentary

## 1 - Query Classification

### How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 1000? To 10000?

- 500 threshold: `559 unique categories`
- 1000 threshold: `406 unique categories`
- 10000 threshold: `79 unique categories`

Link to notebook/algorith/output: https://gist.github.com/jrjames83/35a454dd8312d421186bee2498b7226b

### Best values acheived for `R@1`, `R@2` and `R@3`

#### Model Training and Parameterization
- 25 epochs and wordNgrms of 2 were better than the defaults. I kept getting core dumps during training when experimenting with non default learning rates along with wordNgrams, so I moved on and used the below:
```
WORKING_DIR=/workspace/datasets/fasttext
MODEL_NAME="category_model_week3"
head -80000 $WORKING_DIR/labeled_queries_stratified_False.txt > $WORKING_DIR/training_data_week3.txt
tail -20000 $WORKING_DIR/labeled_queries_stratified_False.txt > $WORKING_DIR/test_data_week3.txt
~/fastText-0.9.2/fasttext supervised -input $WORKING_DIR/training_data_week3.txt -output $MODEL_NAME -epoch 25 -wordNgrams 2
```

#### Model Output at Various K given about parameters
```
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
```


## 2 -For integrating query classification with search

### Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.

<strong>NOTE: below, we're using a threshold of `.20` and taking any predicted categories which are above it</strong>. The clause is generated here: `/workspace/search_with_machine_learning_course/utilities/generate_filter_clause.py` and also shown below:

```python
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
```

A complete example of how I've incorporated the filtering into the query is below, as a reference point. I also learned that using a `filter` clause doesn't impact scoring, it's simply a step that does post-processing on the retrieved results set. I don't know if this is the optimal way. I also wonder if we're predicting a leaf with both ancestors and descendants, should we include all downstream descendants to maximize recall, but I didn't have time to think this through. 

```
GET /bbuy_products/_search
{
  "size": 5,
  "sort": [
    {
      "_score": {
        "order": "desc"
      }
    }
  ],
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": [],
          "should": [
            {
              "match": {
                "name": {
                  "query": "otterbox case",
                  "fuzziness": "1",
                  "prefix_length": 2,
                  "boost": 0.01
                }
              }
            },
            {
              "match_phrase": {
                "name.hyphens": {
                  "query": "iphone case",
                  "slop": 1,
                  "boost": 50
                }
              }
            },
            {
              "multi_match": {
                "query": "iphone case",
                "type": "phrase",
                "slop": "6",
                "minimum_should_match": "2<75%",
                "fields": [
                  "name^10",
                  "name.hyphens^10",
                  "shortDescription^5",
                  "longDescription^5",
                  "department^0.5",
                  "sku",
                  "manufacturer",
                  "features",
                  "categoryPath"
                ]
              }
            },
            {
              "terms": {
                "sku": [
                  "iphone",
                  "case"
                ],
                "boost": 50
              }
            },
            {
              "match": {
                "name.hyphens": {
                  "query": "iphone case",
                  "operator": "OR",
                  "minimum_should_match": "2<75%"
                }
              }
            }
          ],
          "minimum_should_match": 1,
          "filter": {
            "bool": {
              "should": [
                {
                  "term": {
                    "categoryLeaf": "pcmcat214700050000"
                  }
                },
                {
                  "term": {
                    "categoryLeaf": "pcmcat171900050029"
                  }
                }
              ]
            }
          }
        }
      },
      "boost_mode": "multiply",
      "score_mode": "sum",
      "functions": [
        {
          "filter": {
            "exists": {
              "field": "salesRankShortTerm"
            }
          },
          "gauss": {
            "salesRankShortTerm": {
              "origin": "1.0",
              "scale": "100"
            }
          }
        },
        {
          "filter": {
            "exists": {
              "field": "salesRankMediumTerm"
            }
          },
          "gauss": {
            "salesRankMediumTerm": {
              "origin": "1.0",
              "scale": "1000"
            }
          }
        },
        {
          "filter": {
            "exists": {
              "field": "salesRankLongTerm"
            }
          },
          "gauss": {
            "salesRankLongTerm": {
              "origin": "1.0",
              "scale": "1000"
            }
          }
        },
        {
          "script_score": {
            "script": "0.0001"
          }
        }
      ]
    }
  },
  "_source": [
    "name",
    "shortDescription",
    "categoryPathIds",
    "categoryLeaf",
    "categoryPath",
    "categoryLeaf.keyword"
  ]
}
```

<hr>
`hp touchpad` - the first table highlights filtered results, the second doesn't use the ML category filter. You can see we eliminate ink cartirdges using the filter!

```
          _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  2842056  0.059316  [pcmcat209000050008]    [HP - TouchPad Tablet with 16GB Memory - Black]  [HP webOS 3.09.7" multitouch displayWi-Fi16GB ...
1  bbuy_products  _doc  4892132  0.010051  [pcmcat209000050008]  [HP - Refurbished Touchpad Tablet with 32GB Me...  [Refurbished\nHP webOS 3.09.7" multitouch disp...
2  bbuy_products  _doc  5532645  0.010051  [pcmcat209000050008]  [HP - Refurbished Touchpad Tablet with 16GB Me...  [RefurbishedHP webOS 3.0 operating system9.7" ...
3  bbuy_products  _doc  1604034  0.000003  [pcmcat209000050008]  [Hewlett-Packard (HP) - XT962UA 8.9" LED Net-t...  [Multi-touch Screen WSVGA Display - 2 GB RAM -...
4  bbuy_products  _doc  4826016  0.000003  [pcmcat209000050008]  [HP - Slate 2 B2A28UT 8.9" LED Net-tablet PC -...  [Multi-touch Screen 1024 x 600 WSVGA Display -...
5  bbuy_products  _doc  4826025  0.000003  [pcmcat209000050008]  [HP - Slate 2 B2A29UT 8.9" LED Net-tablet PC -...  [Multi-touch Screen 1024 x 600 WSVGA Display -...

          _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  1982034  0.128230        [abcat0807005]                 [HP - 564XL Ink Cartridge - Black]  [Compatible with select HP Photosmart printers...
1  bbuy_products  _doc  8891853  0.120025        [abcat0807005]                   [HP - 564 Ink Cartridge - Black]  [Compatible with HP d5460, ps pro68550, psc638...
2  bbuy_products  _doc  3080048  0.104885        [abcat0808006]   [HP - 564 Series Ink and Photo Paper Combo Pack]  [Compatible with select HP printers; cyan, mag...
3  bbuy_products  _doc  8510961  0.097167        [abcat0401004]     [HP - Photosmart 7.2MP Digital Camera - White]  [3x optical/3x digital zoom; 2.4" indoor/outdo...
4  bbuy_products  _doc  8891933  0.090772        [abcat0807005]                     [HP - 564 Photo Ink Cartridge]  [Compatible with HP d5460, ps pro68550, psc638...
5  bbuy_products  _doc  2884137  0.088987  [pcmcat242000050004]                 [HP - USB Charger for HP TouchPad]  [Compatible with HP TouchPad; compact, twist-t...
6  bbuy_products  _doc  2947041  0.085684  [pcmcat242000050005]   [ZAGG - InvisibleSHIELD for HP TouchPad Tablets]  [Compatible with HP TouchPad tablets; scratch-...
7  bbuy_products  _doc  2883101  0.084993  [pcmcat242000050003]   [HP - Wireless Keyboard for HP TouchPad Tablets]  [Compatible with HP TouchPad tablets; Bluetoot...
8  bbuy_products  _doc  5373873  0.084544  [pcmcat247400050000]  [HP - 17.3" Pavilion Laptop - 4GB Memory - 320...  [ENERGY STAR QualifiedWindows 7 Home Premium 6...
9  bbuy_products  _doc  8761912  0.084434        [abcat0807005]                    [HP - 60 Ink Cartridge - Black]  [Compatible with HP Deskjet F4280 printer; 1 c...
```


`iphone 4s` - the first table (filtered appears to strictly return actual iphones). The second table (not using the model) seems to have a mixture of phones and cell phone cases. At a glance, this is a plus...we're only showing phones...but perhaps the query is generic enough to still show cases. In a sorting situation however, we'd prefer to only show the actual product. There's a lot to unpack however. 
```
Enter your query (type 'Exit' to exit or hit ctrl-c):
iPhone 4s
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
          _index _type      _id       _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  3487648  3468.604500  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
1  bbuy_products  _doc  3487784  3147.890600  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
2  bbuy_products  _doc  3562527  2648.157200  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
3  bbuy_products  _doc  3562379  2462.139400  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
4  bbuy_products  _doc  3566966  2205.873300  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
5  bbuy_products  _doc  3566775  1383.896600  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
6  bbuy_products  _doc  3566902     0.214149  [pcmcat209400050001]  [Apple® - iPhone® 4S with 32GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
7  bbuy_products  _doc  3566948     0.214149  [pcmcat209400050001]  [Apple® - iPhone® 4S with 32GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
8  bbuy_products  _doc  3814033     0.214149  [pcmcat209400050001]  [Apple® - iPhone® 4S with 64GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
9  bbuy_products  _doc  3814088     0.214149  [pcmcat209400050001]  [Apple® - iPhone® 4S with 64GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...

          _index _type      _id      _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  3487648  3468.60450  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
1  bbuy_products  _doc  3487784  3147.89060  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
2  bbuy_products  _doc  3562527  2648.15720  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
3  bbuy_products  _doc  3562379  2462.13940  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
4  bbuy_products  _doc  3869466  2448.36570  [pcmcat171900050029]  [Belkin - Essential 050 Case for Apple® iPhone...  [Compatible with Apple iPhone 4 (AT&T and Veri...
5  bbuy_products  _doc  3566966  2205.87330  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
6  bbuy_products  _doc  3869439  1759.83040  [pcmcat171900050029]  [Belkin - Essential 050 Case for Apple® iPhone...  [Compatible with Apple iPhone 4 (AT&T and Veri...
7  bbuy_products  _doc  3566775  1383.89660  [pcmcat209400050001]  [Apple® - iPhone® 4S with 16GB Memory Mobile P...  [iOS 5 operating systemSiri voice assistanceiC...
8  bbuy_products  _doc  4039955   547.99760  [pcmcat201900050009]  [ZAGG - InvisibleSHIELD HD for Apple® iPhone® ...  [Compatible with Apple iPhone 4 and 4S; scratc...
9  bbuy_products  _doc  3133099   546.55554  [pcmcat171900050029]  [LifeProof - Case for Apple® iPhone® 4 and 4S ...  [Compatible with Apple iPhone 4 and 4S; polyca...
```

The word `touchpad` below. Somewhat ambiguous, but we're not showing dishwashers at least, when using the ML generated filters. 
```
touchpad
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
          _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  2842056  0.056139  [pcmcat209000050008]    [HP - TouchPad Tablet with 16GB Memory - Black]  [HP webOS 3.09.7" multitouch displayWi-Fi16GB ...
1  bbuy_products  _doc  4892132  0.055704  [pcmcat209000050008]  [HP - Refurbished Touchpad Tablet with 32GB Me...  [Refurbished\nHP webOS 3.09.7" multitouch disp...
2  bbuy_products  _doc  5532645  0.055704  [pcmcat209000050008]  [HP - Refurbished Touchpad Tablet with 16GB Me...  [RefurbishedHP webOS 3.0 operating system9.7" ...

          _index _type      _id     _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  3937943  35.338123        [abcat0513001]                            [Logitech - M325 Mouse]  [Optical - Wireless - Radio Frequency - Blue -...
1  bbuy_products  _doc  3937952  30.233597        [abcat0513001]                            [Logitech - M325 Mouse]  [Optical - Wireless - Radio Frequency - Red - ...
2  bbuy_products  _doc  3764993   9.192955        [abcat0513004]                         [Logitech - K400 Keyboard]  [Wireless - RFUSB - TouchPad - PC - Mute, Volu...
3  bbuy_products  _doc  9929052   1.850353        [abcat0903002]  [Oster - 0.7 Cu. Ft. Compact Microwave - Stain...  [700 watts of power; electronic controls; 6 on...
4  bbuy_products  _doc  4744842   0.070043        [abcat0905001]             [Maytag - Touchpad Dishwasher - White]  [Energy star qualified; touch control with 11 ...
5  bbuy_products  _doc  4747251   0.070043        [abcat0905001]             [Maytag - Touchpad Dishwasher - Black]  [Energy star qualified; touch control with 11 ...
6  bbuy_products  _doc  3438568   0.070043        [abcat0513006]             [Logitech - Wireless Touchpad - Black]  [2.4GHz wireless technology; USB connectivity;...
7  bbuy_products  _doc  2884137   0.060864  [pcmcat242000050004]                 [HP - USB Charger for HP TouchPad]  [Compatible with HP TouchPad; compact, twist-t...
8  bbuy_products  _doc  4744405   0.060672        [abcat0905001]  [Maytag - 24" Touchpad Built-In Dishwasher - B...  [Energy Star® qualified; AccuTemp™ option; del...
9  bbuy_products  _doc  4747055   0.060672        [abcat0905001]  [Maytag - 24" Touchpad Built-In Dishwasher - B...  [Energy star® qualified; touch control with 11...
```

Once more! `lcd tv` - using the ML based filter, we're definitely in the correct category!

```
lcv tv
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
          _index _type      _id    _score _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  4740317  0.056223       [abcat0101001]   [Toshiba - 32" Class - LCD - 720p - 60Hz - HDTV]                                                 []
1  bbuy_products  _doc  6986112  0.056174       [abcat0101001]  [Magnavox - 15" HD-Ready LCD TV w/HD Component...  [Picture-in-picture; Smart Picture; Smart Soun...
2  bbuy_products  _doc  4550361  0.049794       [abcat0101001]  [Insignia™ - 32" Class / LCD / 720p / 60Hz / H...                               [Best Buy Exclusive]
3  bbuy_products  _doc  4550458  0.048450       [abcat0101001]  [Insignia™ - 40" Class - LCD - 1080p - 60Hz - ...                               [Best Buy Exclusive]
4  bbuy_products  _doc  4840716  0.044274       [abcat0101001]    [Dynex™ - 32" Class / LCD / 720p / 60Hz / HDTV]                               [Best Buy Exclusive]
5  bbuy_products  _doc  5061732  0.041024       [abcat0101001]  [Insignia™ - 39" Class - LCD - 1080p - 60Hz - ...                               [Best Buy Exclusive]
6  bbuy_products  _doc  4677087  0.039063       [abcat0101001]  [Insignia™ - 24" Class / LCD / 1080p / 60Hz / ...                               [Best Buy Exclusive]
7  bbuy_products  _doc  2893174  0.036760       [abcat0101001]    [Samsung - 32" Class - LCD -720p - 60Hz - HDTV]                                                 []
8  bbuy_products  _doc  4550412  0.027987       [abcat0101001]  [Insignia™ - 19" Class - LCD - 720p - 60Hz - H...                               [Best Buy Exclusive]
9  bbuy_products  _doc  4846522  0.026904       [abcat0101001]       [LG - 37" Class / LCD / 1080p / 60Hz / HDTV]                                                 []

          _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  4854433  0.144818  [pcmcat161100050040]                               [Apple® - Apple TV®]  [3rd generation; compatible with most HDTVs wi...
1  bbuy_products  _doc  1953594  0.090065        [abcat0104000]                  [KCPI - Digital TV Converter Box]  [Compatible with most analog televisions and r...
2  bbuy_products  _doc  5774879  0.089453        [abcat0107036]           [Monster Cable - TV Screen Cleaning Kit]  [High-tech reusable MicroFiber cloth cleans yo...
3  bbuy_products  _doc  9881868  0.065984        [abcat0106004]  [Rocketfish™ - Low-Profile Tilting Wall Mount ...  [Compatible with most 32" - 70" flat-panel TVs...
4  bbuy_products  _doc  8453666  0.063234  [pcmcat172000050000]  [Dynex® - 9" Widescreen LCD Digital Picture Fr...  [USB 2.0 interface; 640 x 220 resolution; 16:9...
5  bbuy_products  _doc  9873278  0.056851        [abcat0106004]  [Dynex™ - Low-Profile Wall Mount For Most 26"-...  [Designed for use with most 26"-40" flat-panel...
6  bbuy_products  _doc  4740317  0.056223        [abcat0101001]   [Toshiba - 32" Class - LCD - 720p - 60Hz - HDTV]                                                 []
7  bbuy_products  _doc  6986112  0.056174        [abcat0101001]  [Magnavox - 15" HD-Ready LCD TV w/HD Component...  [Picture-in-picture; Smart Picture; Smart Soun...
8  bbuy_products  _doc  8327697  0.054734        [abcat0106004]  [Sanus - Full-Motion Mount for 15" - 37" Flat-...  [Supports most flat-panel TVs from 15" - 37" a...
9  bbuy_products  _doc  7964937  0.050776        [abcat0106001]     [Studio RTA - TV Stand for Tube TVs Up to 27"]  [Steel and particleboard; 2 shelves; mesh rear...
```

### Give 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.

`Beats` - the filtered results are empty! The default search results without ML based filtering look OK

```
Beats
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
None

          _index _type      _id  ...  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0  bbuy_products  _doc  1232474  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats iBeats Earbud Headph...  [3-button microphone; noise-canceling design; ...
1  bbuy_products  _doc  9492426  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Tour Earbud Headphon...  [In-ear design; tangle-free flat cable; in-lin...
2  bbuy_products  _doc  1232447  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Solo HD Over-the-Ear...  [Over-the-ear design; built-in microphone and ...
3  bbuy_products  _doc  9836718  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Solo HD Over-the-Ear...  [Over-the-ear design; built-in microphone and ...
4  bbuy_products  _doc  5667118  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Solo High-Definition...  [Over-the-ear design; built-in microphone and ...
5  bbuy_products  _doc  1232483  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Monster PowerBeats Earbud ...  [ControlTalk technology; dual driver technolog...
6  bbuy_products  _doc  9836432  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats (Solo HD) RED Editio...  [Over-the-ear design; built-in microphone and ...
7  bbuy_products  _doc  8913606  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Studio Over-the-Ear ...  [Over-the-ear design; active noise cancellatio...
8  bbuy_products  _doc  3476048  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Studio Over-the-Ear ...  [Over-the-ear design; active noise cancellatio...
9  bbuy_products  _doc  4676961  ...  [pcmcat144700050004]  [Beats By Dr. Dre - Beats Mixr Over-the-Ear He...  [Over-the-ear design; in-line microphone; rota...
```
Why is this the case? It appears that `beats` rolls up to `Movies and TV Shows`, which will likely nullify any records from the retrieval step. Since this is a top query, we wouldn't rely on a model to predict its category, but it would be worth doing a complete error analysis on this one in a production scenario. 

```python
In [1]: from model_parser import *

In [2]: organize_predictions('beats')
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Out[2]: 
             category     score                                               path
0            cat02015  0.861266      Best Buy > Movies & Music > Movies & TV Shows
1            cat09000  0.029712                     Best Buy > Best Buy Gift Cards
2            cat02009  0.017790            Best Buy > Movies & Music > Music > Pop
3  pcmcat247400050000  0.013298  Best Buy > Computers & Tablets > Laptop & Netb...
4        abcat0101001  0.010403  Best Buy > TV & Home Theater > TVs > All Flat-...
```

Had we used a more complex query `beats dre`, it would appear that the model performs OK and selects the correct category. 

```python
In [3]: organize_predictions('beats dre')
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Out[3]: 
             category     score                                               path
0  pcmcat143000050011  0.231879  Best Buy > Audio & MP3 > Headphones > Over-Ear...
1  pcmcat144700050004  0.209635  Best Buy > Audio & MP3 > Headphones > All Head...
2        abcat0305000  0.132412     Best Buy > Car, Marine & GPS > Radar Detectors
3  pcmcat143000050007  0.089059  Best Buy > Audio & MP3 > Headphones > Earbud H...
4  pcmcat171900050028  0.086621  Best Buy > Mobile Phones > Mobile Phone Access...
```

`lion king` is another query where the model doesn't return results. 

```
lion king
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
None

           _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0   bbuy_products  _doc  6045763  0.106531        [abcat0508035]  [Disney's The Lion King Classic Game Collectio...  [All your favorite Lion King games in one fun ...
1   bbuy_products  _doc  6049867  0.098150        [abcat0708005]  [Disney's The Lion King 1-1/2 - Game Boy Advance]    [Clown around the jungle with Timon and Pumbaa]
2   bbuy_products  _doc  6676215  0.094436        [abcat0714011]  [VTech - V.Smile Smartridge: Disney's The Lion...          [Simba needs help as he grows and learns]
3   bbuy_products  _doc  9603949  0.003930        [abcat0707003]                     [Disney Friends - Nintendo DS]            [Play with your favorite animated pals]
4   bbuy_products  _doc  8735782  0.003740        [abcat0707005]                     [Disney Friends - Nintendo DS]            [Play with your favorite animated pals]
5   bbuy_products  _doc  6359585  0.003740        [abcat0704006]       [Disney's Collectors' Edition - PlayStation]      [Three Disney adventures are waiting for you]
6   bbuy_products  _doc  1264215  0.003411        [abcat0714002]   [Screenlife - Scene It?: Disney Magical Moments]  [Can you recall all of your favorite Disney mo...
7   bbuy_products  _doc  5767681  0.003135        [abcat0708005]  [Disney's Extreme Skate Adventure - Game Boy A...  [Hop on your skateboard and take to the street...
8   bbuy_products  _doc  5672195  0.002900        [abcat0704006]  [Disney's Extreme Skate Adventure - PlayStatio...  [Hop on your skateboard and take to the street...
9   bbuy_products  _doc  5767155  0.002900        [abcat0709002]  [Disney's Extreme Skate Adventure - Nintendo G...  [Hop on your skateboard and take to the street...
10  bbuy_products  _doc  1094456  0.002522        [abcat0703002]  [Disney Sing It: Family Hits Bundle - PlayStat...  [Jump into the magic and sing like a Disney star]
```

`printer ink black and white` - a longer tail query without any brand component. Unsurprisingly, `bm25` seems to do much better. 
```
printer ink black and white
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Logging Both Results!
           _index _type      _id    _score  _source.categoryLeaf                                       _source.name                           _source.shortDescription
0   bbuy_products  _doc  1114791  0.079762  [pcmcat144700050004]     [Bose® - IE2 Earbud Headphones - Black, White]  [3 sizes of StayHear tips; 3 sizes of original...
1   bbuy_products  _doc  2197089  0.064146  [pcmcat144700050004]                 [Sony - Earbud Headphones - White]  [13.5mm drivers; neodymium magnet; 3 sets of s...
2   bbuy_products  _doc  2204211  0.063273  [pcmcat144700050004]           [Sony - Over-The-Ear Headphones - White]  [30mm multilayer dome diaphragm; high-energy n...
3   bbuy_products  _doc  8605253  0.056074  [pcmcat144700050004]  [JVC - Gumy Stereo Earbud Headphones - Coconut...  [From our expanded online assortment; compatib...
4   bbuy_products  _doc  2197043  0.045546  [pcmcat144700050004]                 [Sony - Earbud Headphones - Black]  [13.5mm drivers; neodymium magnet; 3 sets of s...
5   bbuy_products  _doc  2204196  0.042396  [pcmcat144700050004]           [Sony - Over-The-Ear Headphones - Black]  [30mm multilayer dome diaphragm; high-energy n...
6   bbuy_products  _doc  5335657  0.037747  [pcmcat144700050004]  [Sony - Clip-On In-Ear Stereo Headphones - Black]  [Vertical in-ear design; ear clips for a secur...
7   bbuy_products  _doc  8605459  0.037287  [pcmcat144700050004]  [JVC - Gumy Stereo Earbud Headphones - Olive B...  [Compatible with most portable audio devices; ...
8   bbuy_products  _doc  3316072  0.036931  [pcmcat144700050004]   [JVC - JVC Sport Clip Earbud Headphones - Black]  [JVC Sport Clip Earbud Headphones: Rubber ear ...
9   bbuy_products  _doc  9251098  0.034602  [pcmcat144700050004]  [Skullcandy - Ink'd Stereo Ear Bud Headphones ...  [Ear bud design; silicone ear gels; noise redu...
10  bbuy_products  _doc  9251105  0.031680  [pcmcat144700050004]  [Skullcandy - Ink'd Ear Bud Stereo Headphones ...  [Ear bud design; 11mm drivers; 

           _index _type      _id    _score _source.categoryLeaf                                       _source.name                           _source.shortDescription
0   bbuy_products  _doc  7264122  2.165899       [abcat0511003]      [HP - LaserJet Black-and-White Laser Printer]  [Up to 15 ppm (print speeds will vary with use...
1   bbuy_products  _doc  7230007  2.052874       [abcat0511003]          [Samsung - Black-and-White Laser Printer]  [Prints up to 22 ppm (print speeds will vary w...
2   bbuy_products  _doc  1982034  0.232712       [abcat0807005]                 [HP - 564XL Ink Cartridge - Black]  [Compatible with select HP Photosmart printers...
3   bbuy_products  _doc  8891853  0.217821       [abcat0807005]                   [HP - 564 Ink Cartridge - Black]  [Compatible with HP d5460, ps pro68550, psc638...
4   bbuy_products  _doc  9942413  0.173654       [abcat0807004]  [Epson - DURABrite Ink Cartridge for Select Ep...  [Compatible with select Epson printers; black ...
5   bbuy_products  _doc  8761912  0.153231       [abcat0807005]                    [HP - 60 Ink Cartridge - Black]  [Compatible with HP Deskjet F4280 printer; 1 c...
6   bbuy_products  _doc  1146635  0.150670       [abcat0807005]                    [HP - 61 Ink Cartridge - Black]  [Compatible with HP Deskjet printer models 300...
7   bbuy_products  _doc  3080048  0.142999       [abcat0808006]   [HP - 564 Series Ink and Photo Paper Combo Pack]  [Compatible with select HP printers; cyan, mag...
8   bbuy_products  _doc  9248879  0.139496       [abcat0807005]       [HP - Officejet 920XL Ink Cartridge - Black]  [Compatible with Officejet 6500 printer series...
9   bbuy_products  _doc  9798556  0.129744       [abcat0807003]          [Canon - PG-210XL Photo Ink Tank - Black]  [Compatible with select canon printers; black ...
10  bbuy_products  _doc  8891933  0.123757       [abcat0807005]                     [HP - 564 Photo Ink Cartridge]  [Compatible with HP d5460, ps pro68550, 
```

Probing the model's outupt for the query we can see that we came close! The 2nd category was the correct one, but for whatever reason, headphones was the highest confidence category and it was over my threshold of .20. In a production scenario, this would definitely be worth debugging. 

```python
In [1]: from model_parser import *

In [2]: organize_predictions("printer ink black and white")
Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
Out[2]: 
             category     score                                               path
0  pcmcat144700050004  0.207527  Best Buy > Audio & MP3 > Headphones > All Head...
1        abcat0807005  0.094323  Best Buy > Office > Printer Ink & Toner > Prin...
2            cat02009  0.066995            Best Buy > Movies & Music > Music > Pop
3        abcat0511002  0.055577     Best Buy > Office > Printers > Inkjet Printers
4        abcat0511004  0.036280  Best Buy > Office > Printers > All-In-One Inkj...
```

### Bottom Line Reflections

Associating queries with categories can be useful for filtering purposes and potentially for shorter tail queries (as seen in the succesful queries section) where `bm25` scoring doesn't do well without deliberate use of function scoring with document signals like popuarlity or through methods like `learning to rank`. Though, to the course authors' point, given the importance of these short tail queries, you're better off investing in human mappings to your taxoomy to avoid classification issues, like was the case with `beats`. 

As the query gets longer and `bm25` tends to do a better job of retrieval, I wonder how impactful this technique can be, outside of an enviornment where the user is filtering by price and you want to avoid running into precision issues, depending on how many results are retrieved and ready to be sorted. Perhaps you can get away with quality sorting for mid to long tail queries using some `minimum_should_match` clauses and document popularity cutoffs before rendering the sorted results. You'd have to benchmark the hand-crafted sort against the model for a variety of queries to get a sense of the tradeoffs. 

Thinking through how the taxonomy is pruned and what effects this has on the classifier quality is difficult, at least without a robust human judgement list or some programmatic heuristic that attempts to address the precision/recall tradeoff. I think I'd want to test multiple classifiers, each time checking how often the filter exercise results in no remaining records after initial retrieval, but also balancing ML classifier thresholds. It's a big effort to say the least!
