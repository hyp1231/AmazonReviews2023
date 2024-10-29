# Product Search

## Reproduction

> [!NOTE]  
> The original code has been refactored to be more concise and clean. As a result, the product search results could be slightly different from the numbers in our paper.

```bash
# First generate dense query/item representations and cache them
python generate_emb.py --plm_name hyp1231/blair-roberta-base --feat_name blair-base
# Then evaluate the product search performance
python eval_search.py --suffix blair-baseCLS --domain
```

## Arguments

* `--plm_name`
    * `roberta-base`
    * `roberta-large`
    * `princeton-nlp/sup-simcse-roberta-base`
    * `princeton-nlp/sup-simcse-roberta-large`
    * `hyp1231/blair-roberta-base`
    * `hyp1231/blair-roberta-large`

> [!NOTE]  
> Please update `feat_name` and `--suffix` accordingly.
