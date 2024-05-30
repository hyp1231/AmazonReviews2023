# Text-Based Sequential Recommendation

Reproduction of text-based sequential recommendation results. e.g., UniSRec.

> [!NOTE]  
> The original code has been refactored to be more concise and clean. As a result, statistics of the processed dataset, as well as the recommendation model results could be slightly different from the numbers in our paper.

## Quick Start

### Process the dataset

```bash
cd seq_rec_results/dataset/
python process_amazon_2023.py \
    --domain All_Beauty \
    --device cuda:0 \
    --plm hyp1231/blair-roberta-base
```

For all available domains/categories, please refer to [Hugging Face Hub](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/all_categories.txt).

### Train and evaluate the models

```bash
cd seq_rec_results/
python run.py \
    -m UniSRec \
    -d All_Beauty \
    --gpu_id=0
```
