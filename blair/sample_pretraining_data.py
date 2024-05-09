import random
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset


num_workers = 64
valid_timestamp = 1628643414042
downsampling_factor = 10
all_cleaned_item_metadata = {}


def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='all_categories.txt',
        repo_type='dataset'
    )
    with open(category_filepath, 'r') as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories


def concat_item_metadata(dp):
    meta = ''
    flag = False
    if dp['title'] is not None:
        meta += dp['title']
        flag = True
    if len(dp['features']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(dp['features'])
        flag = True
    if len(dp['description']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(dp['description'])
    dp['cleaned_metadata'] = meta \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()
    return dp


def filter_reviews(dp):
    # Downsampling
    pr = random.randint(1, downsampling_factor)
    if pr > 1:
        return False
    if dp['timestamp'] >= valid_timestamp:
        return False
    asin = dp['parent_asin']
    if asin not in all_cleaned_item_metadata:
        return False
    if len(dp['cleaned_review']) <= 30:
        return False
    return True


def concat_review(dp):
    review = ''
    flag = False
    if dp['title'] is not None:
        review += dp['title']
        flag = True
    if dp['text'] is not None:
        if flag:
            review += ' '
        review += dp['text']
    dp['cleaned_review'] = review \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()
    return dp


if __name__ == '__main__':
    all_categories = load_all_categories()

    # Load item metadata
    for category in all_categories:
        meta_dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023',
            f'raw_meta_{category}',
            split='full',
            trust_remote_code=True
        )
        concat_meta_dataset = meta_dataset.map(
            concat_item_metadata,
            num_proc=num_workers
        )
        final_meta_dataset = concat_meta_dataset.filter(
            lambda dp: len(dp['cleaned_metadata']) > 30,
            num_proc=num_workers
        )
        for item_id, cleaned_meta in zip(
            final_meta_dataset['parent_asin'],
            final_meta_dataset['cleaned_metadata']
        ):
            all_cleaned_item_metadata[item_id] = cleaned_meta

    # Load reviews
    output_review = []
    output_metadata = []
    for category in all_categories:
        review_dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023',
            f'raw_review_{category}',
            split='full',
            trust_remote_code=True
        )
        concat_review_dataset = review_dataset.map(
            concat_review,
            num_proc=num_workers
        )
        final_review_dataset = concat_review_dataset.filter(
            filter_reviews,
            num_proc=num_workers
        )
        output_review.extend(final_review_dataset['cleaned_review'])
        output_metadata.extend(
            [all_cleaned_item_metadata[_] for _ in final_review_dataset['parent_asin']]
        )

    # Save pretraining data
    df = pd.DataFrame({
        'review': output_review,
        'meta': output_metadata
    })
    df.to_csv('clean_review_meta.tsv', sep='\t', lineterminator='\n', index=False)
