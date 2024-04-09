# Amazon-C4

[[🤗 Huggingface Hub](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4)]

[Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4), which is short for "**C**omplex **C**ontexts **C**reated by **C**hatGPT", is a new dataset for the **complex product search** task.

<center>
    <img src="../assets/amazon-c4-example.png" style="width: 50%;">
</center>

## Quick Start

### Loading Queries

```python
from datasets import load_dataset
dataset = load_dataset('McAuley-Lab/Amazon-C4')['test']
```

```python
>>> dataset
Dataset({
    features: ['qid', 'query', 'item_id', 'user_id', 'ori_rating', 'ori_review'],
    num_rows: 21223
})
```

```python
>>> dataset[288]
{'qid': 288, 'query': 'I need something that can entertain my kids during bath time. It should be able to get messy, like smearing peanut butter on it.', 'item_id': 'B07DKNN87F', 'user_id': 'AEIDF5SU5ZJIQYDAYKYKNJBBOOFQ', 'ori_rating': 5, 'ori_review': 'Really helps in the bathtub. Smear some pb on there and let them go to town. A great distraction during bath time.'}
```

### Loading Item Pool

If you would like to use the same item pool used for our [BLaIR](https://arxiv.org/abs/2403.03952) paper, you can follow these steps:

```python
import json
from huggingface_hub import hf_hub_download

filepath = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-C4',
    filename='sampled_item_metadata_1M.jsonl',
    repo_type='dataset'
)

item_pool = []
with open(filepath, 'r') as file:
    for line in file:
        item_pool.append(json.loads(line.strip()))
```

```python
>>> len(item_pool)
1058417
```

```python
>>> item_pool[0]
{'item_id': 'B0778XR2QM', 'category': 'Care', 'metadata': 'Supergoop! Super Power Sunscreen Mousse SPF 50, 7.1 Fl Oz. Product Description Kids, moms, and savvy sun-seekers will flip for this whip! Formulated with nourishing Shea butter and antioxidant packed Blue Sea Kale, this one-of-a kind mousse formula is making sunscreen super FUN! The refreshing light essence of cucumber and citrus has become an instant hit at Super goop! HQ where we’ve been known to apply gobs of it just for the uplifting scent. Water resistant for up to 80 minutes too! Brand Story Supergoop! is the first and only prestige skincare brand completely dedicated to sun protection. Supergoop! has Super Broad Spectrum protection, which means it protects skin from UVA rays, UVB rays and IRA rays.'}
```

## Dataset Description

### Dataset Summary

Amazon-C4 is designed to assess a model's ability to comprehend complex language contexts and retrieve relevant items.

In conventional product search, users may input short, straightforward keywords to retrieve desired items. In the new product search task with complex contexts, the input is longer and more detailed, but not always directly relevant to the item metadata. Examples of such input include multiround dialogues and complex user instructions.

### Dataset Processing

Amazon-C4 is created by prompting ChatGPT to generate complex contexts as queries.

During data construction:
* 5-star-rated user reviews on items are treated as satisfactory interactions.
* reviews with at least 100 characters are considered valid for conveying sufficient information to be rewritten as complex contextual queries.

We uniformly sample around
22,000 of user reviews from the test set of [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) that meet the rating and review length requirements. ChatGPT rephrases user reviews as complex contexts with a first-person tone, serving as queries in the constructed Amazon-C4 dataset.

## Dataset Structure

### Data Fields

- `test.csv` are query-item pairs that can be used for evaluating the complex product search task. There are 6 columns in this file:
    - `qid (int64)`: Query ID. Unique ID for each query, ranging from 0 to 21222. An example of `conv_id` is:
        ```
        288
        ```
    - `query (string)`: Complex query. For example:
        ```
        I need something that can entertain my kids during bath time. It should be able to get messy, like smearing peanut butter on it.
        ```
    - `item_id (string)`: Unique ID for the ground truth item. This ID corresponds to `parent_asin` in the original Amazon Reviews 2023 dataset. For example:
        ```
        B07DKNN87F
        ```
    - `user_id (string)`: The unique user ID. For example:
        ```
        AEIDF5SU5ZJIQYDAYKYKNJBBOOFQ
        ```
    - `ori_rating (float)`: Rating score of the original user review before rewritten by ChatGPT. Note that this field should not be used for solving this task, but just remained for reference. For example:
        ```
        5
        ```
    - `ori_review (string)`: Original review text before rewritten by ChatGPT. Note that this field should not be used for solving this task, but just remained for reference. For example:
        ```
        Really helps in the bathtub. Smear some pb on there and let them go to town. A great distraction during bath time.
        ```
- `sampled_item_metadata_1M.jsonl` contains ~1M items sampled from the Amazon Reviews 2023 dataset. For each <query, item> pairs, we randomly sample 50 items from the domain of the ground-truth item. This sampled item pool is used for evaluation of the [BLaIR paper](https://arxiv.org/abs/2403.03952). Each line is a json:
    - `item_id (string)`: Unique ID for the ground truth item. This ID corresponds to `parent_asin` in the original Amazon Reviews 2023 dataset. For example:
        ```
        B07DKNN87F
        ```
    - `category (string)`: Category of this item. This attribute can be used to evaluate the model performance under certain category. For example:
        ```
        Pet
        ```
    - `metadata (string)`: We concatenate `title` and `description` from the original item metadata of the Amazon Reviews 2023 dataset together into this attribute.

### Data Statistic

|#Queries|#Items|Avg.Len.q|Avg.Len.t|
|-|-|-|-|
|21,223|1,058,417|229.89|538.97|

Where `Avg.Len.q` denotes the average
number of characters in the queries, `Avg.Len.t` denotes the average number of characters in the item metadata.

### Contact

Please [raise a issue here](https://github.com/hyp1231/AmazonReviews2023/issues/new) at our GitHub repo, or [start a discussion](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4/discussions/new) at the huggingface hub, or directly contact Yupeng Hou @ [yphou@ucsd.edu](mailto:yphou@ucsd.edu) if you have any questions or suggestions.
