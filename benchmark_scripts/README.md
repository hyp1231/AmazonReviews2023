# Benchmark Scripts

Based on the released Amazon Reviews 2023 dataset, we provide scripts to preprocess raw data into standard train/validation/test splits to encourage benchmarking recommendation models.

The prosessed datasets can be found at [[🌐 Website 0-Core](https://amazon-reviews-2023.github.io/data_processing/0core.html#statistics)] · [[🌐 Website 5-Core](https://amazon-reviews-2023.github.io/data_processing/5core.html#statistics)] · [[🤗 Huggingface Datasets](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)]

🚀 Quick Jumps -> [[Structures](#structures)] · [[Raw Data -> rating_only](#raw-data---rating_only)] · [[rating_only -> last_out](#rating_only---last_out)]

## Structures

We take the `Toys_and_Games` domain as an example to describe the structures of all the benchmark files.

```
benchmark_files/        # deduplicate reviews
├── 5core/              # 5-core filtering
│   ├── rating_only/    # before split
│   │   └── Toys_and_Games.csv
│   ├── last_out/       # split by "last out (leave-one-out)"
│   │   ├── Toys_and_Games.train.csv
│   │   ├── Toys_and_Games.valid.csv
│   │   └── Toys_and_Games.test.csv
│   ├── timestamp/      # split by timestamps
│   ├── last_out_w_his/ # convenient for sequential rec
│   └── timestamp_w_his/
└── 0core/
    └── ...
```

## Raw Data -> rating_only

`rating_only` contains review records, containing `user`, `item`, `rating`, `timestamp` in each line, but without text and other attributes.

**Preprocessing**

* We remove repeated reviews (those from the same pair of user & item, but may with different review text and ratings) and only keep the earliest ones.
* We filter the reviews using [k-core filtering](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)), where k = 0 or 5.

**Scripts** [[link]](kcore_filtering.py)

```bash
python kcore_filtering.py -k 5
```

**Sampled Data**

`benchmark_files/5core/rating_only/Toys_and_Games.csv`

```
user_id,parent_asin,rating,timestamp
AGKASBHYZPGTEPO6LWZPVJWB2BVA,B006GBITXC,3.0,1452647382000
AGKASBHYZPGTEPO6LWZPVJWB2BVA,B00TLEMSVK,4.0,1454675785000
AGKASBHYZPGTEPO6LWZPVJWB2BVA,B00SO7HF6I,3.0,1454676014000
AGKASBHYZPGTEPO6LWZPVJWB2BVA,B00MZG6OO8,3.0,1471541996000
AGKASBHYZPGTEPO6LWZPVJWB2BVA,B007JWWUDW,5.0,1471542588000
```

## rating_only -> last_out

`last_out` is short for "leave-last-out data split". The files are based on `rating_only`. We further split the reviews into training set, validation set, and test set for benchmarking.

**Preprocessing**

For each user, the latest review will be used for testing, the second latest review will be used for validation, and all the remaining reviews are used for training.

The data format is the same as `rating_only`.

**Scripts** [[link]](last_out_split.py)

```bash
python last_out_split.py
```

## rating_only -> timestamp

`timestamp` is short for "data split by timestamps". The files are based on `rating_only`.

**Why split by absolute timestamps?** Recommender systems in the real world only access interactions that occurred before a specific timestamp, and aim to predict future interactions. This strategy aligns with real-world scenarios but is not widely used in research. Researchers are encouraged to experiment with this splitting strategy.

**How we choose the timestamps to split?** To be specific, we find two timestamps and split **all the reviews from Amazon Reviews 2023 dataset** in a ratio of 8 : 1 : 1 into training, validation, and test sets. These two timestamps should be used to split data for both pretraining and all downstream evaluation tasks.

**How do we split?** Specially, given a chronological user interaction sequence:
* **Training set**: item interactions with timestamp range (-∞, t_1);
* **Validation set**: item interactions with timestamp range [t_1, t_2);
* **Test set**: item interactions with timestamp range [t_2, +∞).

where t_1 = 1628643414042, t_2 = 1658002729837.

> Note that for each domain, we use the same timestamp to split. Although this strategies may make data splits in different domains have different split ratios, the merit is to be more close to the real recommendation scenarios.

The data format is the same as `rating_only`.

**Scripts** [[link]](timestamp_split.py)

```bash
python timestamp_split.py
```
