# Benchmark Scripts

Based on the released Amazon Review 2023 dataset, we provide scripts to preprocess raw data into standard train/validation/test splits to encourage benchmarking recommendation models.

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
│   ├── last_out_seq/   # convenient for sequential rec
│   └── timestamp_seq/
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

`timestamp` is short for "data split by timestamps". The files are based on `rating_only`. We further split the reviews into training set, validation set, and test set based on the timestamps of each review.

**Preprocessing**

To simulate a more realistic evaluation for recommender systems, we split the reviews according to timestamps. Specifically, we setup two timestamps as the intervals that roughly split all the reviews by 8:1:1.

* Training set: < 1628643414042
* Validation set: < 1658002729837, >= 1628643414042
* Test set: >= 1658002729837

Note that for each domain, we use the same timestamp to split. Although this strategies may make data splits in different domains have different split ratios, the merit is to be more close to the real recommendation scenarios.

The data format is the same as `rating_only`.

**Scripts** [[link]](timestamp_split.py)

```bash
python timestamp_split.py
```
