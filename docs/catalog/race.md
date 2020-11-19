<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>

  <meta itemprop="name" content="race" />
  <meta itemprop="description" content="Race is a large-scale reading comprehension dataset with more than 28,000&#10;passages and nearly 100,000 questions. The dataset is collected from English&#10;examinations in China, which are designed for middle school and high school&#10;students. The dataset can be served as the training and test sets for machine&#10;comprehension.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;race&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/race" />
  <meta itemprop="sameAs" content="https://www.cs.cmu.edu/~glai1/data/race/" />
  <meta itemprop="citation" content="@article{lai2017large,&#10;    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},&#10;    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},&#10;    journal={arXiv preprint arXiv:1704.04683},&#10;    year={2017}&#10;}" />
</div>

# `race`

*   **Description**:

Race is a large-scale reading comprehension dataset with more than 28,000
passages and nearly 100,000 questions. The dataset is collected from English
examinations in China, which are designed for middle school and high school
students. The dataset can be served as the training and test sets for machine
comprehension.

*   **Config description**: Builder config for RACE dataset.

*   **Homepage**:
    [https://www.cs.cmu.edu/~glai1/data/race/](https://www.cs.cmu.edu/~glai1/data/race/)

*   **Source code**:
    [`tfds.text.race.Race`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/text/race/race.py)

*   **Versions**:

    *   **`1.0.0`** (default): No release notes.

*   **Download size**: `24.26 MiB`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    Yes

*   **Features**:

```python
FeaturesDict({
    'answer': Text(shape=(), dtype=tf.string),
    'article': Text(shape=(), dtype=tf.string),
    'question': Text(shape=(), dtype=tf.string),
})
```

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `None`

*   **Citation**:

```
@article{lai2017large,
    title={RACE: Large-scale ReAding Comprehension Dataset From Examinations},
    author={Lai, Guokun and Xie, Qizhe and Liu, Hanxiao and Yang, Yiming and Hovy, Eduard},
    journal={arXiv preprint arXiv:1704.04683},
    year={2017}
}
```

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):
    Missing.

## race/high (default config)

*   **Dataset size**: `127.23 MiB`

*   **Splits**:

Split     | Examples
:-------- | -------:
`'dev'`   | 3,451
`'test'`  | 3,498
`'train'` | 62,445

## race/middle

*   **Dataset size**: `31.35 MiB`

*   **Splits**:

Split     | Examples
:-------- | -------:
`'dev'`   | 1,436
`'test'`  | 1,436
`'train'` | 25,421
