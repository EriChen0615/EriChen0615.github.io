---
title: Huggingface datasets
date: 2021-10
tags:
    - Huggingface
    - Programming
---

🤗 Datasets is a library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks.

# Tutorial

## Load a dataset

Before you take the time to download a dataset, it is often helpful to quickly get all the relevant information about a dataset. The [`datasets.load_dataset_builder()`](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset_builder) method allows you to inspect the attributes of a dataset without downloading it.

```python
>>> from datasets import load_dataset_builder
>>> dataset_builder = load_dataset_builder('imdb')
>>> print(dataset_builder.cache_dir)
/Users/thomwolf/.cache/huggingface/datasets/imdb/plain_text/1.0.0/fdc76b18d5506f14b0646729b8d371880ef1bc48a26d00835a7f3da44004b676
>>> print(dataset_builder.info.features)
{'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None)}
>>> print(dataset_builder.info.splits)
{'train': SplitInfo(name='train', num_bytes=33432835, num_examples=25000, dataset_name='imdb'), 'test': SplitInfo(name='test', num_bytes=32650697, num_examples=25000, dataset_name='imdb'), 'unsupervised': SplitInfo(name='unsupervised', num_bytes=67106814, num_examples=50000, dataset_name='imdb')}
```

Once you are happy with the dataset you want, load it in a single line with [`datasets.load_dataset()`](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset):

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset('glue', 'mrpc', split='train')
```

You need to:

1. specify the dataset
2. specify the configuration
3. specify the split [optional]

signature: `load_dataset(<path>, <configuraion>)`

Some datasets, like the [General Language Understanding Evaluation (GLUE)](https://huggingface.co/datasets/glue) benchmark, are actually made up of several datasets. These sub-datasets are called **configurations**, and you must explicitly select one when you load the dataset.

Use `get_dataset_config_names` to **retrieve a list of all the possible configurations** available to your dataset:

```python
from datasets import get_dataset_config_names

configs = get_dataset_config_names("glue")
print(configs)
# ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
```

A **split** is a specific subset of the dataset like `train` and `test`. Make sure you select a split when you load a dataset. If you don’t supply a `split` argument, 🤗 Datasets will only return a dictionary containing the subsets of the dataset.

You can list the split names for a dataset, or a specific configuration, with the [`datasets.get_dataset_split_names()`](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.get_dataset_split_names) method:

## The `Dataset` Object

### Metadata

The [`datasets.Dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset) object contains a lot of useful information about your dataset. For example, call `dataset.info` to return a short description of the dataset, the authors, and even the dataset size. This will give you a quick snapshot of the datasets most important attributes.

```python
>>> dataset.info
DatasetInfo(
    description='GLUE, the General Language Understanding Evaluation benchmark\n(https://gluebenchmark.com/) is a collection of resources for training,\nevaluating, and analyzing natural language understanding systems.\n\n',
    citation='@inproceedings{dolan2005automatically,\n  title={Automatically constructing a corpus of sentential paraphrases},\n  author={Dolan, William B and Brockett, Chris},\n  booktitle={Proceedings of the Third International Workshop on Paraphrasing (IWP2005)},\n  year={2005}\n}\n@inproceedings{wang2019glue,\n  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},\n  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},\n  note={In the Proceedings of ICLR.},\n  year={2019}\n}\n', homepage='https://www.microsoft.com/en-us/download/details.aspx?id=52398',
    license='',
    features={'sentence1': Value(dtype='string', id=None), 'sentence2': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None), 'idx': Value(dtype='int32', id=None)}, post_processed=None, supervised_keys=None, builder_name='glue', config_name='mrpc', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=943851, num_examples=3668, dataset_name='glue'), 'validation': SplitInfo(name='validation', num_bytes=105887, num_examples=408, dataset_name='glue'), 'test': SplitInfo(name='test', num_bytes=442418, num_examples=1725, dataset_name='glue')},
    download_checksums={'https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv': {'num_bytes': 6222, 'checksum': '971d7767d81b997fd9060ade0ec23c4fc31cbb226a55d1bd4a1bac474eb81dc7'}, 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt': {'num_bytes': 1047044, 'checksum': '60a9b09084528f0673eedee2b69cb941920f0b8cd0eeccefc464a98768457f89'}, 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt': {'num_bytes': 441275, 'checksum': 'a04e271090879aaba6423d65b94950c089298587d9c084bf9cd7439bd785f784'}},
    download_size=1494541,
    post_processing_size=None,
    dataset_size=1492156,
    size_in_bytes=2986697
)
```

### Features and columns

A dataset is a table of rows and typed columns. **Querying a dataset returns a Python dictionary** where the keys correspond to column names, and the values correspond to column values:

List the columns names with [`datasets.Dataset.column_names()`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.column_names):

Get detailed information about the columns with `datasets.Dataset.features`:



```python
>>> dataset.shape
(3668, 4)
>>> dataset.num_columns
4
>>> dataset.num_rows
3668
>>> len(dataset)
3668
>>> dataset.column_names
['idx', 'label', 'sentence1', 'sentence2']
>>> dataset.features
{'idx': Value(dtype='int32', id=None),
    'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
    'sentence1': Value(dtype='string', id=None),
    'sentence2': Value(dtype='string', id=None),
}
```

Return even more specific information about a feature like [`datasets.ClassLabel`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.ClassLabel), by calling its parameters `num_classes` and `names`:

```python
>>> dataset.features['label'].num_classes
2
>>> dataset.features['label'].names
['not_equivalent', 'equivalent']
```

### Rows, slices, batches, and columns

Get several rows of your dataset at a time with slice notation or a list of indices:

```python
>>> dataset[:3]
{'idx': [0, 1, 2],
    'label': [1, 0, 1],
    'sentence1': ['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .", 'They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .'],
    'sentence2': ['Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', "Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .", "On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale ."]
}
>>> dataset[[1, 3, 5]]
{'idx': [1, 3, 5],
    'label': [0, 0, 1],
    'sentence1': ["Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .", 'Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .', 'Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .'],
    'sentence2': ["Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .", 'Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .', "With the scandal hanging over Stewart 's company , revenue the first quarter of the year dropped 15 percent from the same period a year earlier ."]
}
```

Querying by the column name will return its values. For example, if you want to only return the first three examples:

```python
>>> dataset['sentence1'][:3]
['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .", 'They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .']
```

## Train with Datasets

By default, all the dataset columns are returned as Python objects. But you can bridge the gap between a Python object and your machine learning framework by setting the format of a dataset. Formatting casts the columns into compatible PyTorch or TensorFlow types.

### Tokenize

See notes in [transformers](transformers.md) for tokenizer

```python
>>> from transformers import BertTokenizerFast
>>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
>>> encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence1']), batched=True)
>>> encoded_dataset.column_names
['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask']
>>> encoded_dataset[0]
{'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
'label': 1,
'idx': 0,
'input_ids': [  101,  7277,  2180,  5303,  4806,  1117,  1711,   117,  2292, 1119,  1270,   107,  1103,  7737,   107,   117,  1104,  9938, 4267, 12223, 21811,  1117,  2554,   119,   102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

### Format

Set the format with [`datasets.Dataset.set_format()`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.set_format), which accepts two main arguments:

1. `type` defines the type of column to cast to. For example, `torch` returns PyTorch tensors and `tensorflow` returns TensorFlow tensors.
2. `columns` specifies which columns should be formatted.

After you set the format, wrap the dataset in a `torch.utils.data.DataLoader` or a `tf.data.Dataset`

```python
>>> import torch
>>> from datasets import load_dataset
>>> from transformers import AutoTokenizer
>>> dataset = load_dataset('glue', 'mrpc', split='train')
>>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
>>> dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True)
...
>>> dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
>>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
>>> next(iter(dataloader))
{'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
                         ...,
                         [1, 1, 1,  ..., 0, 0, 0]]),
'input_ids': tensor([[  101,  7277,  2180,  ...,     0,     0,     0],
                    ...,
                    [  101,  1109,  4173,  ...,     0,     0,     0]]),
'label': tensor([1, 0, 1, 0, 1, 1, 0, 1]),
'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
                         ...,
                         [0, 0, 0,  ..., 0, 0, 0]])}
```

## Evaluate Predictions

🤗 Datasets provides various common and NLP-specific [metrics](https://huggingface.co/metrics) for you to measure your models performance. In this section of the tutorials, you will load a metric and use it to evaluate your models predictions.

You can see what metrics are available with [`datasets.list_metrics()`](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.list_metrics)

### Load Metric

It is very easy to load a metric with 🤗 Datasets. In fact, you will notice that it is very similar to loading a dataset! Load a metric from the Hub with [`datasets.load_metric()`](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_metric)

```python
>>> from datasets import load_metric
>>> metric = load_metric('glue', 'mrpc')
```

This will load the metric associated with the MRPC dataset from the GLUE benchmark.

### Metrics Object

Before you begin using a [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric) object, you should get to know it a little better. As with a dataset, you can return some basic information about a metric. For example, use `datasets.Metric.inputs_description` to get more information about a metrics expected input format and some usage examples:

### Compute metric

Once you have loaded a metric, you are ready to use it to evaluate a models predictions. Provide the model predictions and references to [`datasets.Metric.compute`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric.compute):

```python
>> model_predictions = model(model_inputs)
>>> final_score = metric.compute(predictions=model_predictions, references=gold_references)
```

