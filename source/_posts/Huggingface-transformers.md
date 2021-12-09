---
title: Huggingface transformers
date: 2021-10
tags:
---
# Quick Tour

Let’s have a quick look at the huggingface 🤗 Transformers library features. The library downloads pretrained models for Natural Language Understanding (NLU) tasks, such as analyzing the sentiment of a text, and Natural Language Generation (NLG), such as completing a prompt with new text or translating in another language.

## Getting started with a pipeline

Let’s see how this work for sentiment analysis (the other tasks are all covered in the [task summary](https://huggingface.co/transformers/task_summary.html)):

```python
>>> from transformers import pipeline
>>> classifier = pipeline('sentiment-analysis')

>>> classifier('We are very happy to show you the 🤗 Transformers library.')
[{'label': 'POSITIVE', 'score': 0.9998}]

>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.",
...            "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

```
transformers.pipeline(*task: str*, *model: Optional = None*, *config: Optional[Union[str, transformers.configuration_utils.PretrainedConfig]] = None*, *tokenizer: Optional[Union[str, transformers.tokenization_utils.PreTrainedTokenizer]] = None*, *feature_extractor: Optional[Union[str, SequenceFeatureExtractor]] = None*, *framework: Optional[str] = None*, *revision: Optional[str] = None*, *use_fast: bool = True*, *use_auth_token: Optional[Union[bool, str]] = None*, *model_kwargs: Dict[str, Any] = {}*, ***kwargs*) → transformers.pipelines.base.Pipeline
```

[[SOURCE\]](https://huggingface.co/transformers/_modules/transformers/pipelines.html#pipeline)

Utility factory method to build a [`Pipeline`](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.Pipeline).

Pipelines are made of:

> - A [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html) in charge of mapping raw textual input to token.
> - A [model](https://huggingface.co/transformers/main_classes/model.html) to make predictions from the inputs.
> - Some (optional) post processing for enhancing model’s output.

See [documentation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline)

## Pretrained Model

The model and tokenizer are created using the `from_pretrained` method

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
>>> model_name = "distilbert-base-uncased-finetuned-sst-2-english"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Using the tokenizer

We mentioned the tokenizer is responsible for the preprocessing of your texts. First, it will split a given text in words (or part of words, punctuation symbols, etc.) usually called tokens. There are multiple rules that can govern that process (you can learn more about them in the [tokenizer summary](https://huggingface.co/transformers/tokenizer_summary.html)), which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules as when the model was pretrained.

The second step is to convert those tokens into numbers, to be able to build a tensor out of them and feed them to the model. To do this, the tokenizer has a **vocab**, which is the part we download when we instantiate it with the `from_pretrained` method, since we need to use the same vocab as when the model was pretrained.

To apply these steps on a given text, we can just feed it to our tokenizer:

```python
>>> inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
```

This returns a dictionary string to list of ints. It contains the [ids of the tokens](https://huggingface.co/transformers/glossary.html#input-ids), as mentioned before, but also additional arguments that will be useful to the model. Here for instance, we also have an [attention mask](https://huggingface.co/transformers/glossary.html#attention-mask) that the model will use to have a better understanding of the sequence:

```python
>>> print(inputs)
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

You can pass a list of sentences directly to your tokenizer. If your goal is to send them through your model as a batch, you probably want to pad them all to the same length, truncate them to the maximum length the model can accept and get tensors back. You can specify all of that to the tokenizer:

```python
>>> pt_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="pt"
... )
```

The padding is automatically applied on the side expected by the model (in this case, on the right), with the padding token the model was pretrained with. The attention mask is also adapted to take the padding into account:

```python
>>> for key, value in pt_batch.items():
...     print(f"{key}: {value.numpy().tolist()}")
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
```

### Using the model

Once your input has been preprocessed by the tokenizer, you can send it directly to the model. As we mentioned, it will contain all the relevant information the model needs. If you’re using a TensorFlow model, you can pass the dictionary keys directly to tensors, for a PyTorch model, you need to unpack the dictionary by adding `**`.

```python
>>> pt_outputs = pt_model(**pt_batch)
```

Once your model is fine-tuned, you can save it with its tokenizer in the following way:

```python
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

You can then load this model back using the [`from_pretrained()`](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModel.from_pretrained) method by passing the directory name instead of the model name. One cool feature of 🤗 Transformers is that you can easily switch between PyTorch and TensorFlow: **any model saved as before can be loaded back either in PyTorch or TensorFlow**. If you are loading a saved PyTorch model in a TensorFlow model, use [`from_pretrained()`](https://huggingface.co/transformers/model_doc/auto.html#transformers.TFAutoModel.from_pretrained) like this:

```
from transformers import TFAutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```

and if you are loading a saved TensorFlow model in a PyTorch model, you should use the following code:

```
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```

### Customizing the model

If you want to change how the model itself is built, you can define a custom configuration class. Each architecture comes with its own relevant configuration. For example, [`DistilBertConfig`](https://huggingface.co/transformers/model_doc/distilbert.html#transformers.DistilBertConfig) allows you to specify parameters such as the hidden dimension, dropout rate, etc for DistilBERT. If you do core modifications, like changing the hidden size, you won’t be able to use a pretrained model anymore and will need to train from scratch. You would then instantiate the model directly from this configuration.

```python
>>> from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
>>> config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
>>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
>>> model = DistilBertForSequenceClassification(config)
```

For something that only changes the head of the model (for instance, the number of labels), **you can still use a pretrained model for the body.** For instance, let’s define a classifier for 10 different labels using a pretrained body. Instead of creating a new configuration with all the default values just to change the number of labels, we can instead pass any argument a configuration would take to the `from_pretrained()` method and it will update the default configuration appropriately:

```python
>>> from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
>>> model_name = "distilbert-base-uncased"
>>> model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
>>> tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

# Fine-tuning a pretrained model

In PyTorch, there is no generic training loop so the 🤗 Transformers library provides an API with the class [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) to let you fine-tune or train a model from scratch easily. Then we will show you how to alternatively write the whole training loop in PyTorch.

## Preparing the dataset

We will use the [🤗 Datasets](https://github.com/huggingface/datasets/) library to download and preprocess the IMDB datasets. We will go over this part pretty quickly. Since the focus of this tutorial is on training, you should refer to the 🤗 Datasets [documentation](https://huggingface.co/docs/datasets/) or the [Preprocessing data](https://huggingface.co/transformers/preprocessing.html) tutorial for more information.

First, we can use the `load_dataset` function to download and cache the dataset:

```python
from datasets import load_dataset

raw_datasets = load_dataset("imdb")
```

To preprocess our data, we will need a tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(sentences, padding="max_length", truncation=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# transform the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

```

## Fine-tuning with Trainer API

First, let’s define our model:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

Then, to define our [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer), we will need to instantiate a [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments). This class contains all the hyperparameters we can tune for the [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) or the flags to activate the different training options it supports. Let’s begin by using all the defaults, the only thing we then have to provide is a directory in which the checkpoints will be saved:

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer")
```

Then we can instantiate a [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) like this:

```python
from transformers import Trainer

trainer = Trainer(
    model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
)
```

To fine-tune our model, we just need to call

```python
trainer.train()
```

To have the [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) compute and report metrics, we need to give it a `compute_metrics` function that takes predictions and labels (grouped in a namedtuple called [`EvalPrediction`](https://huggingface.co/transformers/internal/trainer_utils.html#transformers.EvalPrediction)) and return a dictionary with string items (the metric names) and float values (the metric values).

The 🤗 Datasets library provides an easy way to get the common metrics used in NLP with the `load_metric` function. here we simply use accuracy. 

```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

To check if this works on practice, let’s create a new [`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) with our fine-tuned model:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.evaluate()
```

If you want to fine-tune your model and regularly report the evaluation metrics (for instance at the end of each epoch), here is how you should define your training arguments:

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
```

See the documentation of [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments) for more options.

## Fine-tuning with native PyTorch

We just need to apply a bit of post-processing to our `tokenized_datasets` before doing that to:

- remove the columns corresponding to values the model does not expect (here the `"text"` column)
- rename the column `"label"` to `"labels"` (because the model expect the argument to be named `labels`)
- set the format of the datasets so they return PyTorch Tensors instead of lists.

Our `tokenized_datasets` has one method for each of those steps:

```python
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

We can easily define our dataloaders:

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Next, we define our model:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

We are almost ready to write our training loop. The only two things to add are an **optimizer** and a **learning rate scheduler**. The default optimizer used by the `Trainer` is `AdamW`

```python
from transformers import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```

We will want to use the GPU if we have access to one. To do this, we define a `device`:

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

We now are ready to train! To get some sense of when it will be finished, we add a progress bar over our number of training steps, using the tqdm library.

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Now to check the results, we need to write the **evaluation loop**. Like in the [trainer section](https://huggingface.co/transformers/training.html#trainer) we will use a metric from the datasets library. Here we accumulate the predictions at each batch before computing the final result when the loop is finished.

```
metric= load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```