## DAVID'S R&D SPACE

## XLNet
**XLNet** is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs [Transformer-XL](https://arxiv.org/abs/1901.02860) as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking. [Forked from here](https://github.com/zihangdai/xlnet)

Required packages
```sh
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import XLNetForSequenceClassification, XLNetTokenizer, Trainer, TrainingArguments
```
Genre Dataset encodings and labels using pytorch
```sh
class GenreDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```
Training arguments for model. Issues: Computation cannot be handled on my GPU
```sh
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

## Multinomial Logistic Regression Analysis
**Multinomial Logistic Regression** is a classification method that generalizes logistic regression to multiclass problems, i.e. with more than two possible discrete outcomes. That is, it is a model that is used to predict the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variables (which may be real-valued, binary-valued, categorical-valued, etc.).
The problem: The data for psychology, romance, sports, and travel genre is smaller respectively compared to the other genres, causing imbalance in outcomes.
The solution: We use Multinomial Logistic Regression to predict the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variables (which may be real-valued, binary-valued, categorical-valued, etc.). Then we fine tune the model with OpenAI to include the data from the other genres to improve the accuracy of the model (specifcially for the smaller genres).

## WordCloud Analysis and Modelling (no need for modelling since Multinomial Logistic Regression analysis is already done)
**WordCloud** is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites. In this article, you will learn how to create a word cloud in Python.
top_words.json for top 15 most common words in each genre summary.
