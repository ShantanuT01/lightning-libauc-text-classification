# Usage

## Install
```shell
pip install git+https://github.com/ShantanuT01/lightning-libauc-text-classification.git
```

## Train
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import libauc.losses
import libauc.optimizers
import pandas as pd

from lightning_libauc.trainer import LibAUCTrainer

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

loss_fn = libauc.losses.AUCMLoss()
optimizer = libauc.optimizers.PESG(model.parameters(), loss_fn=loss_fn)

train_df = pd.read_csv("train.csv")

trainer = LibAUCTrainer(model, tokenizer, loss_fn, optimizer)
trainer.train(
    train_df,
    epochs=3,
    batch_size=8,
    text_col="text",
    label_col="label",
)
```

## Evaluate
```python
metrics_df = trainer.evaluate(
    testing_df=pd.read_csv("test.csv"),
    text_col="text",
    label_col="label",
    batch_size=8,
)
print(metrics_df.head())
```
