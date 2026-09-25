import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer


dataset = load_dataset("mteb/tweet_sentiment_extraction")
df = pd.DataFrame(dataset["train"])


# Loading the dataset to train our model
# dataset = load_dataset("mteb/tweet_sentiment_extraction")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
##added for higher batch size
model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = False


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   return metric.compute(predictions=predictions, references=labels)



training_args = TrainingArguments(

   output_dir="training_runs",
   #evaluation_strategy="epoch",
   per_device_train_batch_size=6,  #was 1 # Reduce batch size if needed 
   per_device_eval_batch_size=6,    #was 1 # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=8 ,  #was 4 # Accumulate gradients over more steps. to avoide memory issues
   num_train_epochs=5,
   )


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   compute_metrics=compute_metrics,
   tokenizer=tokenizer,
   data_collator=data_collator,

)

trainer.train()


trainer.evaluate()
