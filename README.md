# Finetuning GPT2 model using pre-trained baseline model and tweet_sentiment dataset

This is a simple example of finetuning an LLM model using pre-existing 'tweet_sentiment' [dataset](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction) from huggingface. 

### Steps: 

- import modules - numpy, pandas, datasets, transformers, and evaluate
- load datasets (uses 'dataset' library by hugging face
- run tokenizer (GPT2 tokenizer) to prepare dataset for training. This step break downs raw data into smaller pieces (tokens) so that computation per indexed dictionary word can be efficiently done.
- Randomly split 'training' and 'evaluation' dataset used for training and eveluation metrices generation after training (can be done per epoch).
- Initialise GPT2 base model with three classes/labels for sentiment classification.
- Define/pass evaluation function to trainer to evaluate the model between training epochs.
- Train the model using 'Trainer' method from transformers library. Set training arguments (training batch, evaluation batch, gradient collection steps, and number of epochs as appropriate based on machine specification (e.g. GPU, memory).
- Start training/evaluation



