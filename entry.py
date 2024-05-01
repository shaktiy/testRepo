import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00" # avoid memory fragmentation on JAX backend.

import keras
import keras_nlp

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from time import time
tqdm.pandas() # progress bar for pandas

from IPython.display import display, Markdown


class Config:
    seed = 42
    dataset_path = "dataset.csv"
    preset = "gemma_2b_en" # name of pretrained Gemma
    sequence_length = 512 # max size of input sequence for training
    batch_size = 1 # size of the input batch in training
    epochs = 10 # number of epochs to train
    
keras.utils.set_random_seed(Config.seed)

def colorize_text(text):
    for word, color in zip(["Question", "Answer", "Total time"], ["blue", "red", "green"]):
        text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

df = pd.read_csv(f"{Config.dataset_path}")

template = "\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
df["prompt"] = df.progress_apply(lambda row: template.format(question=row.question,
                                                             answer=row.answer), axis=1)
data = df.prompt.tolist()

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

x, y, sample_weight = gemma_lm.preprocessor(data[0:1])

gemma_lm.backbone.enable_lora(rank=5)
gemma_lm.summary()

# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = Config.sequence_length 

# Compile the model with loss, optimizer, and metric
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=8e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train model
gemma_lm.fit(data, epochs=Config.epochs, batch_size=Config.batch_size)

class GemmaQA:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.prompt = "\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
        self.gemma_lm = gemma_lm
        
    def query(self, question):
        t_start = time()
        response = self.gemma_lm.generate(
            self.prompt.format(
                question=question,
                answer=""), 
            max_length=self.max_length)
        t_end = time()
        display(Markdown(colorize_text(f"{response}\n\nTotal time: {round(t_end - t_start,2)} sec.")))
        