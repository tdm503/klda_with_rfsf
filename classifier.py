import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from KLDAver2 import KLDA_E

class Classifier:
    def __init__(self, D,level, sigma, num_ensembles, seed=0, model_name='facebook/bart-base'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = SentenceTransformer(model_name).to(self.device)
        if self.backbone.tokenizer.pad_token is None:
            self.backbone.tokenizer.pad_token = self.backbone.tokenizer.eos_token

        self.D = D
        self.level = level
        self.sigma = sigma
        self.num_ensembles = num_ensembles
        self.seed = seed
        self.model = None
        self.labels = []

    def fit(self, df):
        grouped = df.groupby('label')['text'].apply(list).reset_index()
        self.labels = grouped['label'].tolist()
        class_texts = grouped['text'].tolist()
        
        num_classes = len(self.labels)
        d = self.backbone.get_sentence_embedding_dimension()
        self.model = KLDA_E(num_classes, d, self.D,self.level, self.sigma, self.num_ensembles, self.seed, self.device)
        for label, texts in zip(self.labels, class_texts):
            text_embeddings = self.get_embeddings(texts)
            self.model.batch_update(text_embeddings, self.labels.index(label))
        self.model.fit()

    def get_embeddings(self, sentences):
        embeddings = self.backbone.encode(sentences, convert_to_tensor=True, device=self.device)
        return embeddings

    def predict(self, sentence):
        input_embedding = self.get_embeddings(sentence)
        idx = self.model.predict(input_embedding)
        return self.labels[idx]
