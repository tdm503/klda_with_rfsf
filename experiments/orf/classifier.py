import torch
from sentence_transformers import SentenceTransformer
from KLDA import KLDA_E

class Classifier:
    def __init__(self, D, sigma, num_ensembles, seed=0, model_name="sentence-transformers/all-MiniLM-L6-v2", mapping="rff", ridge=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = SentenceTransformer(model_name, device=str(self.device))
        self.D, self.sigma, self.num_ensembles = D, sigma, num_ensembles
        self.seed, self.mapping, self.ridge = seed, mapping, ridge
        self.labels = []
        self.model = None

    def get_embeddings(self, sentences):
        return self.backbone.encode(sentences, convert_to_tensor=True, device=str(self.device))

    def fit(self, df):
        groups = df.groupby("label", sort=True)["text"].apply(list)
        self.labels = groups.index.tolist()
        d = self.backbone.get_sentence_embedding_dimension()
        self.model = KLDA_E(len(self.labels), d, self.D, self.sigma, self.num_ensembles, self.seed, self.device, self.mapping, self.ridge)
        for idx, (_, texts) in enumerate(groups.items()):
            self.model.batch_update(self.get_embeddings(texts), idx)
        self.model.fit()

    def predict(self, sentence):
        return self.labels[self.model.predict(self.get_embeddings([sentence])[0])]
