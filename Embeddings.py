from langchain.embeddings.base import Embeddings
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np


# Define the custom NOMIC embeddings class
class NomicEmbeddings(Embeddings):
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1.5'):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, safe_serialization=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to the model output to get sentence embeddings
        """
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def embed_documents(self, texts):
        """
        Embed a list of documents using the NOMIC model
        """
        embeddings_list = []
        for text in texts:
            if text is None or text.strip() == '':
                embeddings_list.append([0.0] * 768)  # Assuming 768 dimensions
            else:
                # Prepend 'clustering: ' as in your code
                text = 'clustering: ' + text
                inputs = self.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                attention_mask = inputs['attention_mask']
                with torch.no_grad():
                    model_output = self.model(**inputs)
                embedding = self.mean_pooling(model_output, attention_mask)
                embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding = embedding.cpu().numpy()[0].tolist()
                embeddings_list.append(embedding)
        return embeddings_list

    def embed_query(self, text):
        """
        Embed a query text using the NOMIC model
        """
        if text is None or text.strip() == '':
            return [0.0] * 768  # Assuming 768 dimensions
        # Prepend 'clustering: ' as in your code
        text = 'clustering: ' + text
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            model_output = self.model(**inputs)
        embedding = self.mean_pooling(model_output, attention_mask)
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.cpu().numpy()[0].tolist()
        return embedding
