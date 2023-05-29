import torch

#debug

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        
        pos = torch.arange(0, maxlen).unsqueeze(1).float()
        d = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(pos * d)
        pe[:, 1::2] = torch.cos(pos * d)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        
    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        token_embedding = token_embedding + self.pos_emb[:, : token_embedding.size(1)]
        return token_embedding
