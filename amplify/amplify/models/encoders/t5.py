import torch.nn as nn
from transformers import T5Model, T5Tokenizer


class T5(nn.Module):
    def __init__(self, size, frozen=True, return_all_tokens=False, seq_len=32, **kwargs):
        super(T5, self).__init__()
        print(f'Loading T5-{size}...')
        self.tokenizer = T5Tokenizer.from_pretrained(f"t5-{size}")
        self.model = T5Model.from_pretrained(f"t5-{size}").encoder
        self.return_all_tokens = return_all_tokens
        self.max_len = seq_len
        self.seq_len = seq_len if return_all_tokens else 1
        
        embed_dims = {'small': 512, 'base': 768, 'large': 1024}
        self.embed_dim = embed_dims.get(size, 512)  # Default to small if size not found
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f'T5 num params: {num_params}')
    
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.eval()
        
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def preprocess_text(self, text):
        return self.tokenizer.batch_encode_plus(
            text,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
            truncation=True
        )
    
    def forward(self, text, attention_mask=None):
        processed_prompt = self.preprocess_text(text)
        input_ids = processed_prompt['input_ids'].to(self.device)
        attention_mask = processed_prompt['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.return_all_tokens:
            return outputs.last_hidden_state
        else:
            return outputs.last_hidden_state[:, 0, :]  # CLS token representation. Last hidden state has shape (batch_size, seq_len, hidden_size)