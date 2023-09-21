import torch

class T5Classifier(torch.nn.Module):
    def __init__(self, num_labels, *, t5=None):
        import parameterfree
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        assert num_labels == int(num_labels) and num_labels >= 1
        self._num_labels = num_labels
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base') if t5 is None else t5
        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = torch.nn.Linear(hdim, self._num_labels, bias=(self._num_labels==1))
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='left', model_max_length=512)
        self._optim = parameterfree.COCOB(self.parameters())
        self._decoder_input_ids = torch.Tensor([self._tokenizer.pad_token_id]).long().unsqueeze(0).to(self._transformer.device)

    def forward(self, data):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1)
        embeddings = self._transformer(**enc, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state[:,-1,:]
        logits = self._score(embeddings)
        return F.log_softmax(logits, dim=1) if self._num_labels > 1 else logits

    def clone(self):
        import parameterfree

        other = T5Classifier(self._num_labels)
        other.load_state_dict(self.state_dict())
        other._optim = parameterfree.COCOB(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())
        return other

    def predict(self, x):
        self.eval()
        return torch.exp(self.forward(x)) if self._num_labels > 1 else torch.special.expit(self.forward(x))

    def learn(self, x, y):
        import torch.nn.functional as F

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = self(x)
        loss = F.nll_loss(output, y) if self._num_labels > 1 else F.binary_cross_entropy_with_logits(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def bandit_learn(self, x, a, r):
        if self._num_labels == 1:
            return self.learn(x, r)
        else:
            import torch.nn.functional as F

            self.eval()
            self._optim.zero_grad()
            logprobs = self(x)
            indexed_logprobs = logprobs[range(logprobs.shape[0]), a]
            loss = F.binary_cross_entropy(torch.exp(indexed_logprobs), r)
            loss.backward()
            self._optim.step()
            return loss.item()

class PeftT5Classifier(T5Classifier):
    def __init__(self, num_labels, peft_config, *, t5=None):
        import parameterfree
        from peft import get_peft_model

        super().__init__(num_labels, t5=t5)
        self._peft_config = peft_config
        self._transformer = get_peft_model(self._transformer, self._peft_config)
        self._optim = parameterfree.COCOB(self.parameters())

    def clone(self):
        import parameterfree

        other = PeftT5Classifier(self._num_labels, self._peft_config, t5=self._transformer.base_model.model)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._score.load_state_dict(self._score.state_dict())
        other._optim = parameterfree.COCOB(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())

        return other
