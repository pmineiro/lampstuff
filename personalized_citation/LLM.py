import torch

class GPT2Classifier(torch.nn.Module):
    def __init__(self, num_labels, *, gpt2=None):
        import parameterfree
        from transformers import AutoModelForCausalLM, AutoTokenizer


        super().__init__()
        assert num_labels == int(num_labels) and num_labels >= 1
        self._num_labels = num_labels
        self._transformer = AutoModelForCausalLM.from_pretrained('gpt2') if gpt2 is None else gpt2
        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = torch.nn.Linear(hdim, self._num_labels, bias=(self._num_labels==1))
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='right')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._optim = parameterfree.COCOB(self.parameters())

    def forward(self, data):
        import torch.nn.functional as F

        inputs = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        embeddings = self._transformer(**inputs, output_hidden_states=True).hidden_states[-1]
        scores = self._score(embeddings)
        first_zero = inputs['attention_mask'].sum(dim=-1)
        logits = scores[range(scores.shape[0]), first_zero - 1]
        return F.log_softmax(logits, dim=1) if self._num_labels > 1 else logits

    def clone(self):
        import parameterfree

        other = GPT2Classifier(self._num_labels)
        other.load_state_dict(self.state_dict())
        other._optim = parameterfree.COCOB(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())
        return other

    def predict(self, x):
        self.eval()
        return torch.exp(self.forward(x)) if self._num_labels > 1 else torch.special.expit(self.forward(x))

    def learn(self, x, y):
        import torch.nn.functional as F

        self.train()
        self._optim.zero_grad()
        output = self(x)
        loss = F.nll_loss(output, y) if self._num_labels > 1 else F.binary_cross_entropy_with_logits(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def bandit_learn(self, x, a, r):
        import torch.nn.functional as F

        self.train()
        self._optim.zero_grad()
        output = self(x)
        indexed_output = output[range(output.shape[0]), a if self._num_labels > 1 else 0]
        loss = F.binary_cross_entropy_with_logits(indexed_output, r)
        loss.backward()
        self._optim.step()
        return loss.item()

class PeftGPT2Classifier(GPT2Classifier):
    def __init__(self, num_labels, peft_config, *, gpt2=None):
        import parameterfree
        from peft import get_peft_model

        super().__init__(num_labels, gpt2=gpt2)
        self._peft_config = peft_config
        self._transformer = get_peft_model(self._transformer, self._peft_config)
        self._optim = parameterfree.COCOB(self.parameters())

    def clone(self):
        import parameterfree

        other = PeftGPT2Classifier(self._num_labels, self._peft_config, gpt2=self._transformer.base_model.model)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._score.load_state_dict(self._score.state_dict())
        other._optim = parameterfree.COCOB(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())

        return other
