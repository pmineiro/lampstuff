import parameterfree
from peft import get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPT2Classifier(nn.Module):
    def __init__(self, num_labels, *, gpt2=None, opt_factory=None):
        super().__init__()
        assert num_labels == int(num_labels) and num_labels >= 1
        self._num_labels = num_labels
        self._transformer = AutoModelForCausalLM.from_pretrained('gpt2') if gpt2 is None else gpt2
        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = nn.Linear(hdim, self._num_labels, bias=(self._num_labels==1))
        with torch.no_grad(): self._score.weight.fill_(0)
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='right')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())

    def forward(self, data):
        inputs = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        embeddings = self._transformer(**inputs, output_hidden_states=True).hidden_states[-1]
        scores = self._score(embeddings.float())
        first_zero = inputs['attention_mask'].sum(dim=-1)
        outputs = scores[range(scores.shape[0]), first_zero - 1]
        return F.log_softmax(outputs, dim=1) if self._num_labels > 1 else outputs

    # TODO: doesn't work with custom gpt2
    #def clone(self):
    #    other = GPT2Classifier(self._num_labels)
    #    other.load_state_dict(self.state_dict())
    #    other._optim = other._opt_factory(other.parameters())
    #    other._optim.load_state_dict(self._optim.state_dict())
    #    return other

    def predict(self, x):
        self.eval()
        return torch.exp(self.forward(x)) if self._num_labels > 1 else torch.special.expit(self.forward(x))

    def learn(self, x, y):
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
            self.eval()
            self._optim.zero_grad()
            logprobs = self(x)
            indexed_logprobs = logprobs[range(logprobs.shape[0]), a]
            loss = F.binary_cross_entropy(torch.exp(indexed_logprobs), r)
            loss.backward()
            self._optim.step()
            return loss.item()

class PeftGPT2Classifier(GPT2Classifier):
    def __init__(self, num_labels, peft_config, *, gpt2=None, opt_factory=None):
        super().__init__(num_labels, gpt2=gpt2, opt_factory=opt_factory)
        self._peft_config = peft_config
        self._transformer = get_peft_model(self._transformer, self._peft_config)
        self._optim = self._opt_factory(self.parameters())

    def clone(self):
        other = PeftGPT2Classifier(self._num_labels, self._peft_config, gpt2=self._transformer.base_model.model)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._score.load_state_dict(self._score.state_dict())
        other._optim = other._opt_factory(other.parameters())
        other._optim.load_state_dict(other._optim.state_dict())

        return other
