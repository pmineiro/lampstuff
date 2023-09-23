import parameterfree
import torch

optimizer = lambda params: parameterfree.COCOB(params)

class T5Classifier(torch.nn.Module):
    def __init__(self, num_labels, *, t5=None):
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        assert num_labels == 2

        super().__init__()
        assert num_labels == int(num_labels) and num_labels >= 1
        self._num_labels = num_labels
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base') if t5 is None else t5
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='left', model_max_length=512)
        self._optim = optimizer(self.parameters())
        self._decoder_input_ids = self._tokenizer(["["], return_tensors='pt').input_ids
        self._one, self._two = self._tokenizer(["1"]).input_ids[0][0], self._tokenizer(["2"]).input_ids[0][0]

    def forward(self, data):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], -1)
        logits = self._transformer(**enc, decoder_input_ids=decoder_input_ids).logits[:,-1,[self._one,self._two]]
        return F.log_softmax(logits, dim=1) if self._num_labels > 1 else logits

    def clone(self):
        other = T5Classifier(self._num_labels)
        other.load_state_dict(self.state_dict())
        other._optim = optimizer(other.parameters())
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
        from peft import get_peft_model

        super().__init__(num_labels, t5=t5)
        self._peft_config = peft_config
        self._transformer = get_peft_model(self._transformer, self._peft_config)
        self._optim = optimizer(self.parameters())

    def clone(self):
        other = PeftT5Classifier(self._num_labels, self._peft_config, t5=self._transformer.base_model.model)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._score.load_state_dict(self._score.state_dict())
        other._optim = optimizer(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())

        return other
