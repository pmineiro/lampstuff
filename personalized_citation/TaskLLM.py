import parameterfree
import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5=None, opt_factory=None, model_id=None):
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._num_labels = 2
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base' if model_id is None else model_id) if t5 is None else t5
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='left', model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = self._tokenizer(["["], return_tensors='pt').input_ids
        self._one, self._two = self._tokenizer(["1"]).input_ids[0][0], self._tokenizer(["2"]).input_ids[0][0]

    def forward(self, data, *, augment=None):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], -1)
        logits = self._transformer(**enc, decoder_input_ids=decoder_input_ids).logits[:,-1,[self._one,self._two]]
        logprobs = F.log_softmax(logits, dim=1)

        if augment:
            from math import log
            augment_logprobs = self.forward(augment(data))[:,[1,0]]
            return torch.logaddexp(logprobs, augment_logprobs) - log(2)
        else:
            return logprobs

    def predict(self, x, *, augment=None):
        self.eval()
        return self.forward(x, augment = augment)

    def learn(self, x, y, *, augment=None):
        import torch.nn.functional as F

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = self(x, augment = augment)
        loss = F.nll_loss(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)

class PeftTaskLLM(TaskLLM):
    def __init__(self, peft_config, *, t5=None, opt_factory=None, model_id=None):
        from peft import get_peft_model, PeftModel

        super().__init__(t5=t5, opt_factory=opt_factory) # NB: no model_id here
        self._peft_config = peft_config
        if model_id is None:
            self._transformer = get_peft_model(self._transformer, self._peft_config)
        else:
            self._transformer = PeftModel.from_pretrained(self._transformer, model_id)
        self._optim = self._opt_factory(self.parameters())

    def clone(self):
        other = TaskLLM(self._peft_config, t5=self._transformer.base_model.model, opt_factory=self._opt_factory)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._optim = self._opt_factory(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())

        return other
