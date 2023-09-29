import parameterfree
import torch

class RewardPredictor(torch.nn.Module):
    def __init__(self, *, t5=None, opt_factory=None, model_id=None):
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._num_labels = 1
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base' if model_id is None else model_id) if t5 is None else t5
        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = torch.nn.Linear(hdim, self._num_labels, bias=(self._num_labels==1))
        with torch.no_grad():
            if model_id:
                state_dict = torch.load(f'{model_id}/score_layer.pth')
                self._score.weight.copy_(state_dict['fc1.weight'])
                self._score.bias.copy_(state_dict['fc1.bias'])
            else:
                self._score.weight.fill_(0)
                self._score.bias.fill_(0)

        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='left', model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = torch.Tensor([self._tokenizer.pad_token_id]).long().unsqueeze(0).to(self._transformer.device)

    @staticmethod
    def logitsumexpit(a, b):
        from math import log

        logpa = -torch.logaddexp(torch.Tensor([0]).to(a.device), -a)
        logpb = -torch.logaddexp(torch.Tensor([0]).to(b.device), -b) 
        logmeanp = torch.logaddexp(logpa, logpb) - log(2)
        return logmeanp - torch.log1p(-torch.exp(torch.clamp(logmeanp, min=-18, max=-1e-6)))

    def forward(self, data, *, augment=None):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1)
        embeddings = self._transformer(**enc, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state[:,-1,:]
        outputs = self._score(embeddings)

        if augment:
            augment_outputs = self(augment(data))
            # TODO: average probabilities instead of logits (?)
            # seems about the same ...
            return (1/2) * (outputs + augment_outputs)
            #return self.logitsumexpit(outputs, augment_outputs)
        else:
            return outputs

    def predict(self, x, *, augment=None):
        self.eval()
        return self(x, augment=augment)

    def learn(self, x, y, *, augment=None):
        import torch.nn.functional as F

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = self(x, augment=augment)
        loss = F.binary_cross_entropy_with_logits(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        import torch

        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
        torch.save(self._score.state_dict(), f'{model_id}/score_layer.pth')

class PeftRewardPredictor(RewardPredictor):
    def __init__(self, peft_config, *, t5=None, opt_factory=None, model_id=None):
        from peft import get_peft_model

        super().__init__(t5=t5, opt_factory=opt_factory, model_id=model_id)
        self._peft_config = peft_config
        if model_id is None:
            self._transformer = get_peft_model(self._transformer, self._peft_config)
        else:
            self._transformer = PeftModel.from_pretrained(self._transformer, model_id)
        self._optim = self._opt_factory(self.parameters())

    def clone(self):
        other = RewardPredictor(self._peft_config, t5=self._transformer.base_model.model, opt_factory=self._opt_factory)
        other._transformer.load_state_dict(self._transformer.state_dict())
        other._score.load_state_dict(self._score.state_dict())
        other._optim = other._opt_factory(other.parameters())
        other._optim.load_state_dict(self._optim.state_dict())

        return other
