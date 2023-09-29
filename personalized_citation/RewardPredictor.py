import parameterfree
import torch

# https://github.com/huggingface/trl/blob/6b73adc9001cffebc4a13f735fe67dd3149b46b9/trl/models/modeling_base.py#L487

class RewardPredictor(torch.nn.Module):
    def __init__(self, *, t5=None, opt_factory=None, adapter_name=None, model_id=None):
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._adapter_name = adapter_name
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base') if t5 is None else t5
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='left', model_max_length=512)

        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = torch.nn.Linear(hdim, 1)
        with torch.no_grad():
            if model_id:
                state_dict = torch.load(f'{model_id}/score_layer.pth')
                self._score.load_state_dict(state_dict)
            else:
                self._score.weight.fill_(0)
                self._score.bias.fill_(0)

        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = torch.Tensor([self._tokenizer.pad_token_id]).long().unsqueeze(0).to(self._transformer.device)

    def forward(self, data, *, augment=None):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1)
        embeddings = self._transformer(**enc, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state[:,-1,:]
        outputs = self._score(embeddings)

        if augment:
            augment_outputs = self(augment(data))
            return (1/2) * (outputs + augment_outputs)
        else:
            return outputs

    def predict(self, x, *, augment=None):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self.eval()
        return self(x, augment=augment)

    def learn(self, x, y, *, augment=None):
        import torch.nn.functional as F

        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

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

        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
        torch.save(self._score.state_dict(), f'{model_id}/score_layer.pth')
