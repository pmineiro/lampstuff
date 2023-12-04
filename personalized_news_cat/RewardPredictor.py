import parameterfree
import torch

class RewardPredictor(torch.nn.Module):
    def __init__(self, *, t5, opt_factory=None, adapter_suffix=None, model_id=None):
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        import re

        super().__init__()
        self._adapter_suffix = adapter_suffix
        self._transformer = t5
        self.set_adapter()
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, 
                                                        use_fast=True, 
                                                        padding_side='left',
                                                        model_max_length=512)
        hdim = getattr(self._transformer.config, 'n_embd', False) or getattr(self._transformer.config, 'hidden_size')
        self._score = torch.nn.Linear(hdim, 1)
        with torch.no_grad():
            if model_id:
                state_dict = torch.load(f'{model_id}/score_layer.pth', map_location='cpu')
                self._score.load_state_dict(state_dict)
            else:
                self._score.weight.fill_(0)
                self._score.bias.fill_(0)
        self._score.to(self._transformer.device)

        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())

        self._score_ema = torch.nn.Linear(hdim, 1)
        self._score_ema.to(self._transformer.device)
        self._decoder_input_ids = torch.Tensor([self._tokenizer.pad_token_id]).long().unsqueeze(0).to(self._transformer.device)
        self._params_to_copy = { n: re.sub(r'\.raw_', '.ema_', n) for n, _ in self.named_parameters() if f'.raw_{self._adapter_suffix}.' in n }
        self._step = 0
        self._update_ema()

    def set_adapter(self, *, ema=False):
        prefix = 'ema' if ema else 'raw'
        self._transformer.set_adapter(f'{prefix}_{self._adapter_suffix}')

    def _update_ema(self):
        decay = 1 / (1 + self._step)

        with torch.no_grad():
            state_dict = self.state_dict()
            for n, othern in self._params_to_copy.items():
                state_dict[othern].lerp_(state_dict[n], decay)

            self._score_ema.weight.lerp_(self._score.weight, decay)
            self._score_ema.bias.lerp_(self._score.bias, decay)

        self._step += 1

    def forward(self, data):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1).to(self._transformer.device)
        last_hidden_state = self._transformer(**enc, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state[:,-1,:]
        return self._score(last_hidden_state.float())

    def predict(self, x, *, ema=False):
        self.set_adapter(ema=ema)
        self.eval()
        return self(x)

    def learn(self, x, y, *, using=None):
        import torch.nn.functional as F

        self.set_adapter()
        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = using(x) if using else self(x)
        loss = F.binary_cross_entropy_with_logits(output, y.to(output.device))
        loss.backward()
        self._optim.step()
        self._update_ema()
        return loss.item()

    def save_pretrained(self, model_id, *, ema=True):
        import torch

        self.set_adapter(ema=ema)
        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
        wut = self._score_ema if ema else self._score
        torch.save(wut.state_dict(), f'{model_id}/score_layer.pth')
