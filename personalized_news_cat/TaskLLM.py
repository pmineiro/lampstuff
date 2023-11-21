import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5, choices, adapter_suffix, opt_factory=None):
        import parameterfree
        import re
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._adapter_suffix = adapter_suffix
        self._transformer = t5
        self.set_adapter()
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, 
                                                        use_fast=True, 
                                                        padding_side='left',
                                                        model_max_length=1024)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = torch.tensor([self._tokenizer.pad_token_id]).unsqueeze(0) 
        self._outputs = [ self._tokenizer([c]).input_ids[0][0] for c in choices ]
        assert len(self._outputs) == len(set(self._outputs)), self._outputs
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

        self._step += 1

    def forward(self, data):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1).to(self._transformer.device)
        logits = self._transformer(**enc, decoder_input_ids=decoder_input_ids).logits[:,-1,self._outputs]
        return F.log_softmax(logits, dim=1)

    def predict(self, x, *, ema=False, prior=None):
        self.set_adapter(ema=ema)
        self.eval()
        pred = self(x)
        if prior is not None:
            import torch.nn.functional as F
            return F.log_softmax(pred + prior, dim=1)
        else:
            return pred

    def learn(self, x, y, *, using=None, prior=None):
        import torch.nn.functional as F

        self.set_adapter()
        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        pred = using(x) if using else self(x)
        if prior is not None:
            import torch.nn.functional as F
            loss = F.nll_loss(F.log_softmax(pred + prior, dim=1), y.to(pred.device))
        else:
            loss = F.nll_loss(pred, y.to(pred.device))
        loss.backward()
        self._optim.step()
        self._update_ema()
        return loss.item()

    def save_pretrained(self, model_id, *, ema=True):
        self.set_adapter(ema=ema)
        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
