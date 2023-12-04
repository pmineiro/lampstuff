import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5, opt_factory=None, adapter_suffix=None):
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
                                                        model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
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

    def forward(self, data, labels):
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True)
        input_ids = enc.input_ids.to(self._transformer.device)
        attention_mask = enc.attention_mask.to(self._transformer.device)
        labels = self._tokenizer(labels, return_tensors='pt', truncation=True, padding=True).input_ids
        labels[labels == self._tokenizer.pad_token_id] = -100
        labels = labels.to(self._transformer.device)
        return self._transformer(input_ids = input_ids, attention_mask = attention_mask, labels = labels).loss

    def generate(self, data, *, ema=False):
        self.set_adapter(ema=ema)
        self.eval()
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True).to(self._transformer.device)
        output_sequences = self._transformer.generate(input_ids=enc['input_ids'], 
                                                      attention_mask=enc['attention_mask'], 
                                                      num_beams=4,
                                                      num_return_sequences=1,
                                                      no_repeat_ngram_size=1,
                                                      remove_invalid_values=True,
                                                      max_new_tokens=80)

        return self._tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    def learn(self, data, labels, *, using=None):
        self.set_adapter()
        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        loss = using(data, labels) if using else self(data, labels)
        loss.backward()
        self._optim.step()
        self._update_ema()
        return loss.item()

    def save_pretrained(self, model_id, *, ema=True):
        self.set_adapter(ema=ema)
        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
