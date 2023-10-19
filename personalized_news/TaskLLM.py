import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5=None, opt_factory=None, adapter_name=None, model_id=None):
        import parameterfree
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._adapter_name = adapter_name
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base') if t5 is None else t5
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, 
                                                        use_fast=True, 
                                                        padding_side='left',
                                                        model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())

    def _loglikelihood(self, data, labels):
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True)
        input_ids = enc.input_ids.to(self._transformer.device)
        attention_mask = enc.attention_mask.to(self._transformer.device)
        labels = self._tokenizer(labels, return_tensors='pt', truncation=True, padding=True).input_ids
        labels[labels == self._tokenizer.pad_token_id] = -100
        labels = labels.to(self._transformer.device)
        return self._transformer(input_ids = input_ids, attention_mask = attention_mask, labels = labels).loss

    def generate(self, data):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

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

    def learn(self, data, labels):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        loss = self._loglikelihood(data, labels)
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
