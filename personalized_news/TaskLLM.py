import torch

class TaskLLM(torch.nn.Module):
    class GenZ(torch.nn.Module):
        def __init__(self, thing):
            super().__init__()
            self._thing = thing

        def forward(self, *args, **kwargs):
            return self._thing.generate(*args, **kwargs)

    def __init__(self, *, t5=None, opt_factory=None, adapter_name=None, model_id=None):
        import parameterfree
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._adapter_name = adapter_name
        self._transformer = T5ForConditionalGeneration.from_pretrained('t5-base') if t5 is None else t5
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)
        self._device_ids = [ f'cuda:{i}' for i in range(torch.cuda.device_count()) ]
        self._transformer.to(self._device_ids[0])

        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path,
                                                        use_fast=True,
                                                        padding_side='left',
                                                        model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())

    def _loglikelihood(self, data, labels):
        import torch.nn.parallel as P

        replicas = P.replicate(self._transformer, self._device_ids)
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True)
        scatterenc_input_ids = P.scatter(enc.input_ids, self._device_ids)
        scatterenc_attention_mask = P.scatter(enc.attention_mask, self._device_ids)

        labels = self._tokenizer(labels, return_tensors='pt', truncation=True, padding=True).input_ids
        labels[labels == self._tokenizer.pad_token_id] = -100
        scatter_labels = P.scatter(labels, self._device_ids)

        kwargs_lst = [ { 'labels': l } for l in scatter_labels ]

        outputs = P.parallel_apply(modules = replicas[:len(kwargs_lst)],
                                   inputs = list(zip(scatterenc_input_ids, scatterenc_attention_mask)),
                                   kwargs_tup = kwargs_lst)
        loss = P.gather([ o.loss.unsqueeze(0) for o in outputs ], self._device_ids[0])
        weights = torch.Tensor([ i.shape[0] for i in scatterenc_input_ids ]).to(loss.device)

        return torch.dot(weights, loss) / torch.sum(weights)

    def generate(self, data):
        import torch.nn.parallel as P

        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self.eval()
        replicas = P.replicate(self.GenZ(self._transformer), self._device_ids)
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True)
        scatterenc_input_ids = P.scatter(enc.input_ids, self._device_ids)
        scatterenc_attention_mask = P.scatter(enc.attention_mask, self._device_ids)

        kwargs_lst = [ { 'attention_mask': mask,
                         'num_beams': 4,
                         'num_return_sequences': 1,
                         'no_repeat_ngram_size': 1,
                         'remove_invalid_values': True,
                         'max_new_tokens': 80 }
                       for mask in scatterenc_attention_mask
                     ]

        scatter_outputs = P.parallel_apply(modules = replicas[:len(kwargs_lst)],
                                           inputs = scatterenc_input_ids,
                                           kwargs_tup = kwargs_lst)

        return self._tokenizer.batch_decode(P.gather(scatter_outputs, self._device_ids[0]), skip_special_tokens=True)

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
