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
        self._device_ids = [ f'cuda:{i}' for i in range(torch.cuda.device_count()) ]
        self._transformer.to(self._device_ids[0])

        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, 
                                                        use_fast=True, 
                                                        padding_side='left',
                                                        model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = torch.tensor([self._tokenizer.pad_token_id]).unsqueeze(0) 
        self._numbers = [ self._tokenizer([f'{k}']).input_ids[0][0] for k in range(1, 6) ]

    def forward(self, data):
        import torch.nn.functional as F
        import torch.nn.parallel as P

        replicas = P.replicate(self._transformer, self._device_ids)
        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True)
        scatterenc_input_ids = P.scatter(enc.input_ids, self._device_ids)
        scatterenc_attention_mask = P.scatter(enc.attention_mask, self._device_ids)

        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], -1)
        scatter_decoder_input_ids = P.scatter(decoder_input_ids, self._device_ids)

        outputs = P.parallel_apply(replicas[:len(scatter_decoder_input_ids)],
                                   list(zip(scatterenc_input_ids, scatterenc_attention_mask, scatter_decoder_input_ids))
                                   )
        scatter_logits = [ F.log_softmax(v.logits[:,-1,self._numbers], dim=1) for v in outputs ]

        return P.gather(scatter_logits, self._device_ids[0])

    def predict(self, x):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self.eval()
        return self(x)

    def learn(self, x, y):
        import torch.nn.functional as F

        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = self(x)
        # TODO: try different losses ... (?)
        loss = F.nll_loss(output, y.to(output.device))
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
