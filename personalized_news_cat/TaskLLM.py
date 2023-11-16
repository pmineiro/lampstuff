import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5, choices, opt_factory=None, adapter_name=None, model_id=None):
        import parameterfree
        from transformers import T5ForConditionalGeneration, AutoTokenizer

        super().__init__()
        self._adapter_name = adapter_name
        self._transformer = t5
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, 
                                                        use_fast=True, 
                                                        padding_side='left',
                                                        model_max_length=512)
        self._opt_factory = parameterfree.COCOB if opt_factory is None else opt_factory
        self._optim = self._opt_factory(self.parameters())
        self._decoder_input_ids = torch.tensor([self._tokenizer.pad_token_id]).unsqueeze(0) 
        self._outputs = [ self._tokenizer([c]).input_ids[0][0] for c in choices ]
        assert len(self._outputs) == len(set(self._outputs)), self._outputs

    def set_adapter(self):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

    def forward(self, data):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], 1).to(self._transformer.device)
        logits = self._transformer(**enc, decoder_input_ids=decoder_input_ids).logits[:,-1,self._outputs]
        return F.log_softmax(logits, dim=1)

    def predict(self, x):
        self.set_adapter()
        self.eval()
        return self(x)

    def learn(self, x, y, *, using=None):
        import torch.nn.functional as F

        self.set_adapter()
        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = using(x) if using else self(x)
        loss = F.nll_loss(output, y.to(output.device))
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        self.set_adapter()
        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
