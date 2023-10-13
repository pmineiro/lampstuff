import torch

class TaskLLM(torch.nn.Module):
    def __init__(self, *, t5=None, opt_factory=None, adapter_name=None, model_id=None):
        import parameterfree
        from LogisticRegression import ResidualLogisticRegressor
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
        self._decoder_input_ids = torch.tensor([self._tokenizer.pad_token_id]).unsqueeze(0) 
        self._numbers = [ self._tokenizer([f'{k}']).input_ids[0][0] for k in range(1, 6) ]
        self._userbias = ResidualLogisticRegressor(in_features=6, out_features=5, depth=2, model_id=model_id)

    def forward(self, data, *, counts=None):
        import torch.nn.functional as F

        enc = self._tokenizer(data, return_tensors='pt', truncation=True, padding=True).to(self._transformer.device)
        decoder_input_ids = self._decoder_input_ids.expand(enc.input_ids.shape[0], -1)
        logits = self._transformer(**enc, decoder_input_ids=decoder_input_ids).logits[:,-1,self._numbers]
        if counts is not None:
            logits += self._userbias(counts)
        return F.log_softmax(logits, dim=1)

    def predict(self, x, *, counts=None):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self.eval()
        return self(x, counts=counts)

    def learn(self, x, y, *, counts=None):
        import torch.nn.functional as F

        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        # self.train() gives bad results ... (?)
        self.eval()
        self._optim.zero_grad()
        output = self(x, counts=counts)
        # TODO: try different losses ... (?)
        loss = F.nll_loss(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        if self._adapter_name:
            self._transformer.set_adapter(self._adapter_name)

        self._transformer.save_pretrained(model_id)
        self._tokenizer.save_pretrained(model_id)
        self._userbias.save_pretrained(model_id)
