import torch

class ZeroShotClassifier(torch.nn.Module):
    def __init__(self, *, gpt2=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__()
        self._transformer = AutoModelForCausalLM.from_pretrained('gpt2' if gpt2 is None else gpt2)
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='right')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def logprobs(self, multiplechoices):
        allquestions = [ f'Question: {problem}\nAnswer: {c}'
                         for (problem, choices) in multiplechoices
                         for c in choices ]
        input_ids = self._tokenizer(allquestions, return_tensors='pt', padding='longest')
        dev_input_ids = input_ids.to(self._transformer.device)
        output = self._transformer(**dev_input_ids)
        output_logits = torch.nn.functional.softmax(output.logits, dim=-1)

        shift_logits = output_logits[:, :-1, :]
        shift_labels = dev_input_ids['input_ids'][:, 1:].unsqueeze(2)
        logits = torch.gather(output_logits, dim=2, index=shift_labels)
        mask = dev_input_ids['attention_mask'][:, 1:].unsqueeze(2)

        nproblems = len(multiplechoices)
        nchoices = len(multiplechoices[0][1])

        return torch.bmm(mask.float().transpose(1, 2), logits).view(nproblems, nchoices)

    def forward(self, multiplechoices):
        guess_indices = torch.argmax(self.logprobs(multiplechoices), dim=1)
        guesses = [ choices[ind]
                    for (_, choices), ind in zip(multiplechoices, guess_indices.to('cpu'))
                  ]

        return guesses

class FewShotClassifier(torch.nn.Module):
    def __init__(self, *, gpt2=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__()
        self._transformer = AutoModelForCausalLM.from_pretrained('gpt2') if gpt2 is None else gpt2
        self._tokenizer = AutoTokenizer.from_pretrained(self._transformer.config._name_or_path, use_fast=True, padding_side='right')
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _getlogprobs(self, texts, nproblems, nchoices):
        input_ids = self._tokenizer(texts, return_tensors='pt', padding='longest')
        dev_input_ids = input_ids.to(self._transformer.device)
        output = self._transformer(**dev_input_ids)
        output_logits = torch.nn.functional.log_softmax(output.logits, dim=-1)

        shift_logits = output_logits[:, :-1, :]
        shift_labels = dev_input_ids['input_ids'][:, 1:].unsqueeze(2)
        logits = torch.gather(output_logits, dim=2, index=shift_labels)
        shift_mask = dev_input_ids['attention_mask'][:, 1:]

        px = torch.bmm(shift_mask.float().unsqueeze(2).transpose(1, 2), logits).view(nproblems, nchoices + 1)
        return px[:,:-1] - px[:,[-1]]

    def logprobs(self, multiplechoices, shots):
        nproblems = len(multiplechoices)
        nchoices = len(multiplechoices[0][1])

        allquestions = [ f'User profile; {preamble}\n\nQuestion: {problem}\nAnswer: {c}'
                         for (problem, choices), examples in zip(multiplechoices, shots)
                         for preamble in ('\n'.join(examples),)
                         for c in choices + ['']
                       ]

        return self._getlogprobs(allquestions, nproblems, nchoices)

    def forward(self, multiplechoices, shots):
        guess_indices = torch.argmax(self.logprobs(multiplechoices, shots), dim=1)
        guesses = [ choices[ind]
                    for (_, choices), ind in zip(multiplechoices, guess_indices.to('cpu'))
                  ]

        return guesses, guess_indices

    def predict(self, multiplechoices, shots):
        self.eval()
        return self(multiplechoices, shots)

class PEFTFewShotClassifier(FewShotClassifier):
    def __init__(self, peft_config, *, gpt2=None):
        import parameterfree
        from peft import IA3Config, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(gpt2=gpt2)
        self._peft_config = peft_config
        self._transformer = get_peft_model(self._transformer, self._peft_config)
        self._optim = parameterfree.COCOB(self.parameters())

    def bandit_learn(self, x, a, r):
        import torch.nn.functional as F

        multiplechoices, shots = x

        self.eval()
        self._optim.zero_grad()
        logprobs = F.log_softmax(self.logprobs(multiplechoices, shots), dim=1)
        indexed_logprobs = logprobs[range(logprobs.shape[0]), a]
        loss = F.binary_cross_entropy(torch.exp(indexed_logprobs), r)
        #loss = -torch.dot(indexed_logprobs, r) / r.shape[0]
        loss.backward()
        self._optim.step()
        return loss.item()
