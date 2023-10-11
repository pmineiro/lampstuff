def download_and_compress(url, *, timesplit=False):
    import gzip
    import re
    import requests
    from urllib.parse import urlparse, unquote
    from tqdm import tqdm

    filename = unquote(urlparse(url).path.split("/")[-1])
    filename += '.gz'
    if timesplit:
        filename = f'time_{filename}'
        url = re.sub(r'downloads/LaMP/', r'downloads/LaMP/time/', url)

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with gzip.open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    assert total_size_in_bytes > 0 and progress_bar.n == total_size_in_bytes

class DataLoader(object):
    @staticmethod
    def _annotate_examples(exs):
        import re

        for n, ex in enumerate(exs):
            # TODO
            m = re.search(r'For an author who has written the paper with the title "(.*?)", which reference is related?', ex['input'])
            ex['review'] = m.group(1)
            ex['dsind'] = n

    def __init__(self, batch_size, *, split, max_index, timesplit):
        import gzip
        import json
        import os
        from math import inf
        from sentence_transformers import SentenceTransformer

        extra = "time_" if timesplit else ""

        if split != 'test':
            if not os.path.isfile(f'{extra}{split}_outputs.json.gz'):
                download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/{split}/{split}_outputs.json', timesplit=timesplit)

            with gzip.open(f'{extra}{split}_outputs.json.gz', 'r') as fin:
                data = json.loads(fin.read().decode('utf-8'))
                assert data['task'] == 'LaMP_3'
                self._labels = { v['id']:v['output'] for v in data['golds']}
        else:
            self._labels = None

        if not os.path.isfile(f'{extra}{split}_questions.json.gz'):
            download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/{split}/{split}_questions.json', timesplit=timesplit)

        with gzip.open(f'{extra}{split}_questions.json.gz', 'r') as fin:
            self._ds = json.loads(fin.read().decode('utf-8'))
            self._annotate_examples(self._ds)

        self._num_classes = 5
        self._batch_size = batch_size
        self._max_index = inf if max_index is None else max_index
        assert self._max_index >= self._batch_size
        self._embedder = SentenceTransformer('all-mpnet-base-v2')

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def num_examples(self):
        return min(self._max_index, len(self._ds))

    @property
    def choices(self):
        return [ f'{k}' for k in range(1,6) ]

    @property
    def batch_size(self):
        return self._batch_size

    def embed(self, stuff):
        import torch
        embeddings = self._embedder.encode(stuff, convert_to_tensor=True)
        normalized = torch.nn.functional.normalize(embeddings)
        return normalized

    def prepend_to_prompt(self, example, profile_examples):
        preamble = ', and '.join([ f'{profex["score"]} is the score for "{profex["text"]}"' for profex in profile_examples ])
        return f'{premable}\n{example["input"]}'
                   
    def __iter__(self):
        def items():
            from more_itertools import chunked
            import torch

            for batch in chunked(torch.randperm(self.num_examples, device='cpu').tolist(), self._batch_size):
                examples = [ ex for ind in batch for ex in (self._ds[ind],) ]
                labels = [ self._labels[ex['id']] for ind in batch for ex in (self._ds[ind],) ] if self._labels else [None]*len(examples)

                yield (examples, labels)

        return items()

def train_loader(batch_size, *, max_index=None, timesplit=False):
    return DataLoader(batch_size, split='train', max_index=max_index, timesplit=timesplit)

def dev_loader(batch_size, *, max_index=None, timesplit=False):
    return DataLoader(batch_size, split='dev', max_index=max_index, timesplit=timesplit)

def test_loader(batch_size, *, max_index=None, timesplit=False):
    return DataLoader(batch_size, split='test', max_index=max_index, timesplit=timesplit)
