def download_and_compress(url):
    import gzip
    import requests
    from urllib.parse import urlparse, unquote
    from tqdm import tqdm

    filename = unquote(urlparse(url).path.split("/")[-1])
    filename += '.gz'

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
    def _augment_examples(exs):
        import re

        for n, ex in enumerate(exs):
            m = re.search(r'For an author who has written the paper with the title "(.*?)", which reference is related?', ex['input'])
            ex['title'] = m.group(1)
            ex['dsind'] = n
            m = re.search(r'without explanation. \[1\]: "(.*?)" \[2\]: "(.*)"', ex['input'])
            ex['ref1'] = m.group(1)
            ex['ref2'] = m.group(2)

    def __init__(self, batch_size, *, split, max_index):
        import gzip
        import json
        import os
        from math import inf
        from sentence_transformers import SentenceTransformer

        if not os.path.isfile(f'{split}_outputs.json.gz'):
            download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/{split}/{split}_outputs.json')

        with gzip.open(f'{split}_outputs.json.gz', 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
            assert data['task'] == 'LaMP_1'
            self._labels = { v['id']:v['output'] for v in data['golds']}

        if not os.path.isfile(f'{split}_questions.json.gz'):
            download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/{split}/{split}_questions.json')

        with gzip.open(f'{split}_questions.json.gz', 'r') as fin:
            self._ds = json.loads(fin.read().decode('utf-8'))
            self._augment_examples(self._ds)

        self._num_classes = 2
        self._batch_size = batch_size
        self._max_index = inf if max_index is None else max_index
        assert self._max_index >= self._batch_size
        self._embeddings = {}
        self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def choices(self):
        return [ "[1]", "[2]" ]

    @property
    def batch_size(self):
        return self._batch_size

    def embed(self, stuff):
        import torch
        embeddings = self._embedder.encode(stuff, convert_to_tensor=True)
        normalized = torch.nn.functional.normalize(embeddings)
        return normalized

    def append_to_title(self, example, stuff):
        import re

        m = re.match(r'^(.*?the title )(".*?")(, which reference.*)', example['input'])
        return m.group(1) + m.group(2) + f' and {stuff}' + m.group(3)
                   

    def __iter__(self):
        def items():
            from more_itertools import chunked
            import torch

            permlen = min(self._max_index, len(self._ds))

            for batch in chunked(torch.randperm(permlen, device='cpu').tolist(), self._batch_size):
                examples = [ ex for ind in batch for ex in (self._ds[ind],) ]
                labels = [ self._labels[ex['id']] for ind in batch for ex in (self._ds[ind],) ]
                yield (examples, labels)

        return items()

def train_loader(batch_size, *, max_index=None):
    return DataLoader(batch_size, split='train', max_index=max_index)

def dev_loader(batch_size, *, max_index=None):
    return DataLoader(batch_size, split='dev', max_index=max_index)
