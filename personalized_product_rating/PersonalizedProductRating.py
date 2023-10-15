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
            m = re.search(r'What is the score of the following review on a scale of 1 to 5\? just answer with 1, 2, 3, 4, or 5 without further explanation. review: ',
                          ex['input'])
            ex['review'] = ex['input'][m.end():]
            ex['dsind'] = n

    def __init__(self, batch_size, *, split, max_index, timesplit, augment=False):
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
        self._augment = augment

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def num_raw_examples(self):
        return min(self._max_index, len(self._ds))

    @property
    def num_examples(self):
        return self.num_raw_examples * (2 if self._augment else 1)

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

    @staticmethod
    def prepend_to_prompt(example, profile_examples):
        # TODO: number of characters >= number of tokens, so this truncation is conservative

        maxlen = 256 // min(1, len(profile_examples))
        preamble = ', and '.join([ f'{profex["score"]} is the score for "{text:.{maxlen-6}s}"'
                                   for profex in profile_examples
                                   for text in (' '.join(profex["text"].split()),)
                                 ])
        return f'{preamble}\n\n{example["input"]}'

    def rewrite_input(self, ex, newreview):
        import re

        try:
            m = re.match(
                  r'(^What is the score of the following review on a scale of 1 to 5\? just answer with 1, 2, 3, 4, or 5 without further explanation. review: )(.*)$',
                  ex['input'],
                  re.DOTALL)
            ex['input'] = m.group(1) + newreview
        except:
            print(f'wtf {ex["input"]}')
            raise

    def swap_with_profile(self, ex):
        from copy import deepcopy
        import torch

        profs = [ (n, v) for n, v in enumerate(ex['profile']) if v['text'] != ex['review'] ]
        nprof = len(profs)
        if nprof and self._labels:
            rawindex = torch.randint(high=nprof, size=(1,), device='cpu').item()
            index = profs[rawindex][0]
            copyex = deepcopy(ex)
            copyex['review'] = ex['profile'][index]['text']
            self.rewrite_input(copyex, copyex['review'])
            copyex['profile'][index]['text'] = ex['review']
            copyex['profile'][index]['score'] = self._labels[ex['id']]
            label = ex['profile'][index]['score']
            return copyex, label
        else:
            return None

    def __iter__(self):
        def items():
            from more_itertools import chunked
            import torch

            for batch in chunked(torch.randperm(self.num_raw_examples, device='cpu').tolist(), self._batch_size):
                examples = [ ex for ind in batch for ex in (self._ds[ind],) ]
                labels = [ self._labels[ex['id']] for ex in examples ] if self._labels else [None]*len(examples)
                if self._augment:
                    moreexamples, morelabels = zip(*[ v for ex in examples for v in (self.swap_with_profile(ex),) if v is not None ])
                    examples.extend(moreexamples)
                    labels.extend(morelabels)

                yield (examples, labels)

        return items()

def train_loader(batch_size, *, max_index=None, timesplit=False, augment=False):
    return DataLoader(batch_size, split='train', max_index=max_index, timesplit=timesplit, augment=augment)

def dev_loader(batch_size, *, max_index=None, timesplit=False):
    return DataLoader(batch_size, split='dev', max_index=max_index, timesplit=timesplit)

def test_loader(batch_size, *, max_index=None, timesplit=False):
    return DataLoader(batch_size, split='test', max_index=max_index, timesplit=timesplit)
