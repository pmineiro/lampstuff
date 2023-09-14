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

class TrainLoader(object):
    def __init__(self, batch_size):
        import gzip
        import json
        import os

        if not os.path.isfile('train_outputs.json.gz'):
            download_and_compress('https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/train/train_outputs.json')

        with gzip.open('train_outputs.json.gz', 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
            assert data['task'] == 'LaMP_1'
            self.labels = { v['id']:v['output'] for v in data['golds']}


        if not os.path.isfile('train_questions.json.gz'):
            download_and_compress('https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/train/train_questions.json')

        with gzip.open('train_questions.json.gz', 'r') as fin:
            self._ds = json.loads(fin.read().decode('utf-8'))

        self._num_classes = 2
        self._batch_size = batch_size

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        def items():
            from more_itertools import chunked
            import torch
            
            for batch in chunked(torch.randperm(len(self._ds), device='cpu').tolist(), self._batch_size):
                inputs = [ ex['input'] for ind in batch for ex in (self._ds[ind],) ]
                profiles = [ ex['profile'] for ind in batch for ex in (self._ds[ind],) ]
                labels = [ self.labels[ex['id']] for ind in batch for ex in (self._ds[ind],) ]
                yield (inputs, profiles, labels)

        return items()

def train_loader(batch_size):
    return TrainLoader(batch_size)
