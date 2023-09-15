def embedder(stuff):
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(stuff, convert_to_tensor=True)
    normalized = torch.nn.functional.normalize(embeddings)
    return normalized

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
    def to_star_schema(obj):
        ds = []
        articles = []
        article_ids = {}
        for ex in obj:
            inp, profile = ex['input'], ex['profile']

            for article in ex['profile']:
                if article['id'] not in article_ids:
                    article_ids[article['id']] = len(articles)
                    article['id'] = len(articles)
                    articles.append(article)
                else:
                    article['id'] = article_ids[article['id']]

            ex['profile'] = [ article['id'] for article in ex['profile'] ]

            ds.append(ex)

        return ds, articles

    def __init__(self, batch_size, *, split):
        import gzip
        import json
        import os

        if not os.path.isfile(f'{split}_outputs.json.gz'):
            download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/{split}/{split}_outputs.json')

        with gzip.open(f'{split}_outputs.json.gz', 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
            assert data['task'] == 'LaMP_1'
            self.labels = { v['id']:v['output'] for v in data['golds']}

        if not os.path.isfile(f'{split}_questions.json.gz'):
            download_and_compress(f'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_1/{split}/{split}_questions.json')

        with gzip.open(f'{split}_questions.json.gz', 'r') as fin:
            self._ds, self._articles = self.to_star_schema(json.loads(fin.read().decode('utf-8')))

        self._num_classes = 2
        self._batch_size = batch_size
        self._embeddings = {}

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def choices(self):
        return [ "[1]", "[2]" ]

    @property
    def batch_size(self):
        return self._batch_size

    def stringify_articles(self, article_ids):
        return [ f'Title: {title}\nAbstract: {abstract}'
                 for article_id in article_ids
                 for article in (self._articles[article_id],)
                 for title in (' '.join(article['title'].strip().split()),)
                 for abstract in (' '.join(article['abstract'].strip().split()),)
               ]

    def embeddings(self, article_ids):
        import torch

        for article_id in article_ids:
            if article_id not in self._embeddings:
                stuff = self.stringify_articles([article_id])
                self._embeddings[article_id] = embedder(stuff)

        return torch.cat([ self._embeddings[article_id] for article_id in article_ids ], dim=0)

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
    return DataLoader(batch_size, split='train')

def dev_loader(batch_size):
    return DataLoader(batch_size, split='dev')
