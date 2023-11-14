def step_one(rank, world_size):
    import evaluate
    import os
    from PersonalizedNews import train_loader, dev_loader
    from ProgressPrinter import ProgressPrinter
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from TaskLLM import TaskLLM
    from transformers import T5ForConditionalGeneration
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from Util import interleave, set_directory
    import warnings

    k = int(os.environ.get('k', '4'))
    r = int(os.environ.get('r', '5'))
    max_iteration = int(os.environ.get('max_iteration', '5'))
    augment = int(os.environ.get('AUGMENT', '2'))
    model_type = os.environ.get('MODEL_TYPE', 'base')
    batch_size = int(os.environ.get('BATCH_SIZE', '1'))
    inner_batch_size = int(os.environ.get('INNER_BATCH_SIZE', '128'))
    output_dir = os.environ.get('AMLT_OUTPUT_DIR', '.')

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '31337'
    dist.init_process_group(rank=rank, world_size=world_size)

    torch.manual_seed(2112)
    torch.cuda.set_device(rank)

    train = train_loader(batch_size=batch_size, augment=augment, multi=(rank, world_size))
    dev = dev_loader(batch_size=batch_size, multi=(rank, 1))

    if model_type == 'xxl':
        t5 = prepare_model_for_kbit_training(T5ForConditionalGeneration.from_pretrained('google/flan-t5-xxl', load_in_8bit=True, device_map=rank))
    elif model_type == 'base':
        t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(rank)
    else:
        assert False

    taskllm_config = LoraConfig(r=r, task_type=TaskType.SEQ_2_SEQ_LM)
    t5.add_adapter(taskllm_config, "taskllm")
    t5.enable_adapters()

    taskllm = DDP(TaskLLM(t5=t5, adapter_name="taskllm"), device_ids=[rank])
    rouge_metric = evaluate.load('rouge')

    def inner_batch(func, inner_batch_size, inputs):
        from more_itertools import chunked
        return [ func(*ib) for ib in zip(*[ chunked(g, inner_batch_size) for g in inputs ]) ]

    if rank == 0:
        print(f'******** augment = {augment} max_iteration = {max_iteration} model_type = {model_type} *********')
    with ProgressPrinter('iter', f'{k} loss', f'{k} rouge1', f'{k} rouge1 (dev)', silent=(rank > 0)) as printer, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
        warnings.filterwarnings("ignore", message=".*If you want to save 8-bit models.*")
        cumsum = lambda z, acc=0: [0] + [ acc := acc + v for v in z ]

        for iteration in range(max_iteration):
            for istrain, (examples, labels) in interleave(train, dev, sequential=True):
                with torch.no_grad():
                    texts_to_embed = [ [ text[:256]
                                         for text in (' '.join(ex['article'].split()), )
                                       ] +
                                       [ text[:256]
                                         for v in ex['profile']
                                         for text in (' '.join(v['text'].split()), )
                                       ]
                                       for ex in examples
                                     ]
                    embeddings = torch.cat(inner_batch(func = dev.embed,
                                                       inner_batch_size = inner_batch_size,
                                                       inputs = (sum(texts_to_embed, []),)
                                                      ),
                                           dim=0)

                    splits = cumsum(map(len, texts_to_embed))
                    indices = [ torch.topk(embeddings[a,:] @ embeddings[a+1:b,:].T, dim=0, k=k).indices for a, b in zip(splits, splits[1:]) ]
                    prompts = [ dev.prepend_to_prompt(ex, [ ex['profile'][ind] for ind in index.to('cpu').tolist() ])
                                for ex, index in zip(examples, indices) ]
                    guesses = taskllm.module.generate(prompts)
                    scores = rouge_metric.compute(predictions=guesses, references=labels)['rouge1']

                loss = taskllm.module.learn(prompts, labels, using=taskllm) if istrain else None
                printer.addobs(iteration, loss, scores if istrain else None, scores if not istrain else None)

            printer.print()
            printer.autoprint = False
            with set_directory(output_dir):
                if rank == 0:
                    taskllm.module.save_pretrained(f'User_keq{k}_t5{model_type}_step1_iter{iteration}_augment{augment}')

    dist.destroy_process_group()
