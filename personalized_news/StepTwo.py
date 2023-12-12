def step_two(rank, world_size):
    from copy import deepcopy
    import evaluate
    import os
    from PersonalizedNews import train_loader, dev_loader
    from ProgressPrinter import ProgressPrinter
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from RewardPredictor import RewardPredictor
    from SimpleRegret import SimpleRegretHypercubeSampler
    from TaskLLM import TaskLLM
    from transformers import T5ForConditionalGeneration
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms.join import Join
    from torch.nn.parallel import DistributedDataParallel as DDP
    from Util import interleave, set_directory
    import warnings

    k = int(os.environ.get('k', '4'))
    r = int(os.environ.get('r', '5'))
    max_iteration = int(os.environ.get('max_iteration', '5'))
    step1_iter = os.environ.get('STEP1_ITER', '0_augment8')
    augment = int(os.environ.get('AUGMENT', '1'))
    gamma = float(os.environ.get('GAMMA', '10'))
    model_type = os.environ.get('MODEL_TYPE', 'base')
    batch_size = int(os.environ.get('BATCH_SIZE', '1'))
    learn_batch_size = int(os.environ.get('LEARN_BATCH_SIZE', str(batch_size)))
    gradfree_batch_size = int(os.environ.get('GRAD_FREE_BATCH_SIZE', '128'))
    n_randos = int(os.environ.get('N_RANDOS', '128'))
    output_dir = os.environ.get('AMLT_OUTPUT_DIR', '.')

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '31337'
    dist.init_process_group(rank=rank, world_size=world_size)

    torch.manual_seed(8675309)

    train = train_loader(batch_size=batch_size, augment=augment, multi=(rank, world_size))
    dev = dev_loader(batch_size=batch_size, multi=(rank, 1))

    if model_type == 'xxl':
        t5 = prepare_model_for_kbit_training(T5ForConditionalGeneration.from_pretrained('google/flan-t5-xxl', load_in_8bit=True, device_map=rank))
    elif model_type == 'base':
        t5 = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(rank)
    else:
        assert False
    taskllm_model_id = f'User_keq{k}_t5{model_type}_step1_iter{step1_iter}'
    t5.load_adapter(taskllm_model_id, 'raw_taskllm')
    t5.load_adapter(taskllm_model_id, 'ema_taskllm')

    rhat_config = LoraConfig(r=r, task_type=TaskType.SEQ_2_SEQ_LM)
    t5.add_adapter(rhat_config, "raw_rhat")
    t5.add_adapter(deepcopy(rhat_config), "ema_rhat")
    t5.enable_adapters()

    taskllm = TaskLLM(t5=t5, adapter_suffix="taskllm")
    rewardpredictor = DDP(RewardPredictor(t5=t5, adapter_suffix="rhat"), device_ids=[rank], find_unused_parameters=True)
    rouge_metric = evaluate.load('rouge')

    gumbel = torch.distributions.gumbel.Gumbel(0,1)
    def randomized_similarity(embeddings, nsamples):
        scores = embeddings[0,:] @ embeddings[1:,:].T
        temperature = scores[0].item() - scores[min(scores.shape[0]-1, 4)].item()
        gumbel_shape = torch.Size([nsamples, scores.shape[0]])
        gumbels = temperature * gumbel.sample(gumbel_shape).to(scores.device)
        safek = min(k, scores.shape[0])
        return torch.unique(torch.topk(scores.unsqueeze(0) + gumbels, dim=1, k=safek).indices, sorted=False, dim=0)

    def inner_batch(func, inner_batch_size, inputs):
        from more_itertools import chunked
        return [ func(*ib) for ib in zip(*[ chunked(g, inner_batch_size) for g in inputs ]) ]

    if rank == 0:
        print(f'******** augment = {augment} max_iteration = {max_iteration} model_type = {model_type} *********')

    if model_type == 'xxl':
        # ugh ... wtf ... warnings not ignored (?) ... join context manager is sus
        import sys
        sys.stderr = open('/dev/null', 'w')

    with ProgressPrinter('iter', f'{k} loss', f'{k} rouge',  f'{k} ema rouge', f'{k} ema rouge (dev)', 'nsamps', silent=(rank > 0)) as printer, warnings.catch_warnings():
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
                                                       inner_batch_size = gradfree_batch_size,
                                                       inputs = (sum(texts_to_embed, []),)
                                                      ),
                                           dim=0)
                    splits = cumsum(map(len, texts_to_embed))
                    randos = [ randomized_similarity(embeddings[a:b,:], n_randos) for a, b in zip(splits, splits[1:]) ]
                    prompts = [ [ dev.prepend_to_prompt(ex, [ ex['profile'][ind] for ind in indices ])
                                  for indices in rando.to('cpu').tolist()
                                ]
                                for ex, rando in zip(examples, randos)
                              ]
                    rhats = torch.cat(inner_batch(func = rewardpredictor.module.predict,
                                                  inner_batch_size = gradfree_batch_size,
                                                  inputs = (sum(prompts, []),)
                                                 ),
                                      dim=0)
                    emarhats = torch.cat(inner_batch(func = lambda p: rewardpredictor.module.predict(p, ema=True),
                                                     inner_batch_size = gradfree_batch_size,
                                                     inputs = (sum(prompts, []),)
                                                    ),
                                         dim=0)
                    splits = cumsum(map(len, prompts))
                    samples = [ SimpleRegretHypercubeSampler(rhats[a:b].view(1, -1), gamma=gamma) for a, b in zip(splits, splits[1:]) ]
                    actionind = [ [ exploit.item() ] + [ n for n, observed in enumerate(explore) if observed > 0 ]
                                  for exploit, exploreraw in samples
                                  for explore in (exploreraw[0].tolist() if istrain else [], )
                                ]
                    nsamps = [ len(aind) for aind in actionind ]
                    guessprompts = [ [ prompt[a] for a in aind ] for prompt, aind in zip(prompts, actionind) ]
                    guesses = sum(inner_batch(func = taskllm.generate,
                                              inner_batch_size = gradfree_batch_size,
                                              inputs = (sum(guessprompts, []),)
                                             ),
                                  [])
                    emaactions = [ SimpleRegretHypercubeSampler(emarhats[a:b].view(1, -1), gamma=gamma)[0].item() for a, b in zip(splits, splits[1:]) ]
                    emaguessprompts = [ prompt[a] for prompt, a in zip(prompts, emaactions) ]
                    emaguesses = sum(inner_batch(func = taskllm.generate,
                                                 inner_batch_size = gradfree_batch_size,
                                                 inputs = (emaguessprompts,)
                                                ),
                                     [])
                    splits = cumsum(map(len, guessprompts))
                    rewards = sum( ( rouge_metric.compute(predictions=guesses[a:b], 
                                                          references=[label]*(b-a),
                                                          use_aggregator=False)['rouge1']
                                     for a, b, label in zip(splits, splits[1:], labels)
                                  ),
                                  [])
                    greedyrewards = rouge_metric.compute(predictions=[guesses[a] for a in splits[:-1]],
                                                         references = labels,
                                                         use_aggregator=False)['rouge1']
                    emagreedyrewards = rouge_metric.compute(predictions=emaguesses,
                                                            references = labels,
                                                            use_aggregator=False)['rouge1']

                if istrain:
                    rhatprompts = sum(guessprompts, [])
                    with Join([rewardpredictor]):
                        predlosses = inner_batch(func = lambda a, b: (len(a),
                                                                      rewardpredictor.module.learn(a, torch.Tensor([ [ r ] for r in b ]), using=rewardpredictor)
                                                                     ),
                                                 inner_batch_size = learn_batch_size,
                                                 inputs = (rhatprompts, rewards))
                    predloss = sum(n * v for n, v in predlosses) / sum(n for n, v in predlosses)
                else:
                    predloss = None

                greedyreward = torch.Tensor(greedyrewards, device='cpu').float().mean().item()
                emagreedyreward = torch.Tensor(emagreedyrewards, device='cpu').float().mean().item()
                nsamps = torch.Tensor(nsamps, device='cpu').float().mean().item() if istrain else None

                printer.addobs(iteration,
                               predloss,
                               greedyreward if istrain else None,
                               emagreedyreward if istrain else None,
                               emagreedyreward if not istrain else None,
                               nsamps)

            printer.print()
            printer.autoprint = False
            with set_directory(output_dir):
                if rank == 0:
                    rewardpredictor.module.save_pretrained(f'User_keq{k}_t5{model_type}_step2_iter{iteration}_augment{augment}')

    dist.destroy_process_group()
