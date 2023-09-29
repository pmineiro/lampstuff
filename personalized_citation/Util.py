def interleave(a, b):
    from math import inf
    
    atot, btot = a.num_examples, b.num_examples
    aiter, biter = a.__iter__(), b.__iter__()
    aelem, belem = next(aiter), next(biter)
    anum, bnum = 1, 1

    while anum != inf and bnum != inf:
        if anum * btot <= bnum * atot:
            yield (True, aelem)
            try:
                aelem = next(aiter)
                anum += 1
            except StopIteration:
                anum = inf
        else:
            yield (False, belem)
            try:
                belem = next(biter)
                bnum += 1
            except StopIteration:
                bnum = inf
