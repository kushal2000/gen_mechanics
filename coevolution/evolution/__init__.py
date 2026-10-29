"""Embodiment evolution (29 Oct): co-design of grammar hands by evolution with an archive, one shared policy.

Kit-free bookkeeping; the simulator side is the olympics trainer (experiments/12oct_inhand_olympics) and
coevolution/eval/eval_hands.py. One generation:

    train the active population (one shared policy, template scene, from the last checkpoint)
    -> propose: pick parents from the archive, mutate each once with the grammar, build and check the children
    -> evaluate: the active hands and the children, zero-shot, in one scene (eval_hands.py)
    -> admit: record fitness, keep the fittest survivors, admit the best children, write the next population

Parent selection is the only thing the compared methods change (select.py): ``archive`` (the Darwin Goedel
Machine's rule over the whole archive: the baseline), ``niches`` (the same rule over the elite of each niche of hand:
ours), ``greedy`` (the fittest few, no archive) and ``random`` (fresh random hands, no mutation).

    config.py      EvolutionConfig: every setting of a run, saved with it
    descriptors.py what kind of hand a design is: niche cell, genetic vector for novelty
    archive.py     every hand ever made: lineage, fitness history, status
    select.py      parent selection strategies
    variation.py   one grammar mutation + build + checks
    generation.py  the CLI the generation .sub calls: init / propose / admit / status
"""
