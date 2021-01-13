from . import reg_pruner, l1_pruner, oracle_pruner

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'GReg-1': reg_pruner,
    'GReg-2': reg_pruner,
    'L1': l1_pruner,
    'Oracle': oracle_pruner,
}