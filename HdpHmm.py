import numpy as np
import pandas as pd
import bayesian_hmm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


hmm = bayesian_hmm.HDPHMM(XXX, sticky=True)
hmm.initialise(k=XX)
results = hmm.mcmc(n=XX, burn_in=XX, ncores=XX, save_every=XX, verbose=True)
hmm.print_probabilities()

# accessing HMM parameters
map_index = results['chain_loglikelihood'].index(min(results['chain_loglikelihood']))
parameters_map = results['parameters'][map_index]
eEmissions = parameters_map['p_emission']
eEmissions