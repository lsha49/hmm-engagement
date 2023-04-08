import lime
from lime import lime_tabular

explainer = lime_tabular.RecurrentTabularExplainer(XXX, training_labels=XXX, feature_names=['state'],
                                                   discretize_continuous=True,
                                                   class_names=['Non-dropout', 'Dropout'],
                                                   discretizer='decile')
