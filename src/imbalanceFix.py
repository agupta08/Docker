from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
X, y = make_classification(n_classes=25, class_sep=2, weights=[0.1, 0.9],
    n_informative=30, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=10000, random_state=10)
print('Original dataset shape {}'.format(Counter(y)))
sme = SMOTEENN(random_state=42)
X_res, y_res = sme.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))
