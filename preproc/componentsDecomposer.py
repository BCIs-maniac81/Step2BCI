import numpy as np
from picard import picard
from ica import ica1  # infomax
from sklearn.decomposition import FastICA, PCA


def ica_decomposing(observation=None, ncomp=None, ica_method="picard", ortho=False, X_mean=False, max_iter=100,
                    random_state=0):
    # Paramètre X (Mélange) --> shape(n_samples, n_components)
    X = observation.copy()
    if ica_method.lower() == 'picard':
        if ortho:
            print("Utilisation de la méthode Picard-O...")
        else:
            print("Utilisation de la méthode Picard...")
        X = X.T
        if X_mean:
            K, W, S_, X_mean = picard(X=X, n_components=ncomp, ortho=ortho, return_X_mean=X_mean,
                                      max_iter=max_iter, random_state=random_state)
            # Calculer la matrice de mélange estimée
            w = np.dot(W, K)
            A_ = np.dot(w.T, np.linalg.inv(np.dot(w, w.T)))
            return S_, A_, X_mean  # return shape (n_components, n_samples)
        else:
            K, W, S_ = picard(X=X, n_components=ncomp, ortho=ortho, return_X_mean=X_mean, max_iter=max_iter,
                              random_state=random_state)
            # Calculer la matrice de mélange estimée
            w = np.dot(W, K)
            A_ = np.dot(w.T, np.linalg.inv(np.dot(w, w.T)))
            return S_, A_  # return shape(n_components, n_samples)
    elif ica_method.lower() == "infomax":
        print("Utilisation de la méthode Infomax...")
        X = X.T
        A_, S_, W = ica1(X, ncomp=ncomp, verbose=False)
        return S_, A_  # return shape(n_components, n_samples)
    elif ica_method.lower() == "fastica":
        print("Utilisation de la méthode FastICA...")
        ica = FastICA(n_components=ncomp, max_iter=max_iter, random_state=random_state, whiten='arbitrary-variance')
        S_ = ica.fit_transform(X)
        A_ = ica.mixing_
        S_ = S_.T
        return S_, A_  # return shape(n_components, n_samples)
    else:
        print("type ICA inconnu")
        msg = "Type ICA inconnu"
        return msg
