import serial
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pywt
from mne.decoding import CSP

# Configuration de la connexion série
# ser = serial.Serial('COM3', 250000)

# Définir les paramètres des filtres
sfreq = 250
num_channels = 8
nyquist = 0.5 * sfreq  # La fréquence de Nyquist
low_cutoff = 8  # La fréquence de coupure basse du filtre passe-bande.
high_cutoff = 30  # La fréquence de coupure haute du filtre passe-bande.
order = 4
b, a = butter(order, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')

# Définir les variables pour les fonctions de prétraitement et de post-traitement
buffer_size = 500  # Le nombre d'échantillons à mettre dans le tampon.
y_size = 500
window_size = 250  # Le nombre d'échantillons à mettre dans chaque fenêtre.
window_overlap = 125  # Le nombre d'échantillons qui se chevauchent entre les fenêtres.
data_buffer = np.zeros((buffer_size, num_channels))  # Initialisation du tampon
window_buffer = np.zeros((window_size, num_channels))  # Initialisation de la fenêtre.
window_index = 0  # indice de la fenêtre courante.
window_count = 0  # Compteur du nombre de fenêtres traitées
clf = LinearDiscriminantAnalysis()  # Initialisation du classifieur


def wavelet_features(data_, wavelet='db4', level=4):
    # Decompose the signal into wavelet coefficients
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Pad the arrays to ensure they have the same length
    max_len = max([len(c) for c in coeffs])
    cA4, cD4, cD3, cD2, cD1 = [np.pad(c, (0, max_len - len(c)), 'constant') for c in coeffs]

    # Concatenate the wavelet coefficients
    features_ = np.concatenate((cA4, cD4, cD3, cD2, cD1), axis=0)

    return features_


def compute_csp(data_):
    cov_left = np.cov(data_[: window_size // 2, :], rowvar=False)
    cov_right = np.cov(data_[window_size // 2:, :], rowvar=False)

    eig_vals, eig_secs_ = np.linalg.eig(np.dot(np.linalg.inv(cov_left + cov_right), cov_right))
    sort_idx = np.argsort(eig_vals)[::-1]
    W = eig_secs_[:, sort_idx[:data_.shape[1] // 2]]
    X_csp = np.dot(data_, W)
    return X_csp


# Implémentation de la fonction de prétraitement.
def preprocess(data_):
    # Appliquer le filtrage passe-bande aux données.
    filtered_data = filtfilt(b, a, data_, axis=0)

    # Décomposition en ondelettes des données filtrées.
    X = wavelet_features(filtered_data, 'db4', level=4)

    X_csp = compute_csp(X)
    print(X_csp)
    # Calculer la densité spectrale de puissance.
    psd = np.abs(np.fft.fft(filtered_data, axis=0)) ** 2
    psd = psd[:psd.shape[0] // 2, :]  # prendre uniquement la première moitié du spectre (fréquences positives)

    # Calcul du logarithm de la densité spectrale de la puissance psd
    log_psd = np.log10(psd)

    # Remodeler X_csp pour avoir le même nombre de lignes que log_psd

    X_csp = X_csp[:log_psd.shape[0], :]

    #  Concaténer les caractéristiques LOG(PSD) et les CSP
    features_ = np.concatenate((log_psd, X_csp), axis=1)

    return features_


# Définir la fonction de post-traitement des données
def postprocess(data_):
    # Appliquer la classifieur
    pred = clf.predict(data_.reshape(1, -1))

    return pred[0]


def generate_fake_data(n_samples, n_channels):
    time = np.linspace(0, 1, n_samples)
    data_ = np.zeros((n_samples, n_channels))
    for j in range(n_channels):
        freq = np.random.uniform(5, 15)
        noise = np.random.normal(0, 0.1, n_samples)
        signal = np.sin(2 * np.pi * freq * time)
        data_[:, j] = signal + noise
    return data_


if __name__ == '__main__':
    i = 0
    # Function to generate fake data
    data = generate_fake_data(n_samples=2000, n_channels=8)
    while i < 500:
        # Lire les données du port série
        # data = ser.read_until(b'\r\n').strip().decode().split('\t')
        # data = [float(d) for d in data]

        # Ajouter les données au tampon
        window_buffer[window_index, :] = data[i]

        # Mise à jour de l'indice de la fenêtre
        window_index += 1
        #
        # Si le tampon de la fenêtre est plein --> prétraitement --> post-traitement.
        if window_index == window_size:
            # Réinitialiser l'indice de la fenêtre
            window_index = 0
            #
            # Ajouter le tampon fenêtre au tampon des données
            data_buffer[:window_size - window_overlap, :] = window_buffer[window_overlap:, :]
            data_buffer[window_size - window_overlap: window_size, :] = window_buffer[:window_overlap, :]

            # Incrémenter le compteur de la fenêtre
            window_count += 1

            # Prétraitement
            features = preprocess(data_buffer)

        #     # Si le classifieur est trainé, alors --> classification
        #     pred = postprocess(features)
        #     print(pred)  # Remplacer avec le code de contrôle du dispositif externe.
        #
        #     # Add current window to data buffer
        #     data_buffer = np.roll(data_buffer, -window_overlap, axis=0)
        #     data_buffer[window_size - window_overlap:, :] = window_buffer[window_size - window_overlap:, :][::-1, :]
        i += 1
