import numpy as np
import matplotlib.pyplot as plt
import preproc.IOfiles as iof
import preproc.dataFilter as filt
import preproc.dataEditor as edit
import postproc.featuresExtractor as featext
import postproc.featuresPlotter as featplt
import postproc.classifierTools as tools

# Load test and training data
train_data_, true_data_ = iof.read_competition3_dataset1('../datasets/berlindata/comp3_ds1_mi')

l_freq = 9
h_freq = 13

print(train_data_['X'].shape)
print(train_data_['Y'].shape)

y_train = train_data_['Y']
y_true = true_data_['Y']

cls1 = y_train[0, 0]
cls2 = y_train[y_train != cls1][0]

train_data = filt.low_pass(train_data_['X'], lowfreq=13, order=5, sfreq=1000, t_ax=1)
train_data = filt.high_pass(train_data, highfreq=9, order=5, sfreq=1000, t_ax=1)
true_data = filt.low_pass(true_data_['X'], lowfreq=13, order=5, sfreq=1000, t_ax=1)
true_data = filt.high_pass(true_data, highfreq=9, order=5, sfreq=1000, t_ax=1)

train_dat, _ = edit.subsampler(train_data, 50, 1000, t_ax=1)
true_dat, _ = edit.subsampler(true_data, 50, 1000, t_ax=1)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
axs[0].plot(train_data_['X'][0, :, 0])
axs[1].plot(train_data[0, :, 0])
axs[2].plot(true_dat[0, :, 0])
plt.show()

train_dat_cls1 = train_dat[(y_train == cls1).flatten()]
train_dat_cls2 = train_dat[(y_train == cls2).flatten()]
true_dat_cls1 = true_dat[(y_true == cls1).flatten()]
true_dat_cls2 = true_dat[(y_true == cls2).flatten()]

train_dat = {'cls1': train_dat_cls1, 'cls2': train_dat_cls2}
true_dat = {'cls1': true_dat_cls1, 'cls2': true_dat_cls2}

csp_train, W = featext.csp_feat(train_dat, class_names=[0, 1], t_ax=1)
csp_true, W_ = featext.csp_feat(true_dat, class_names=[0, 1], fitted=True, W=W, t_ax=1)

# Calculer Log-var des deux ensembles
train_cls1 = csp_train['cls1'][:, :, [0, -1]]
train_cls2 = csp_train['cls2'][:, :, [0, -1]]
logvar_train_cls1 = featext.logvar_feat(train_cls1, t_ax=1)
logvar_train_cls2 = featext.logvar_feat(train_cls2, t_ax=1)
featplt.features_scatter_plot(logvar_train_cls1, logvar_train_cls2)
train = np.concatenate((logvar_train_cls1, logvar_train_cls2), axis=0)

y_train = np.concatenate((np.full((logvar_train_cls1.shape[0],), 0), np.full((logvar_train_cls2.shape[0],), 1)), axis=0)

# Calculer Log-var des deux ensembles
true_cls1 = csp_true['cls1'][:, :, [0, -1]]
true_cls2 = csp_true['cls2'][:, :, [0, -1]]
logvar_true_cls1 = featext.logvar_feat(true_cls1, t_ax=1)
logvar_true_cls2 = featext.logvar_feat(true_cls2, t_ax=1)
featplt.features_scatter_plot(logvar_true_cls1, logvar_true_cls2)
true = np.concatenate((logvar_true_cls1, logvar_true_cls2), axis=0)

y_true = np.concatenate((np.full((logvar_true_cls1.shape[0],), 0), np.full((logvar_true_cls2.shape[0],), 1)), axis=0)

features_ = np.concatenate((train, true), axis=0)
targets_ = np.concatenate((y_train, y_true), axis=0)

# Sélection des estimateurs
splitted_data = train, true, y_train, y_true
scores, y_pred = tools.estimator_select(None, None, cv=4, ts=None, rs=None, splitted_data=splitted_data,
                                        with_predict=True)
for item in scores:
    print(item)
