import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

mne.set_log_level("ERROR")


def build_csp_lda(cfg, sfreq):
    def bandpass(X):
        return mne.filter.filter_data(X.astype(np.float64), sfreq, cfg["l_freq"], cfg["h_freq"],
                                      verbose="ERROR")

    return make_pipeline(FunctionTransformer(bandpass), CSP(n_components=cfg["n_components"], log=True),
                         LinearDiscriminantAnalysis())
