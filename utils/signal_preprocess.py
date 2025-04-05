import mne
import numpy as np
from scipy.signal import resample

def preprocess_edf(file_path, target_fs=128):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.resample(target_fs)
    data = raw.get_data()
    return data.T 
