"""This module contains functions for recording audio, loading audio from 
a wav file, and processing audio signals needed for the ECE 3210 Shazam lab. """

import numpy as np
import pyaudio
from scipy import ndimage, signal
from scipy import io as spio


CHUNK = 1024
FORMAT = pyaudio.paInt16
RATE = 48000
CHANNELS = 1


def record_audio(duration, f_s):
    """Sets up the audio stream and records audio for a given duration.

    Parameters
    ----------
    duration : float
        duration of the recording in seconds
    f_s : int
        sampling rate in Hz

    Returns
    -------
    ndarray
        mono audio signal sampled at f_s Hz and normalized to have a zero mean
    """

    p = pyaudio.PyAudio()
    # print(p.get_default_input_device_info())

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("* recording")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate(frames).astype(float)
    audio = _resample(audio, RATE, f_s)

    audio -= np.mean(audio)

    return audio


def wav_processing(wav_file, f_s):
    """Loads a wav file and resamples it to f_s Hz.

    Parameters
    ----------
    wav_file : str
        path to the wave file
    f_s : int
        desired sampling rate in Hz

    Returns
    -------
    ndarray
        audio signal sampled at f_s Hz and normalized to have a zero mean
    """

    # Load the audio file
    audio_all = spio.wavfile.read(wav_file)
    f_s_orig = audio_all[0]
    audio = audio_all[1].astype(float)

    # combine channels
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # remove the mean
    audio -= np.mean(audio)

    # resample the audio file
    audio = _resample(audio, f_s_orig, f_s)

    return audio


def local_peaks(spectrum, gs):
    """Finds the location of the local peaks in a gs x gs neighborhood.

    Parameters
    ----------
    spectrum : ndarray
        2D array, the spectrogram of the signal
    gs : int
        neighborhood size

    Returns
    -------
    ndarray
        boolean array, the local peaks
    """

    # create a footprint for the maximum filter
    footprint = np.ones((gs, gs))

    # apply the maximum filter to the spectrum
    max_spectrum = ndimage.maximum_filter(spectrum, footprint=footprint)

    # the local peaks are where the original spectrum is equal to the filtered spectrum
    local_peak = spectrum == max_spectrum

    return local_peak


def _resample(x_t, f_s_old, f_s_new):

    # resample the audio file to 8 kHz
    denom = np.gcd(f_s_old, f_s_new)
    L = f_s_old // denom
    M = f_s_new // denom

    x_t = signal.resample_poly(x_t, M, L)

    return x_t


if __name__ == "__main__":
    f_s = 1000
    duration = 10

    audio_test = record_audio(duration, f_s)
    spio.wavfile.write("test.wav", f_s, audio_test.astype(np.int16))
