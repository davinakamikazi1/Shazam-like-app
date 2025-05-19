import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from audio_utils import record_audio, wav_processing, local_peaks
import h5py
import json
import argparse
import wave
import pyaudio
import glob
import os


def compute_spectrogram(signal_in, fs=1.0, nperseg=512, noverlap=None, nfft=None):
    if noverlap is None:
        noverlap = nperseg // 8
    if nfft is None:
        nfft = nperseg

    win = np.ones(nperseg)
    step = nperseg - noverlap
    num_s = (len(signal_in) - noverlap) // step
    spectrogram = np.zeros((nfft // 2 + 1, num_s), dtype=complex)
    for i in range(num_s):
        start = i * step
        segment = signal_in[start:start + nperseg]

        if len(segment) < nfft:
            segment = np.pad(segment, (0, nfft - len(segment)))

        spectrogram[:, i] = np.fft.rfft(segment, n=nfft)

    frequencies = np.fft.rfftfreq(nfft, 1 / fs)
    times = np.arange(num_s) * step / fs + nperseg / (2 * fs)
    return frequencies, times, np.abs(spectrogram)


def find_peaks_with_threshold(spectrogram, desired_peaks_per_second, times):
    log_spec = 10 * np.log10(np.maximum(spectrogram, 1e-10))
    duration = times[-1] - times[0]
    total_peaks = int(duration * desired_peaks_per_second)
    peak_mask = local_peaks(log_spec, 3)
    peak_pos = np.where(peak_mask)
    peak_values = log_spec[peak_pos]
    sorted_indices = np.argsort(peak_values)[::-1]
    sorted_indices = sorted_indices[:total_peaks]

    peak_f = peak_pos[0][sorted_indices]
    peak_t = peak_pos[1][sorted_indices]

    if len(peak_f) > 0:
        threshold = log_spec[peak_f[-1], peak_t[-1]]
    else:
        threshold = 0

    return peak_f, peak_t, threshold


def build_peak_database(peak_f, peak_t, delta_t_l=3, delta_t_u=6, delta_f=9, max_fan_out=3):
    pairs = []
    peak_f = np.array(peak_f)
    peak_t = np.array(peak_t)

    for i in range(len(peak_t)):
        f1 = peak_f[i]
        t1 = peak_t[i]

        valid_indices = np.where(
            (peak_t > t1 + delta_t_l) &
            (peak_t < t1 + delta_t_u) &
            (np.abs(peak_f - f1) <= delta_f)
        )[0]

        if len(valid_indices) > max_fan_out:
            freq_diff = np.abs(peak_f[valid_indices] - f1)
            closest_indices = np.argsort(freq_diff)[:max_fan_out]
            valid_indices = valid_indices[closest_indices]

        for j in valid_indices:
            f2 = peak_f[j]
            t2 = peak_t[j]
            pairs.append([f1, f2, t1, t2 - t1])

    pairs_array = np.array(pairs) if pairs else np.array([])
    if len(pairs_array) > 0:
        sorted_indices = np.argsort(pairs_array[:, 2])
        pairs_array = pairs_array[sorted_indices]

    return pairs_array


def compute_hash(f1: int, f2: int, dt: int) -> int:
    """h(f1, f2, t2-t1) = (t2-t1)·2¹⁶ + f1·2⁸ + f2"""
    return (dt << 16) + (f1 << 8) + f2


def build_hash_database(pairs: np.ndarray, song_id: str, num_freq_bins: int) -> Dict[int, List[dict]]:
    """Build hash database from peak pairs."""
    database = defaultdict(list)
    freq_scale = 255 / (num_freq_bins - 1)

    for f1, f2, t1, dt in pairs:
        f1_scaled = int(f1 * freq_scale)
        f2_scaled = int(f2 * freq_scale)

        dt_scaled = int(dt * 100)
        hash_value = compute_hash(f1_scaled, f2_scaled, dt_scaled)

        database[hash_value].append({
            'song_id': song_id,
            't1': float(t1),
            'f1': f1_scaled,
            'f2': f2_scaled,
            'dt': dt_scaled
        })

    return database


def save_database(database: Dict, filename: str = 'hash_database.json'):
    """Save hash database to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(database, f)
        return True
    except Exception as e:
        print(f"Error saving database: {str(e)}")
        return False


def load_database(filename: str = 'hash_database.json') -> Dict:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return {}


def identify_song(audio_data: np.ndarray, fs: int, database: Dict, debug: bool = False) -> Tuple[str, float]:
    """Identify song from audio data."""
    frequencies, times, spectrogram = compute_spectrogram(audio_data, fs)
    peak_frequencies, peak_times, _ = find_peaks_with_threshold(
        spectrogram, desired_peaks_per_second=30, times=times
    )

    pairs = build_peak_database(
        peak_frequencies, peak_times,
        delta_t_l=3, delta_t_u=6, delta_f=9, max_fan_out=3
    )

    sample_hashes = build_hash_database(
        pairs, "sample", spectrogram.shape[0]
    )

    matches = defaultdict(list)
    for hash_value, sample_entries in sample_hashes.items():
        if str(hash_value) in database:
            db_entries = database[str(hash_value)]
            for sample_entry in sample_entries:
                sample_time = sample_entry['t1']
                for db_entry in db_entries:
                    song_id = db_entry['song_id']
                    db_time = db_entry['t1']
                    offset = db_time - sample_time
                    matches[song_id].append(offset)

    best_score = 0
    best_song = None

    for song_id, offsets in matches.items():
        offset_counts = Counter(np.round(offsets, decimals=2))
        score = max(offset_counts.values())

        if score > best_score:
            best_score = score
            best_song = song_id

    confidence = best_score / len(pairs) if len(pairs) > 0 else 0
    return best_song, confidence


def process_songs_h5(file_path: str) -> Dict:
    """Process all songs in an HDF5 file and build a database."""
    database = defaultdict(list)
    fs = 8000

    # Open the HDF5 file
    with h5py.File(file_path, "r") as h5_file:
        for song_id in h5_file.keys():
            print(f"Processing song: {song_id}")

            # Retrieve audio data for each song
            audio_data = h5_file[song_id][()]
            frequencies, times, spectrogram = compute_spectrogram(audio_data, fs)
            peak_frequencies, peak_times, _ = find_peaks_with_threshold(
                spectrogram, 30, times
            )
            pairs = build_peak_database(peak_frequencies, peak_times)

            song_hashes = build_hash_database(pairs, song_id, spectrogram.shape[0])
            for k, v in song_hashes.items():
                database[k].extend(v)

    return database

def main():
    parser = argparse.ArgumentParser(description='Shazam-like audio fingerprinting')
    parser.add_argument('--build-db', action='store_true',
                        help='Build hash database from audio files')
    parser.add_argument('--wav-input', type=str,
                        help='Input WAV file to identify')
    parser.add_argument('--record', action='store_true',
                        help='Record audio for identification')
    parser.add_argument('--duration', type=int, default=10,
                        help='Recording duration in seconds')
    parser.add_argument('--output', type=str, default='hash_database.json',
                        help='Output database file')

    args = parser.parse_args()
    fs = 8000

    try:
        if args.build_db:
            print("Building database from songs_test directory...")
            database = process_songs_h5("songs.h5")
            save_database(database, 'hash_database.json')
            print("Database successfully created and saved at: hash_database.json")
            return 0

        elif args.record:
            try:
                database = load_database('hash_database.json')
            except FileNotFoundError:
                print("Error: Database not found. Please run --build-db first.")
                return 1

            audio_data = record_audio(args.duration, fs)
            song_id, confidence = identify_song(audio_data, fs, database)

            if song_id:
                print(f"Your song is: {song_id} (confidence: {confidence:.2f})")
                return 0
            else:
                print("No match found")
                return 1

        elif args.wav_input:
            try:
                database = load_database('hash_database.json')
            except FileNotFoundError:
                print("Error: Database not found. Please run --build-db first.")
                return 1

            audio_data = wav_processing(args.wav_input, fs)
            song_id, confidence = identify_song(audio_data, fs, database)

            if song_id:
                print(f"Your song is: {song_id} (confidence: {confidence:.2f})")
                return 0
            else:
                print("No match found")
                return 1

        else:
            parser.print_help()
            return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    main()