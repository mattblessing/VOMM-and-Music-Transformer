"""
Functions for quantitative evaluation of the system.
"""
from vocabulary import note_on_events
from tokenisation import events_parser
import numpy as np
from scipy.stats import ttest_ind


def pitch_count(events):
    """
    Count the number of different pitches in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - int count: the number of different pitches
    """
    events = np.array(events)

    # Get all pitches in the MIDI data
    pitches = events[(0 <= events) & (events < note_on_events)]

    # Count the number of unique pitches
    count = len(np.unique(pitches))

    return count


def pitch_class_histogram(events):
    """
    Count the number of occurrences of each pitch class in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - np.array hist: the number of times each pitch class occurs
    """
    events = np.array(events)

    # Get the pitch class of each note in the MIDI data
    pitch_classes = events[(0 <= events) & (events < note_on_events)] % 12

    # Count how often each pitch class occurs
    classes, counts = np.unique(pitch_classes, return_counts=True)

    # Create histogram
    hist = np.zeros(12)
    num_classes = len(classes)
    for i in range(num_classes):
        hist[classes[i]] = counts[i]

    # C4 = middle C = 60 ---> pitch class 0 corresponds to C

    return hist


def pitch_class_transition_matrix(events):
    """
    Count how often each possible pitch class transition occurs in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - np.array transition_matrix: the pitch class transition matrix
    """
    events = np.array(events)

    # Get the pitch class of each note in the MIDI data
    pitch_classes = events[(0 <= events) & (events < note_on_events)] % 12

    # Initialise transition matrix
    transition_matrix = np.zeros((12, 12))

    if len(pitch_classes) != 0:
        # Iterate through the sequence of pitch classes and update the transition matrix
        prev_class = pitch_classes[0]
        for pitch_class in pitch_classes:
            transition_matrix[prev_class][pitch_class] += 1
            prev_class = pitch_class

    return transition_matrix


def pitch_range(events):
    """
    Get the difference between the highest and lowest pitches in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - int pitch_range: the difference between the highest and lowest pitch
    """
    events = np.array(events)

    # Get all pitches in the MIDI data
    pitches = events[(0 <= events) & (events < note_on_events)]

    if len(pitches) != 0:
        # Get the difference between the highest and lowest pitch
        pitch_range = np.max(pitches) - np.min(pitches)
    else:
        pitch_range = np.nan

    return pitch_range


def avg_pitch_interval(events):
    """
    Get the average interval between two consecutive pitches in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - float avg_pi: the average pitch interval
    """
    events = np.array(events)

    # Get all pitches in the MIDI data
    pitches = events[(0 <= events) & (events < note_on_events)]

    # Iterate through the sequence of pitches and calculate the average pitch interval
    if len(pitches) != 0:
        avg_pi = 0
        prev_pitch = pitches[0]
        for pitch in pitches:
            avg_pi += pitch - prev_pitch
            prev_pitch = pitch
        avg_pi /= len(pitches)
    else:
        avg_pi = np.nan

    return avg_pi


def note_count(events):
    """
    Get the number of notes in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - int count: the number of notes in the MIDI data
    """
    events = np.array(events)

    # Get all pitches in the MIDI data
    pitches = events[(0 <= events) & (events < note_on_events)]

    count = len(pitches)

    return count


def avg_inter_onset_interval(events):
    """
    Get the average time interval between the onsets of consecutive notes in a MIDI recording.
    Params:
    - list[int] events: a list of indices of events in the vocabulary
    Return:
    - float avg_ioi: the average inter-onset interval
    """
    # Convert to a MIDI file track
    midi_track = events_parser(events).tracks[1]

    # Initialisation
    intervals = []      # List of onset intervals
    delta_time = 0      # Time between note onsets
    first_onset = True  # Indicator as to whether the next onset is the first onset in the data

    for msg in midi_track:
        # Update delta time
        delta_time += msg.time
        # Get message type
        msg_type = msg.type

        # If note onset
        if msg_type == "note_on":
            if first_onset:
                # If the first note onset, don't add the delta time to the list of intervals
                first_onset = False
            else:
                # Otherwise, add the delta time to the list of intervals
                intervals.append(delta_time)
            # Reset delta time
            delta_time = 0

    if len(intervals) != 0:
        avg_ioi = np.nanmean(intervals)
    else:
        avg_ioi = np.nan

    return avg_ioi


def features(samples):
    """
    Compute the above features for a set of samples.
    Params:
    - list[list[int]] samples: the set of samples
    Return:
    - dict features: the features of each sample
    """
    pitch_counts = []
    pitch_class_histograms = []
    pitch_class_transition_matrices = []
    pitch_ranges = []
    avg_pitch_intervals = []
    note_counts = []
    avg_inter_onset_intervals = []

    for sample in samples:
        pitch_counts.append(pitch_count(sample))
        pitch_class_histograms.append(pitch_class_histogram(sample))
        pitch_class_transition_matrices.append(pitch_class_transition_matrix(sample))
        pitch_ranges.append(pitch_range(sample))
        avg_pitch_intervals.append(avg_pitch_interval(sample))
        note_counts.append(note_count(sample))
        avg_inter_onset_intervals.append(avg_inter_onset_interval(sample))

    features = {"pitch_counts": pitch_counts,
                "pitch_class_histograms": pitch_class_histograms,
                "pitch_class_transition_matrices": pitch_class_transition_matrices,
                "pitch_ranges": pitch_ranges,
                "avg_pitch_intervals": avg_pitch_intervals,
                "note_counts": note_counts,
                "avg_inter_onset_intervals": avg_inter_onset_intervals}

    return features


def mean_std_features(samples):
    """
    Compute the (element-wise) mean and standard deviation of each feature for a set of samples.
    Params:
    - list[list[int]] samples: the set of samples
    Return:
    - dict mean_std_features: the (element-wise) mean and standard deviation of each feature
    """
    sample_features = features(samples)
    pitch_counts = sample_features["pitch_counts"]
    pitch_class_histograms = sample_features["pitch_class_histograms"]
    pitch_class_transition_matrices = sample_features["pitch_class_transition_matrices"]
    pitch_ranges = sample_features["pitch_ranges"]
    avg_pitch_intervals = sample_features["avg_pitch_intervals"]
    note_counts = sample_features["note_counts"]
    avg_inter_onset_intervals = sample_features["avg_inter_onset_intervals"]

    # Mean and standard deviation for pitch count
    mean_pitch_count = np.nanmean(pitch_counts)
    std_dev_pitch_count = np.nanstd(pitch_counts)

    # Mean and standard deviation for pitch class histogram
    pitch_class_histograms = np.vstack(pitch_class_histograms)
    mean_pitch_class_histogram = np.nanmean(pitch_class_histograms, axis=0)
    std_dev_pitch_class_histogram = np.nanstd(pitch_class_histograms, axis=0)

    # Mean and standard deviation for pitch class transition matrix
    pitch_class_transition_matrices = np.stack(pitch_class_transition_matrices, axis=-1)
    mean_pitch_class_transition_matrix = np.nanmean(pitch_class_transition_matrices, axis=2)
    std_dev_pitch_class_transition_matrix = np.nanstd(pitch_class_transition_matrices, axis=2)

    # Mean and standard deviation for pitch range
    mean_pitch_range = np.nanmean(pitch_ranges)
    std_dev_pitch_range = np.nanstd(pitch_ranges)

    # Mean and standard deviation for avg pitch interval
    mean_avg_pitch_interval = np.nanmean(avg_pitch_intervals)
    std_dev_avg_pitch_interval = np.nanstd(avg_pitch_intervals)

    # Mean and standard deviation for note count
    mean_note_count = np.nanmean(note_counts)
    std_dev_note_count = np.nanstd(note_counts)

    # Mean and standard deviation for avg inter-onset interval
    mean_avg_inter_onset_interval = np.nanmean(avg_inter_onset_intervals)
    std_dev_avg_inter_onset_interval = np.nanstd(avg_inter_onset_intervals)

    mean_std_features = {"mean_pitch_count": mean_pitch_count,
                         "std_dev_pitch_count": std_dev_pitch_count,
                         "mean_pitch_class_histogram": mean_pitch_class_histogram,
                         "std_dev_pitch_class_histogram": std_dev_pitch_class_histogram,
                         "mean_pitch_class_transition_matrix": mean_pitch_class_transition_matrix,
                         "std_dev_pitch_class_transition_matrix": std_dev_pitch_class_transition_matrix,
                         "mean_pitch_range": mean_pitch_range,
                         "std_dev_pitch_range": std_dev_pitch_range,
                         "mean_avg_pitch_interval": mean_avg_pitch_interval,
                         "std_dev_avg_pitch_interval": std_dev_avg_pitch_interval,
                         "mean_note_count": mean_note_count,
                         "std_dev_note_count": std_dev_note_count,
                         "mean_avg_inter_onset_interval": mean_avg_inter_onset_interval,
                         "std_dev_avg_inter_onset_interval": std_dev_avg_inter_onset_interval}

    return mean_std_features


def t_test(sample1, sample2, sig_level=0.05):
    """
    Perform an unequal variance t-test (Welch's t-test) for two samples.
    Params:
    - list[float] sample1: a sample from one variable
    - list[float] sample2: a sample from the other variable
    - float sig_level: the significance level to use for the tests
    Return:
    - float p: the p-value
    - bool rej: whether result is statistically significant
    """
    # Get p-value
    p = ttest_ind(a=sample1, b=sample2, equal_var=False, nan_policy="omit").pvalue

    # Determine whether null hypothesis should be rejected or not - indicates which results are significant
    rej = False
    if p <= sig_level:
        rej = True

    return p, rej
