"""
The functions used for preprocessing the tokenised data.
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""
from vocabulary import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
seed = 42
torch.manual_seed(seed)
random.seed(seed)


def crop_sequences(seqs, length, factor=6):
    """
    Randomly crop sequences of roughly `length` events from a set of sequences `seqs`.
    Params:
    - list[list[int]] seqs: list of sequences
    - int length: approximate length to cut sequences to
    - int factor: factor to vary the range of output lengths (a higher factor narrows the length range)
    Return:
    - list[list[int]] crops: list of cropped sequences
    """
    crops = []

    for seq in seqs:
        seq_length = len(seq)

        # Get length of crop
        crop_length = min(seq_length, random.randint(length - (length // factor), length + (length // factor)))

        # Get starting index of crop
        if crop_length == seq_length:
            index = 0
        else:
            index = random.randint(0, seq_length - crop_length)

        crops.append(seq[index: (index + crop_length)])

    return crops


def crop_endings(seqs, length, factor=6):
    """
    Randomly crop endings of roughly `length` events from a set of sequences `seqs`.
    Params:
    - list[list[int]] seqs: list of sequences
    - int length: approximate length for the endings
    - int factor: factor to vary range of output lengths (a higher factor narrows the length range)
    Return:
    - list[list[int]] endings: list of cropped endings
    """
    endings = []

    for seq in seqs:
        # Get lower bound for where ending should begin
        lower_bound = max(0, len(seq) - length)

        # Get starting index of ending
        index = random.randint(lower_bound, lower_bound + length // factor)

        endings.append(seq[index:])

    return endings


def augment(data, pitch_shifts=list(range(-2, 3)), time_stretches=[1, 1.05, 1.1]):
    """
    Augment the data with pitch shifts and time stretching, add start and end tokens to each sequence, 
    and pad each sequence to the max sequence length in the data.
    Params:
    - list[list[int]] data: list of sequences
    - list[int] pitch_shifts: list of pitch shifts in semitones
    - list[float] time_stretches: list of time stretches (must be positive)
    Return:
    - torch.Tensor augmented_data: the full dataset after augmentation and padding
    """
    # Add 1 to the time stretches if not there already
    if 1 not in time_stretches:
        time_stretches.append(1)

    # Add the inverse time stretches to the list if not there already
    ts = []
    for t in time_stretches:
        if t not in ts:
            ts.append(t)
        if t != 1 and 1 / t not in ts:
            ts.append(1 / t)
    ts.sort()
    time_stretches = ts

    # Perform pitch shifts

    pitch_shifted_data = []
    for seq in data:
        for shift in pitch_shifts:
            pitch_shifted_seq = []
            out_of_bounds = False
            for i in seq:
                # Get the shifted index
                shifted_i = i + shift

                # If note on event
                if 0 < i <= note_on_events:
                    # If shifted event is a note on event
                    if 0 < shifted_i <= note_on_events:
                        # Add shifted index to sequence
                        pitch_shifted_seq.append(shifted_i)
                    # If shifted event is now a note off event
                    else:
                        # Don't include the shifted sequence
                        out_of_bounds = True
                        break

                # If note off event
                elif note_on_events < i <= note_events:
                    # If shifted event is a note off event
                    if note_on_events < shifted_i <= note_events:
                        # Add shifted index to sequence
                        pitch_shifted_seq.append(shifted_i)
                    else:
                        # Don't include the shifted sequence
                        out_of_bounds = True
                        break

                # If any other event
                else:
                    # Add the original index
                    pitch_shifted_seq.append(i)

            # If no notes were shifted out of bounds then add shifted sequence
            if not out_of_bounds:
                #  Convert to tensor
                pitch_shifted_seq = torch.LongTensor(pitch_shifted_seq)
                pitch_shifted_data.append(pitch_shifted_seq)

    # Perform time stretches

    time_stretched_data = []
    delta_time = 0
    for seq in pitch_shifted_data:
        for time_stretch in time_stretches:
            time_stretched_seq = []
            for i in seq:
                # If a time shift event
                if note_events < i <= note_events + time_shift_events:
                    # Get time shift value
                    time = i - (note_events - 1)
                    # Scale this according to the time stretching parameter, then add to delta time
                    delta_time += my_round(time * time_shift_ms * time_stretch)

                # If any other event
                else:
                    # Add new time shift events for the new delta time to the sequence
                    delta_time_to_time_shifts(delta_time, time_stretched_seq)
                    # Add the event to the sequence
                    time_stretched_seq.append(i)
                    # Reset delta time
                    delta_time = 0

            # Add time stretched sequence
            time_stretched_seq = torch.LongTensor(time_stretched_seq)  # Convert to tensor
            time_stretched_data.append(time_stretched_seq)

    # Add start and end tokens to each sequence
    augmented_data = []
    for seq in time_stretched_data:
        augmented_data.append(F.pad(F.pad(seq, (1, 0), value=start_token_index), (0, 1), value=end_token_index))

    # Pad all sequences to the max sequence length
    augmented_data = nn.utils.rnn.pad_sequence(augmented_data, padding_value=pad_token_index).transpose(-1, -2)

    return augmented_data
