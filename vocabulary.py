"""
The "MIDI-Like" Vocabulary from Oore et al. (2018) which is used to tokenise the MIDI files.
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""

# Note on and note off events involve notes being switched on/off
note_on_events = 128  #  128 possible notes
note_off_events = note_on_events
note_events = note_on_events + note_off_events

# Time shift events move the time step forward
time_shift_events = 100  # 100 time shift incrementst
max_time_shift = 1000  # Max time shift length in ms (1s)
time_shift_ms = max_time_shift // time_shift_events  # Each time shift is 10ms

# Velocity events change the velocity applied to all subsequent notes
velocity_events = 32
max_velocities = 128  # Velocity goes from 0-127
# Number of velocities in each velocity bin
velocity_bin_step = max_velocities // velocity_events

# Total number of possible events
total_midi_events = note_on_events + note_off_events + time_shift_events + velocity_events

# Create vocabulary
note_on_vocab = [f"note_on_{i}" for i in range(note_on_events)]
note_off_vocab = [f"note_off_{i}" for i in range(note_off_events)]
time_shift_vocab = [f"time_shift_{i}" for i in range(time_shift_events)]
velocity_vocab = [f"set_velocity_{i}" for i in range(velocity_events)]
vocab = ["<pad>"] + note_on_vocab + note_off_vocab + time_shift_vocab + velocity_vocab + ["<start>", "<end>"]
vocab_size = len(vocab)

# Indices for useful tokens
pad_token_index = vocab.index("<pad>")
start_token_index = vocab.index("<start>")
end_token_index = vocab.index("<end>")

"""
Helper functions related to the vocabulary.
"""


def events_to_indices(event_list):
    """
    Convert a list of events to a list of indices in the vocabulary.
    """
    index_list = []

    for event in event_list:
        index_list.append(vocab.index(event))

    return index_list


def indices_to_events(index_list):
    """
    Convert a list of vocabulary indices to a list of events.
    """
    event_list = []

    for i in index_list:
        event_list.append(vocab[i])

    return event_list


def velocity_to_bin(velocity):
    """
    Get the bin that a given velocity falls in.
    """
    if not (0 <= velocity <= 127):
        raise ValueError(f"Velocities must be between 0 and 127, not {velocity}.")

    bin = velocity // velocity_bin_step

    return bin


def bin_to_velocity(bin):
    """
    Get the velocity corresponding to a particular bin.
    """
    velocity = int(bin * velocity_bin_step)

    if not (0 <= velocity <= 127):
        raise ValueError(f"Bin must be between 0 and {velocity_events}, not {bin}.")

    return velocity


def my_round(a):
    """
    Custom rounding function to round *.5 up rather than down.
    """
    b = int(a)
    decimal = a % 1
    if decimal >= 0.5:
        b += 1
    return b


def time_cutter(time):
    """
    Convert the time between MIDI events into a sequence of finite-length time shifts.
    This sequence will be expressed as k instances of the maximum time shift followed by a leftover time shift:
    time = k * max_time_shift + leftover_time_shift
    """
    time_shifts = []

    for _ in range(time // max_time_shift):
        time_shifts.append(time_shift_events)

    leftover_time_shift = my_round((time % max_time_shift) / time_shift_ms)
    if leftover_time_shift > 0:
        time_shifts.append(leftover_time_shift)

    return time_shifts


def delta_time_to_time_shifts(delta_time, index_list):
    """
    Convert time between MIDI events into a sequence of time shift events.
    Update the input event list and list of vocabulary indices.
    """
    time = time_cutter(delta_time)

    for i in time:
        # Add time shift event to the input lists
        vocab_index = note_events + i
        index_list.append(vocab_index)
