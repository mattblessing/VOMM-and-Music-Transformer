"""
The functions used for tokenisation and detokenisation.
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""
from vocabulary import *
import mido


def force_note_off(track):
    """
    Convert note on messages with 0 velocity to note off messages.
    """
    # Initialise new MIDI track
    new_track = mido.MidiTrack()

    for msg in track:
        # Replace note on message with note off if velocity = 0
        if msg.type == "note_on" and msg.velocity == 0:
            new_msg = mido.Message(
                "note_off", channel=msg.channel, note=msg.note, velocity=0, time=msg.time)
            new_track.append(new_msg)
        else:
            new_track.append(msg)

    return new_track


def midi_parser(filename):
    """
    Tokenise a MIDI file into a sequence of events using the MIDI-Like vocabulary.
    """
    #  Load MIDI file
    midi_file = mido.MidiFile(filename)

    # Initialisation
    delta_time = 0      # Time between MIDI events
    index_list = []     # List of vocabulary indices for the MIDI events
    pedal_events = {}   # Dictionary to handle pedal events
    pedal_flag = False  # Flag for pedal events

    # Convert note on messages with 0 velocity to note off messages
    for i in range(len(midi_file.tracks)):
        midi_file.tracks[i] = force_note_off(midi_file.tracks[i])

    for track in midi_file.tracks:
        for msg in track:
            # Update delta time
            delta_time += msg.time
            # Initialise note velocity
            vel = 0
            # Get message type
            msg_type = msg.type

            # If key pressed
            if msg_type == "note_on":
                # Get event index
                vocab_index = 1 + msg.note
                # Get velocity bin
                vel = velocity_to_bin(msg.velocity)
            # If key released
            elif msg_type == "note_off":
                note = msg.note
                # If pedal is down
                if pedal_flag:
                    #  Add to pedal events
                    if note not in pedal_events:
                        pedal_events[note] = 1
                    else:
                        pedal_events[note] += 1
                    continue
                # If pedal not down
                else:
                    # Get event index
                    vocab_index = 1 + note_on_events + note
            # If pedal up/down event
            elif msg_type == "control_change" and msg.control == 64:
                # If pedal down event
                if msg.value >= 64:
                    pedal_flag = True
                # If pedal up event
                elif pedal_events:
                    pedal_flag = False

                    # Add time shift events to output lists
                    delta_time_to_time_shifts(delta_time, index_list)
                    # Reset delta time
                    delta_time = 0

                    # Now that pedal has been lifted, perform the note off events that occurred whilst the pedal was down
                    for note in pedal_events:
                        # Get event index
                        vocab_index = 1 + note_on_events + note
                        # Add note off events to output list
                        for i in range(pedal_events[note]):
                            index_list.append(vocab_index)

                    # Reset pedal events
                    pedal_events = {}
                continue
            else:
                continue

            # Add time shift events to output list
            delta_time_to_time_shifts(delta_time, index_list)
            # Reset delta time
            delta_time = 0

            # If key pressed, add velocity change event to output list
            if msg_type == "note_on":
                index_list.append(1 + note_events + time_shift_events + vel)

            # Add event to output list
            index_list.append(vocab_index)

    return index_list


def events_parser(index_list):
    """
    Translate a sequence of events into a MIDI file.
    """
    # Initialise MIDI file
    midi_file = mido.MidiFile()
    meta_track = mido.MidiTrack()
    track = mido.MidiTrack()

    # Set default messages at start of file
    meta_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=1))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("control_change", control=64, value=0, time=0))

    # Initialise delta time and velocity
    delta_time = 0
    vel = 0

    for i in index_list:
        # If not pad token
        if i > 0:
            # Adjust index to ignore pad token
            i -= 1

            if 0 <= i < note_events:
                # If note on event
                if 0 <= i < note_on_events:
                    note = i
                    msg_type = "note_on"
                    v = vel
                # If note off event
                else:
                    note = i - note_on_events
                    msg_type = "note_off"
                    v = 0
                # Add note message to track
                track.append(mido.Message(msg_type, note=note, velocity=v, time=delta_time))

                # Reset delta time and velocity
                delta_time = 0
                vel = 0

            # If a time shift event
            elif note_events <= i < note_events + time_shift_events:
                #  Add shift time in ms to delta time
                delta_time += (i - note_events + 1) * time_shift_ms

            # If a velocity change event
            elif note_events + time_shift_events <= i < total_midi_events:
                vel = bin_to_velocity(i - (note_events + time_shift_events))

    # End the track
    track.append(mido.MetaMessage("end_of_track", time=1))
    midi_file.tracks.append(meta_track)
    midi_file.tracks.append(track)

    return midi_file
