import pretty_midi

Fs = 16000


def get_audio(filename, seconds=None):
    """
    Convert a MIDI file to audio using the FluidSynth synthesiser (by default is a piano sound).
    Params:
    - str filename: the MIDI file name
    - int seconds: the number of seconds of audio to output (default = None, in which case the whole audio is returned)
    Return:
    - waveform: the corresponding audio data
    """
    # Load the file as a pretty midi object
    pm = pretty_midi.PrettyMIDI(filename)

    # Get the audio data
    waveform = pm.fluidsynth(fs=Fs)

    if seconds == None:
        return waveform
    else:
        return waveform[:seconds*Fs]
