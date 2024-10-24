"""
Functions to generate MIDI using the Music Transformer.
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""
from vocabulary import start_token_index, end_token_index
from tokenisation import events_to_indices, events_parser
from music_transformer import *


def load_model(filepath):
    """
    Load a saved music transformer model for generation
    Params:
    - str filepath: the path to a .pt file containing a dictionary including the key : value pairs
                    {"model_state_dict": MusicTransformer.state_dict(),
                     "hparams": hparams}
    Return:
    - MusicTransformer model: the model to generate from
    """
    file = torch.load(filepath, map_location=device)

    model = MusicTransformer(**file["hparams"]).to(device)
    model.load_state_dict(file["model_state_dict"])
    model.eval()

    return model


def decode(model, input, mode="categorical", temperature=1.0, k=None):
    """
    Generate a sequence from a model until it predicts an end token.
    Params:
    - model: the model to generate from
    - list[int] input: list of MIDI event indices for the model to continue from
    - str mode: mode to choose next event given model output 
    ("argmax": choose event with max logit, "categorical": sample next event with logits as probability distribution)
    - float temperature: softmax temperature to make the model outputs more diverse (high temp) or less diverse (low temp) (~ 1.0)
    - int k: if not None, will choose next event from the top k model output logits
    Return:
    - list[int] generation: list of generated MIDI event indices
    """
    # Make sure generation starts with the start token
    if input[0] != start_token_index:
        input = [start_token_index] + input

    # Convert to torch tensor and convert to correct dimensions for masking
    generation = torch.tensor(input, dtype=torch.int64, device=device).unsqueeze(0)
    n = generation.dim() + 2

    try:
        with torch.no_grad():
            # Autoregressively generate output
            while True:
                # Get logits for next predicted index
                predictions = model(generation, mask=create_mask(generation, n))
                # Divide logits by temperature
                predictions /= temperature

                if mode == "argmax":
                    # Next event = the one with max logit
                    prediction = torch.argmax(predictions[..., -1, :], dim=-1)
                elif k is not None:
                    # Get top k predictions
                    top_k_preds = torch.topk(predictions[..., -1, :], k, dim=-1)
                    # Choose next event from top k predictions
                    predicted_idx = torch.distributions.Categorical(logits=top_k_preds.values[..., -1, :]).sample()
                    prediction = top_k_preds.indices[..., predicted_idx]
                elif mode == "categorical":
                    # Sample next event using logits as probs
                    prediction = torch.distributions.Categorical(logits=predictions[..., -1, :]).sample()
                else:
                    raise ValueError("Invalid mode or k.")

                # If end token, end generation
                if prediction == end_token_index:
                    return generation.squeeze()

                # Otherwise, add event to output and move to the next prediction
                generation = torch.cat([generation, prediction.view(1, 1)], dim=-1)

    # KeyboardInterrupt occurs if generation takes a very long time and user wants to cut it short
    # RuntimeError occurs if more tokens are generated than there are absolute positional encodings for
    except (KeyboardInterrupt, RuntimeError):
        pass

    return generation.squeeze()


def generate(model, input=None, mode="categorical", temperature=1.0, k=None):
    """
    Generate a MIDI file and save at `save_path`.
    Params:
    - model: MusicTransformer model to generate audio with
    - list[int] input (optional): list of tokens for the model to continue from
    - str mode (default = "categorical"): mode to choose next event given model output 
    ("argmax": choose event with max logit, "categorical": sample next event with logits as probability distribution)
    - float temperature (default 1.0): softmax temperature to make the model outputs more diverse (high temperature) or less diverse (low temperature) (~ 1.0)
    - int k (optional): if not None, will choose next event from the top k model output logits
    Return:
    - list[int] sequence: the generated sequence
    """
    if input is None:
        input = [start_token_index]

    # Perform decoding
    token_ids = decode(model, input, mode, temperature, k)
    sequence = token_ids.tolist()

    return sequence
