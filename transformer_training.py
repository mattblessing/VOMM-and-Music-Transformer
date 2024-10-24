"""
Trainer class and functions for the Music Transformer.
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""
from music_transformer import *
import time
import os


def loss_with_mask(prediction, target, criterion=F.cross_entropy):
    """
    Calculate loss while masking padded values.
    Params:
    - torch.Tensor prediction: output of the model
    - torch.Tensor target: true value that the model was supposed to predict
    - criterion: loss criterion
    Return:
    - float loss_masked: the loss
    """
    # Get loss without masking
    loss = criterion(prediction, target, reduction="none")

    # Get mask with ones where the target is not 0
    mask = torch.ne(target, torch.zeros_like(target))

    # Multiply mask and loss element-wise
    mask = mask.to(loss.dtype)
    loss *= mask

    # Output is averaged over the number of values that were not masked
    loss_masked = torch.sum(loss) / torch.sum(mask)

    return loss_masked


def train_step(model, optimiser, input, target):
    """
    Compute loss and backward pass for a single training step of the model.
    Params:
    - model: the transformer model to train
    - optimiser: optimiser initialized with model's parameters
    - torch.Tensor input: input batch
    - torch.Tensor target: input batch shifted right by 1 position
    Return:
    - float loss: the loss
    """
    # Forward pass
    pred = model(input, mask=create_mask(input, n=input.dim() + 2))

    # Backward pass
    optimiser.zero_grad()
    loss = loss_with_mask(pred.transpose(-1, -2), target)
    loss.backward()
    optimiser.step()

    return float(loss)


def val_step(model, input, target):
    """
    Compute loss for a single evaluation of the model.
    Params:
    - model: MusicTransformer model to evaluate
    - torch.Tensor input: input batch
    - torch.Tensor target: input batch shifted right by 1 position
    Return:
    - float loss: the loss
    """
    # Forward pass
    pred = model(input, mask=create_mask(input, n=max(input.dim() + 2, 2))).detach()
    loss = loss_with_mask(pred.transpose(-1, -2), target)
    return float(loss)


class MusicTransformerTrainer:
    """
    A class to train the music transformer with loading and saving functionality.
    """

    def __init__(self, hparams, train_data, val_data, batch_size, checkpoint_path="music_transformer.pt", load_from_checkpoint=False):
        """
        Params:
        - dict hparams: the model hyperparameters
        - torch.Tensor train_data: the training data
        - torch.Tensor val_data: the validation data
        - int batch_size: batch size
        - str checkpoint_path: path at which to save checkpoints while training
        - bool load_from_checkpoint (optional): indicator for whether to load the model from a checkpoint or not
        """
        train_data = train_data.to(device)
        val_data = val_data.to(device)

        self.batch_size = batch_size

        # The max absolute position must be able to account for the largest sequence in the data
        if hparams["max_abs_position"] > 0:
            hparams["max_abs_position"] = max(hparams["max_abs_position"], train_data.shape[-1])

        # Datasets and dataloaders - split the data into the first (n-1) tokens and the last (n-1) tokens
        self.train_ds = torch.utils.data.TensorDataset(train_data[:, :-1], train_data[:, 1:])
        self.train_dl = torch.utils.data.DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True)

        self.val_ds = torch.utils.data.TensorDataset(val_data[:, :-1], val_data[:, 1:])
        self.val_dl = torch.utils.data.DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=True)

        # Create model
        self.model = MusicTransformer(**hparams).to(device)
        self.hparams = hparams

        # Set up optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98))

        # Set up checkpointing
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []

        # Load checkpoint if necessesary
        if load_from_checkpoint and os.path.isfile(self.checkpoint_path):
            self.load()

    def load(self, checkpoint_path=None):
        """
        Load a checkpoint.
        Params:
        - str checkpoint_path (optional): if None, loads checkpoint from self.checkpoint_path, 
        otherwise loads from new path and updates self.checkpoint_path
        """
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path

        checkpoint = torch.load(self.checkpoint_path)

        del self.model, self.optimiser

        # Create and load model
        self.model = MusicTransformer(**checkpoint["hparams"]).to(device)
        self.hparams = checkpoint["hparams"]
        print("Loading the model...", end="")
        print(self.model.load_state_dict(checkpoint["model_state_dict"]))

        # Create and load optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

        # Load losses
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["validation_losses"]

    def save(self, checkpoint_path=None):
        """
        Save a checkpoint.
        Params:
        - str checkpoint_path (optional): if None, saves checkpoint at self.checkpoint_path, 
        otherwise saves at new path and updates self.checkpoint_path
        """
        if checkpoint_path is not None:
            self.ckpt_path = checkpoint_path

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "hparams": self.hparams
        }

        torch.save(checkpoint, self.checkpoint_path)

    def fit(self, epochs):
        """
        Training loop that will save every epoch and can be terminated early with a KeyboardInterrupt.
        Params:
        - int epochs: number of epochs to train for
        """
        train_losses = []
        val_losses = []
        start = time.time()

        print("Beginning training...")
        print(time.strftime("%Y-%m-%d %H:%M"))

        try:
            for epoch in range(epochs):
                train_epoch_losses = []
                val_epoch_losses = []

                self.model.train()
                for train_input, train_target in self.train_dl:
                    loss = train_step(self.model, self.optimiser, train_input, train_target)
                    train_epoch_losses.append(loss)

                self.model.eval()
                for val_input, val_target in self.val_dl:
                    loss = val_step(self.model, val_input, val_target)
                    val_epoch_losses.append(loss)

                # Get mean losses for the epoch
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)

                # Store complete history of losses
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)

                print(f"Epoch {epoch} Time taken {round(time.time() - start, 2)} seconds "
                      f"Train Loss {train_losses[-1]} Val Loss {val_losses[-1]}")

                self.save()

                start = time.time()

                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            pass

        print("Done")
        print(time.strftime("%Y-%m-%d %H:%M"))
