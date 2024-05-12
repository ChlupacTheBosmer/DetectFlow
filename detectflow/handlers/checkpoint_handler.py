import json
import os
import logging

class CheckpointHandler:
    """
    Base class for handling persistent checkpointing of data processing tasks.

    This class provides methods to manage checkpoint data, which facilitates resuming
    long-running tasks after interruption. It uses a JSON file to store and retrieve
    checkpoint information.

    Attributes:
        checkpoint_file (str): Path to the JSON file used for storing checkpoint data.

    Methods:
        _load_checkpoint(): Load checkpoint data from the JSON file.
        _save_checkpoint(): Save the current checkpoint data to the JSON file.
        update_checkpoint(**kwargs): Update checkpoint data with provided keyword arguments.
        get_checkpoint_data(key, default=None): Retrieve a specific piece of data from the checkpoint.
        remove_checkpoint(): Delete the checkpoint file and clear in-memory data.
        process_data(): Abstract method for processing data, to be implemented by subclasses.
    """
    def __init__(self, checkpoint_file='checkpoint.json'):
        """
        Initialize the CheckpointHandler with a specific checkpoint file.

        Args:
            checkpoint_file (str): Path to the JSON file to use for checkpointing. Default is 'checkpoint.json'.
        """
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self._load_checkpoint()

    def _load_checkpoint(self):
        """
        Load checkpoint data from a JSON file. If the file does not exist or is corrupt, returns an empty dictionary.
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as file:
                    return json.load(file)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {self.checkpoint_file}: {e}")
                return {}
        return {}

    def _save_checkpoint(self):
        """
        Save the current checkpoint data to the JSON file.
        This method ensures that all changes to checkpoint data are persisted.
        """
        with open(self.checkpoint_file, 'w') as file:
            json.dump(self.checkpoint_data, file, indent=4)

    def update_checkpoint(self, **kwargs):
        """
        Update the checkpoint data with specified key-value pairs and immediately persist to disk.

        Args:
            **kwargs: Arbitrary key-value pairs to be added or updated in the checkpoint data.
        """
        self.checkpoint_data.update(kwargs)
        self._save_checkpoint()

    def get_checkpoint_data(self, key, default=None):
        """
        Retrieve a piece of data from the checkpoint using a key.

        Args:
            key (str): The key for the data to retrieve.
            default (optional): The default value to return if the key is not found. Default is None.

        Returns:
            The value from the checkpoint data corresponding to the provided key, or the default value if the key is not found.
        """
        return self.checkpoint_data.get(key, default)

    def remove_checkpoint(self):
        """
        Remove the checkpoint file from the disk and clear any in-memory data. This is typically done when the process is complete and the checkpoint is no longer needed.
        """
        try:
            os.remove(self.checkpoint_file)
            self.checkpoint_data = {}
        except OSError as e:
            logging.error(f"Failed to remove checkpoint file: {e}")
