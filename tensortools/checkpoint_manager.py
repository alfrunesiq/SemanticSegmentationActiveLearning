import glob
import os

from tensorflow.compat import v1 as tf
from tensorflow.python.training import checkpoint_management
if tf.__version__ < "1.14.0":
    from tensorflow.python.training.checkpointable import data_structures
else:
    from tensorflow.python.training.tracking import data_structures

class CheckpointManager(object):

    def __init__(self, checkpoint, directory, max_to_keep=10):

        self._checkpoint   = checkpoint
        self._checkpoints  = []
        self._directory    = directory
        self._max_to_keep  = max_to_keep
        self._CACHE_PREFIX = os.path.join(self._directory, "tmp")
        self._cached_checkpoint = None

    def cache(self, session):
        """
        Cache a checkpoint for temporary storage.
        """
        if self._cached_checkpoint != None:
            # Remove uncommitted checkpoint in cache
            for filename in self._get_checkpoint_filenames(
                    self._cached_checkpoint):
                os.remove(filename)

        self._cached_checkpoint = self._checkpoint.write(self._CACHE_PREFIX,
                                                         session)

    def commit(self, prefix, session):
        """
        Commit the latest checkpoint.
        """

        if self._cached_checkpoint == None:
            if len(self._checkpoints) > 0:
                return self._checkpoints[-1]
            else:
                return ""

        if len(self._checkpoints) == self._max_to_keep:
            for filename in self._get_checkpoint_filenames(
                    self._checkpoints.pop(0)):
                os.remove(filename)

        # Replication from checkpoint.save
        if self._checkpoint._save_counter is None:
            session.run(self._checkpoint.save_counter.initializer)
        if self._checkpoint._save_assign_op is None:
            self._checkpoint._save_assign_op = data_structures.NoDependency(
                self._checkpoint.save_counter.assign_add(1, read_value=True))

        checkpoint_count = session.run(self._checkpoint._save_assign_op)
        filename_prefix = "%s-%d" % (prefix, checkpoint_count)

        for filename in self._get_checkpoint_filenames(
                self._cached_checkpoint):
            # Change prefix
            os.rename(filename,
                      filename.replace(self._cached_checkpoint,
                                       filename_prefix))

        self._checkpoints.append(filename_prefix)
        self._cached_checkpoint = None
        # Update checkpoint state file (@tf.train.latest_checkpoint)
        checkpoint_management.update_checkpoint_state_internal(
                self._directory, self._checkpoints[-1], self._checkpoints)
        return filename_prefix

    def chdir(self, directory):
        self._directory = directory
        self._CACHE_PREFIX = os.path.join(self._directory, "tmp")
        # Reset tracked checkpoints
        self._checkpoints.clear()

        if self._cached_checkpoint != None:
            # Clear checkpoint cache
            for filename in self._get_checkpoint_filenames(
                    self._cached_checkpoint):
                os.remove(filename)
            self._cached_checkpoint = None

    @property
    def latest_checkpoint(self):
        if len(self._checkpoints) > 0:
            return self._checkpoints[-1]
        else:
            return ""

    def save(self, prefix, session):
        checkpoint_name = self._checkpoint.save(prefix, session)

        if len(self._checkpoints) == self._max_to_keep:
            # Remove first checkpoint in FIFO
            for filename in self._get_checkpoint_filenames(
                    self._checkpoints.pop(0)):
                os.remove(filename)
        # Append last checkpoint
        self._checkpoints.append(checkpoint_name)
        return checkpoint_name

    def write(self, prefix, session):
        return self._checkpoint.write(prefix, session)

    def _get_checkpoint_filenames(self, prefix):
        _d = "[0-9]"
        filenames = []
        for filename in [glob.glob(prefix + ".index"),
                         glob.glob(prefix + ".meta"),
                         glob.glob(prefix + ".data-" + 5*_d + "-of-" + 5*_d)]:
            filenames.extend(filename)
        return filenames
