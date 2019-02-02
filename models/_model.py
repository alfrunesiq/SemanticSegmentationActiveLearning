import abc
import sys

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})

class Model(ABC):

    @abc.abstractmethod
    def build(self):
        """
        Build the Tensorflow graph of the model and store the parameters
        in a dict organized by scopes.
        :returns: model output, model parameters
        :rtype:   tf.Tensor, dict
        """
        pass

    @abc.abstractmethod
    def get_vars(self):
        """
        Return a list of all model parameters.
        """
        pass

    @abc.abstractmethod
    def get_logits(self):
        """
        Return the unnormalized network outputs.
        """
        pass

    @abc.abstractmethod
    def get_output(self):
        """
        Returns the normalized softmax output.
        """
        pass
