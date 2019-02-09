import tensorflow as tf
from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):
    """
    base model
    """
    def __init__(self):
        self._session = None

    @abstractmethod
    def train(self, *args):
        """
        learning
        :param args:
        :return    :
        """
        pass

    @abstractmethod
    def _predict(self, inputs):
        """
        predict to create graph
        :param inputs: input data
        :return      : result
        """
        pass
    
    @abstractmethod
    def loss(self, *args):
        """
        get loss
        :param args:
        :return    : loss value
        """
        pass

    @abstractmethod
    def evaluate(self, *args):
        """
        evaluate this model
        :param inputs : input data
        :param teacher: teacher data
        :return       : result 
        """
        pass

    def save(self, file_path: str):
        """
        save this model
        :param file_path:
        """
        saver = tf.train.Saver()
        saver.save(self._session, file_path)

    def load(self, file_path: str):
        """
        load a model
        :param file_path:
        """
        saver = tf.train.Saver()
        saver.restore(self._session, file_path)
    
    @abstractmethod
    def predict(self, inputs):
        """
        predict
        :param inputs: input data
        :return      : result
        """
        pass