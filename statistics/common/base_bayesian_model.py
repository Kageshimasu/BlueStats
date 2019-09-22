from abc import ABCMeta, abstractmethod


class BayesianModel(metaclass=ABCMeta):

    @abstractmethod
    def update(self, x):
        pass

    @abstractmethod
    def get_sample(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def pdf(self, x):
        pass
