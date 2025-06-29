from abc import ABC, abstractmethod

class BaseAssetGenerator(ABC):
    @abstractmethod
    def generate_asset(self, image):
        pass
