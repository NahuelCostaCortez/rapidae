from abc import ABC
import os
from datetime import datetime

class BasePipeline(ABC):
    """
    Base class for all pipelines of the library.
    All data produced with the pipelines will be saved in ''output_dir''.
    A folder ''PipelineName_YYY-MM-DD_HH-MM-SS'' will be created inside ''output_dir'',
    where data of an specific pipeline will be saved.
    """

    def __init__(self, name=None, output_dir="output_dir"):
        if name == None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.output_dir = output_dir

    def __call__(self):
        """
        Launch the pipeline
        """
        # create a dir self.output_dir/training_YYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(self.output_dir, f"{self.name}_{timestamp}")
        os.makedirs(folder_path)