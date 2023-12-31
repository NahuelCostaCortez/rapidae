import os
from abc import ABC
from datetime import datetime

from rapidae.conf import Logger


class BasePipeline(ABC):
    """
    Base class for all pipelines in the library.

    This base class provides a structure for creating pipelines in the library.
    All data produced with the pipelines will be saved in the 'output_dir'.
    A folder 'PipelineName_YYY-MM-DD_HH-MM-SS' will be created inside 'output_dir',
    where data specific to a pipeline will be saved.

    Args:
        name (str): Name of the pipeline. Defaults to the class name if not provided.
        output_dir (str): Directory where the pipeline output will be saved. Defaults to "output_dir".

    Usage:
        # Example usage of a custom pipeline class
        class CustomPipeline(BasePipeline):
            def __call__(self):
                # Custom pipeline logic
                pass
        custom_pipeline = CustomPipeline(name='CustomPipeline', output_dir='output_data')
        custom_pipeline()
    """

    def __init__(self, name=None, output_dir="output_dir"):
        if name == None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.output_dir = output_dir
        self.logger = Logger()


    def __call__(self):
        """
        Launches the pipeline and creates a directory for saving output.

        Returns:
            None
        """
        # create a dir self.output_dir/training_YYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(
            '..', self.output_dir, f"{self.name}_{timestamp}")

        self.logger.log_info('+++ {} +++'.format(self.name))
        self.logger.log_info('Creating folder in {}'.format(folder_path))

        os.makedirs(folder_path)
        self.output_dir = str(folder_path)
