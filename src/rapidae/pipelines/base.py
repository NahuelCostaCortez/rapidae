from abc import ABC

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
        Launches the pipeline.
        """
        raise NotImplementedError
