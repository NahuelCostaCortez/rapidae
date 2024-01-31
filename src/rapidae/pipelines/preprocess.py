from typing import Union

from rapidae.pipelines.base import BasePipeline


class PreprocessPipeline(BasePipeline):
    """
    PreprocessPipeline class for applying preprocessing steps to datasets.

    This pipeline is responsible for applying preprocessing steps to datasets.
    It extends the BasePipeline class and includes functionality for handling different types
    of preprocessors, such as a list of functions or a custom callable function.

    Attributes:
        preprocessor (Union[list, callable]): Preprocessor to be applied to the dataset.
    """

    def __init__(
        self, name=None, output_dir="output_dir", preprocessor=Union[list, callable]
    ):
        super().__init__(name, output_dir)
        self.preprocessor = preprocessor

    def __call__(self, **kwargs):
        # TODO: check dataset format

        # The preprocessor contains a list of methods, these will be applied in order to the data
        if type(self.preprocessor) == list:
            self.logger.log_info("Selected preprocessor is a series of functions.")
            for i in self.preprocessor:
                # TODO
                pass

        # The preprocessor is a custom function to be applied to a specific dataset
        if callable(self.preprocessor):
            self.logger.log_info("Selected preprocessor is a function.")
            data = self.preprocessor(**kwargs)

        return data
