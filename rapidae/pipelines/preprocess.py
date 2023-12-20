from typing import Union

from rapidae.pipelines.base import BasePipeline
from rapidae.conf import Logger


class PreprocessPipeline(BasePipeline):

    def __init__(self, name=None, output_dir="output_dir", preprocessor=Union[list, callable]):
        super().__init__(name, output_dir)
        self.preprocessor = preprocessor

    def __call__(self, **kwargs):
        super().__call__()

        # TODO: check dataset format

        # The preprocessor contains a list of methods, these will be applied in order to the data
        if type(self.preprocessor) == list:
            self.logger.log_info(
                'Selected preprocessor is a bunch of functions.')
            for i in self.preprocessor:
                # TODO
                pass

        # The preprocessor is a custom function to be applied to a specific dataset
        if callable(self.preprocessor):
            self.logger.log_info('Selected preprocessor is a function.')
            x_train, y_train, x_val, y_val, x_test, y_test = self.preprocessor(
                **kwargs)

        return x_train, y_train, x_val, y_val, x_test, y_test
