from .output_base import OutputBase

class NullOutput(OutputBase):
    _aliases = ["none"]
    def _write_parameters(self, params):
        pass

    def _write_metadata(self, key, value, comment):
        pass

    def _write_comment(self, comment):
        pass

    def _write_final(self, key, value, comment):
        pass

    @classmethod
    def from_options(cls, options, resume=False):
        if resume:
            raise ValueError("Cannot resume from null output")
        return cls()

    @classmethod
    def load_from_options(cls, options):
        raise ValueError("No output was saved from this run")
