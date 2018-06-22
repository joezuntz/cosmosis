from .output_base import OutputBase

class InMemoryOutput(OutputBase):
    _aliases = ["memory"]
    def __init__(self):
        super(InMemoryOutput,self).__init__()
        self.rows = []
        self.meta = {}
        self.final_meta = {}
        self.comments = []

    def _write_parameters(self, params):
        self.rows.append(params)

    def _write_metadata(self, key, value, comment):
        self.meta[key] = (value,comment)

    def _write_comment(self, comment):
        self.comments.append(comment)

    def _write_final(self, key, value, comment):
        self.final_meta[key] = (value,comment)

    @classmethod
    def from_options(cls, options):
        return cls()

    @classmethod
    def load_from_options(cls, options):
        raise ValueError("No output was saved from this run")
