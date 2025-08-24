from .output_base import OutputBase
import numpy as np

class InMemoryOutput(OutputBase):
    _aliases = ["memory"]
    def __init__(self, blinding_offset_file=None):
        super(InMemoryOutput,self).__init__()
        self.rows = []
        self.meta = {}
        self.final_meta = {}
        self.comments = []

        if blinding_offset_file is not None:
            self._blinding_offsets = np.load(blinding_offset_file)
        else:
            self._blinding_offsets = None

    def _write_parameters(self, params):
        self.rows.append(params)

    def _write_metadata(self, key, value, comment):
        self.meta[key] = (value,comment)

    def _write_comment(self, comment):
        self.comments.append(comment)

    def _write_final(self, key, value, comment):
        self.final_meta[key] = (value,comment)

    def __getitem__(self, key_or_index):
        if isinstance(key_or_index, int):
            return self.rows[key_or_index]
        else:
            column_index = [c[0] for c in self.columns].index(key_or_index)
            return np.array([row[column_index] for row in self.rows])

    @classmethod
    def from_options(cls, options, resume=False):
        if resume:
            raise ValueError("Cannot resume from in-memory output")
        blinding_offset_file = options.get('blinding_offsets', None)
        return cls(blinding_offset_file=blinding_offset_file)

    @classmethod
    def load_from_options(cls, options):
        raise ValueError("No output was saved from this run")
    
    def reset_to_chain_start(self):
        self.rows = []


