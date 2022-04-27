from .output_base import OutputBase
import numpy as np

class InMemoryOutput(OutputBase):
    _aliases = ["memory"]
    def __init__(self, apply_blinding_offsets=False, blinding_offset_file=None):
        super(InMemoryOutput,self).__init__()
        self.rows = []
        self.meta = {}
        self.final_meta = {}
        self.comments = []

        self.apply_blinding_offsets = apply_blinding_offsets
        if apply_blinding_offsets:
            self._blinding_offsets = np.load(blinding_offset_file)

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
        apply_blinding_offsets = utils.boolean_string(options.get('apply_blinding_offsets', False))
        print("apply_blinding_offsets", type(apply_blinding_offsets), apply_blinding_offsets)
        blinding_offset_file = options.get('blinding_offsets', None)
        if apply_blinding_offsets & (blinding_offset_file is None):
            raise RuntimeError("You set apply_blinding_offsets but did not provide blinding_offset_file")
        return cls(apply_blinding_offsets=apply_blinding_offsets, blinding_offset_file=blinding_offset_file)

    @classmethod
    def load_from_options(cls, options):
        raise ValueError("No output was saved from this run")
