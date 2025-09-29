import torch
from einops import repeat

################################# QUERY FORMATS #################################
#    Standard (forward_dynamics) query format:
#      Shape: (batch, n_tracks, (row, col)) 
#      Normalization: (row, col) in [-1, 1]
#
#    ATM query format:
#      Shape: (batch, track_length, n_tracks, (col, row))
#      Normalization: (col, row) in [0, 1]
#
#    Cotracker query format:
#      Shape: (batch, n_tracks, (t, col, row))
#      Normalization: (col, row) in [0, img_size-1]
#                     t in [0, n_frames-1], usually just set to 0
#################################################################################

class Query:
    """
    Class to convert standard queries to different query formats.
    """
    def __init__(self, tensor, allow_out_of_frame=False):
        # check that query conforms to standard shape and normalization
        assert tensor.ndim == 3
        assert tensor.shape[-1] == 2
        if not allow_out_of_frame:
            assert torch.all(tensor >= -1) and torch.all(tensor <= 1)
        self.tensor = tensor
        self.device = tensor.device

    def standard(self):
        return self.tensor

    def atm(self, track_length):
        tensor = self.tensor[..., [1, 0]] # swap (row, col)
        tensor = (tensor + 1) / 2 # normalize to [0, 1]
        tensor = repeat(tensor, "b n d -> b tl n d", tl=track_length) # expand to track length
        return tensor

    def cotracker(self, img_size):
        tensor = self.tensor[..., [1, 0]] # swap (row, col)
        # tensor = (tensor + 1) * (img_size - 1) / 2 # normalize to [0, img_size-1] TODO: change back to this later? (also check Track class)
        tensor = (tensor + 1)  * img_size / 2 # normalize to [0, img_size-1]
        tensor = torch.cat([torch.zeros(tensor.shape[0], tensor.shape[1], 1).to(self.device), tensor], dim=-1) # add time dimension
        return tensor

    def sample_indices(self, n_samples, manual_queries=None):
        """
        Gets `n_samples` indices from the query tensor. If `manual_queries` is provided, gets the indices of the closest points to the manual queries.

        Args:
        - n_samples (int): Number of samples to generate.
        - manual_queries: Query object with shape (batch, n_queries, 2).

        Returns:
        - indices: Tensor of indices with shape (batch, n_samples).
        """
        n_queries = self.tensor.shape[1]
        assert n_samples <= n_queries, f"Number of samples ({n_samples}) must be less than or equal to number of queries  available ({n_queries})."
        if manual_queries is None:
            indices = torch.randperm(n_queries)[:n_samples].to(self.device)
            indices = indices.repeat(self.tensor.shape[0], 1)
        else:
            assert n_samples == manual_queries.tensor.shape[1], f"Number of samples ({n_samples}) must be equal to number of manual queries provided ({manual_queries.tensor.shape[1]})."
            indices = torch.cdist(self.tensor, manual_queries.standard()).argmin(dim=1)

        return indices

    def sample(self, n_samples, manual_queries=None):
        """
        Samples `n_samples` queries from the query tensor. If `manual_queries` is provided, samples the closest points to the manual queries.

        Args:
        - n_samples (int): Number of samples to generate.
        - manual_queries: Query object with shape (batch, n_queries, 2).

        Returns:
        - sampled_queries: Query object with shape (batch, n_samples, 2).
        """
        indices = self.sample_indices(n_samples, manual_queries)
        sampled_queries = torch.gather(self.tensor, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
        sampled_queries = Query(sampled_queries, allow_out_of_frame=True)

        return sampled_queries


if __name__ == "__main__":
    # test Query class
    query = torch.rand(2, 4, 2) * 2 - 1
    query = Query(query)
    print(query.standard())
    print(query.atm(2))
    print(query.cotracker(64))
