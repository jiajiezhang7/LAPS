import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat

from amplify.utils.cfg_utils import get_device
from amplify.utils.kp_utils.query import Query
from amplify.utils.libero_utils.flow_utils import sample_double_grid


def grid_queries(views, n_tracks, device, sample=False, grid_size=None):
    """
    Generates a grid of query points (padding at edge of image) in the standard format.
    """
    grid_size = int((n_tracks)**0.5) if grid_size is None else grid_size
    # assert grid_size**2 == n_tracks, f"n_tracks must be a perfect square, got {n_tracks}"
    padding = 1 / grid_size
    min_val = -1 + padding
    max_val = 1 - padding
    x = torch.linspace(min_val, max_val, grid_size)
    y = torch.linspace(min_val, max_val, grid_size)
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack([yy, xx], dim=-1).reshape(-1, 2).to(device) # (n_tracks, 2)
    points = repeat(points, "n d -> v n d", v=views) # (views, n_tracks, 2)

    if sample:
        # sample n_tracks points from each of the two views
        # (views, grid_size**2, 2) -> (views, n_tracks, 2)
        new_points = torch.empty((views, n_tracks, 2), device=device)
        for i in range(views):
            new_points[i] = points[i][torch.randperm(points[i].shape[0])[:n_tracks]]
        points = new_points

    return Query(points)

def grid_queries_nonsquare(views, n_tracks, device, image_height, image_width, sample=False, grid_size=None):
    """
    Generates a grid of query points (padding at edge of image) in the standard format for non-square images.
    """
    # Calculate the grid dimensions based on the aspect ratio
    aspect_ratio = image_width / image_height
    if grid_size is None:
        grid_height = int((n_tracks / aspect_ratio)**0.5)
        grid_width = int(grid_height * aspect_ratio)
    else:
        grid_height = grid_size
        grid_width = int(grid_height * aspect_ratio)
    
    padding_x = 1 / grid_width
    padding_y = 1 / grid_height
    min_val_x = -1 + padding_x
    max_val_x = 1 - padding_x
    min_val_y = -1 + padding_y
    max_val_y = 1 - padding_y
    
    x = torch.linspace(min_val_x, max_val_x, grid_width)
    y = torch.linspace(min_val_y, max_val_y, grid_height)
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack([yy, xx], dim=-1).reshape(-1, 2).to(device)  # (grid_height * grid_width, 2)
    points = repeat(points, "n d -> v n d", v=views)  # (views, grid_height * grid_width, 2)

    if sample:
        # Sample n_tracks points from each of the views
        new_points = torch.empty((views, n_tracks, 2), device=device)
        for i in range(views):
            new_points[i] = points[i][torch.randperm(points[i].shape[0])[:n_tracks]]
        points = new_points

    return Query(points)

def atm_queries(views, n_tracks, device):
    """
    Generates double grid of query points like atm
    """
    tracks_per_grid = n_tracks // 2
    grid_size = int((tracks_per_grid)**0.5)
    assert 2 * grid_size**2 == n_tracks, f"n_tracks must be two times a perfect square, got {n_tracks}"
    points = sample_double_grid(grid_size, device=device) # (num_points, 2) with (col, row) in [0,1]
    points = points * 2 - 1 # normalize to [-1, 1]
    points = points[...,[1,0]] # swap (col, row) to (row, col)
    points = repeat(points, "n d -> v n d", v=views) # (views, n_tracks, 2)

    return Query(points)


class MultiViewImageClicker:
    def __init__(self, images, n_tracks, save_dir="./saved_queries"):
        """
        images: A numpy array of shape (views, H, W, C)
        n_tracks: Number of clicks (coordinates) to collect per view
        """
        self.images = images
        self.n_tracks = n_tracks
        self.save_dir = save_dir
        self.coords = []  # To store coordinates for all views
        self.current_coords = []  # To store coordinates for the current view
        self.ax = None # Placeholder for the current axis

    def prompt_load_or_click(self):
        """
        Prompt the user to load saved coordinates or proceed with clicking.
        """
        choice = input("Do you want to load saved coordinates? (y/n): ").strip().lower()
        if choice == 'y':
            # print available files
            print("Available files:")
            for file in os.listdir(self.save_dir):
                if file.endswith(".json"):
                    print(file)
            filename = input(f"Enter the filename to load (relative to {self.save_dir}, without extension): ")
            path = os.path.join(self.save_dir, filename) + ".json"
            print(path)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    self.coords = json.load(file)
                print("Coordinates loaded successfully.")
                return True
            else:
                print("File not found. Proceeding with click collection.")
        return False
    
    def onclick(self, event):
        """Event handler for mouse click."""
        if event.xdata is not None and event.ydata is not None:
            row, col = int(event.ydata), int(event.xdata)
            self.current_coords.append((row, col))  # Append the clicked coordinates to the current list
            self.ax.plot(col, row, 'ro')  # Plot the point immediately
            plt.draw()
            progress = len(self.current_coords)
            print(f"Clicked at: ({row}, {col})")
            print(f"{progress}/{self.n_tracks} clicks recorded for current view.")
            if progress >= self.n_tracks:
                plt.close()  # Close the figure once the required number of clicks have been recorded

    def display_and_collect(self):
        """Displays each view and collects clicks."""
        if self.prompt_load_or_click():
            return self.coords
        
        for view in range(self.images.shape[0]):
            self.current_coords = []  # Reset/initialize the list for the current view
            fig, self.ax = plt.subplots()
            self.ax.imshow(self.images[view])
            fig.canvas.mpl_connect('button_press_event', self.onclick)
            print(f"View {view+1}/{self.images.shape[0]}: Please click on the image {self.n_tracks} times.")
            plt.show()
            self.coords.append(self.current_coords)  # Append the coordinates of the current view to the main list

        # Prompt to save
        if input("Do you want to save these coordinates? (y/n): ").strip().lower() == 'y':
            filename = input("Enter the filename to save (without extension): ") + ".json"
            path = os.path.join(self.save_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as file:
                json.dump(self.coords, file)
            print("Coordinates saved successfully.")

        return self.coords
    

def load_coords(filename, save_dir="./saved_queries"):
    """
    Load saved coordinates from a file
    args:
        filename: str - The filename to load coords from.
    returns:
        coords: list - The list of coordinates.
    """
    path = os.path.join(save_dir, filename) + ".json"
    if os.path.exists(path):
        with open(path, 'r') as file:
            coords = json.load(file)
        print("Coordinates loaded successfully.")
        return coords
    else:
        print(f"File not found: {path}")
        raise FileNotFoundError
    

def click_queries(img, n_tracks, device, filename=None):
    """
    Prompts user to select query points interactively
    args:
        img: np.ndarray (views, H, W, C) - The image(s) on which the user will click.
        n_tracks: int - The number of clicks (per image) required from the user.
        device: torch.device - The device on which the queries will be stored.
        filename: str - The filename to load queries from.
    returns:
        Query: The queries in the standard format.
    """
    if filename is not None:
        coords = load_coords(filename)
    else:
        clicker = MultiViewImageClicker(img, n_tracks)
        coords = clicker.display_and_collect() # (views, n_tracks, 2) with (row, col) in pixel coordinates

    # Convert coords to a PyTorch tensor and normalize to [-1, 1]
    views = img.shape[0]
    img_size = img.shape[1]
    queries = torch.empty((views, n_tracks, 2), device=device)
    for i, view_coords in enumerate(coords):
        queries[i] = torch.tensor(view_coords, device=device) # (n_tracks, 2) with (row, col) in pixel coordinates
        queries[i] = (2 * queries[i] / img_size) - 1 # normalize to [-1, 1]

    return Query(queries)


def query_from_tracks(track, t):
    """
    Take track at timestep t and return queries in standard format
    args:
        track: torch.Tensor (views, track_length, n_tracks, 2)
        t: int - The timestep at which to extract the tracks.
    """
    queries = track[:, t, :, :]
    return Query(queries, allow_out_of_frame=True)


def _resample_near_moving(tracks, std):
    """
    Resample the tracks that moved most, with some noise
    args:
        tracks: torch.Tensor (views, n_samples, horizon, n_tracks, 2)
        std: float - The standard deviation of the noise to add to the queries.
    """
    v, ns, t, nt, _ = tracks.shape
    # Compute track lengths
    lengths = torch.linalg.norm(tracks[:, :, 1:] - tracks[:, :, :-1], dim=-1).sum(dim=(1,2)) # (views, n_tracks)
    # choose n_tracks with probability proportional to track length
    probs = lengths / lengths.sum(dim=1, keepdim=True) # (views, n_tracks)
    chosen_idxs = torch.multinomial(probs, nt, replacement=True) # (views, n_tracks)
    
    # get previous queries (first track points, same across all samples)
    prev_queries = tracks[:, 0, 0, :, :] # (views, n_tracks, 2)

    # get new queries as chosen idxs of previous queries
    queries = prev_queries.gather(1, chosen_idxs.unsqueeze(2).expand(v, nt, 2)) # (views, n_tracks, 2)

    # add noise
    noise = torch.randn_like(queries) * std
    queries += noise

    # clamp to [-1, 1]
    queries = torch.clamp(queries, -1, 1) # (views, n_tracks, 2)

    return queries


def _resample_some_near_moving(tracks, std, resample_rate, random=True):
    """
    Resample a percentage of the tracks that moved most, with some noise
    args:
        tracks: torch.Tensor (views, n_samples, horizon, n_tracks, 2)
        std: float - The standard deviation of the noise to add to the queries.
        resample_rate: float - The percentage of tracks to resample.
        random: bool - If True, remaining queries are chosen randomly in [-1, 1], else they are the first track points.
    """
    n_tracks = tracks.shape[3]
    n_resample = int(n_tracks * resample_rate)
    # pick random indices to resample
    resample_indices = torch.randperm(n_tracks)[:n_resample]
    resample_tracks = tracks[:, :, :, resample_indices, :] # (views, n_samples, horizon, n_resample, 2)
    resample_queries = _resample_near_moving(resample_tracks, std)
    if random:
        remaining_queries = 2 * torch.rand(tracks.shape[0], n_tracks-n_resample, 2, device=tracks.device) - 1 # (views, n_tracks-n_resample, 2)
    else:
        remaining_queries = tracks[:, 0, 0, ~resample_indices, :] # (views, n_tracks-n_resample, 2)
    queries = torch.cat([resample_queries, remaining_queries], dim=1) # (views, n_tracks, 2)

    return queries


def query_from_moving_tracks(tracks, resample_rate=0.7, std=0.05):
    """
    Resample queries near the tracks that moved most
    args:
        tracks: torch.Tensor (views, n_samples, horizon, n_tracks, 2)  
    """
    queries = _resample_some_near_moving(tracks, std, resample_rate)

    return Query(queries)

    
def query_from_moving_tracks_agentview(tracks, resample_rate=0.7, std=0.05):
    """
    Resample queries near the tracks that moved most, but only in agentview
    args:
        tracks: torch.Tensor (2, n_samples, horizon, n_tracks, 2)  
    """
    # resample percentage of agentview queries by track length
    agentview_tracks = tracks[0, :, :, :, :].unsqueeze(0) # (1, n_samples, horizon, n_tracks, 2)
    agentview_queries = _resample_some_near_moving(agentview_tracks, std, resample_rate) # (1, n_tracks, 2)

    # keep eyeinhand queries the same
    eyeinhand_queries = tracks[1, 0, 0, :, :].unsqueeze(0) # (1, n_tracks, 2)

    queries = torch.cat([agentview_queries, eyeinhand_queries], dim=0) # (2, n_tracks, 2)

    return Query(queries)


if __name__ == "__main__":
    img_size = 128
    views = 2
    n_tracks = 400
    img = torch.zeros(views, img_size, img_size, 3)  # Example image
    device = get_device()
    queries = grid_queries(img.shape[0], n_tracks, device)
    square_queries = grid_queries_nonsquare(img.shape[0], n_tracks, device, image_height=img_size, image_width=img_size)
    nonsquare_queries = grid_queries_nonsquare(img.shape[0], n_tracks, device, image_height=img_size*2, image_width=img_size*3)
    assert torch.allclose(queries.standard(), square_queries.standard()), f"Queries should have the same value, got shapes {queries.standard().shape} and {nonsquare_queries.standard().shape}"
    print("queries match for square and non-square images")
    # queries = atm_queries(img.shape[0], n_tracks, device)
    # queries = click_queries(img, n_tracks, device)
    print(queries.standard())
    print(queries.atm(2))
    print(queries.cotracker(img_size))

    display_view = 1
    # # plot queries on blank image with origin in top left
    # coords = queries.standard().cpu().numpy()[display_view]
    # plt.imshow(img[display_view], extent=[-1, 1, -1, 1])
    # plt.scatter(coords[:, 1], coords[:, 0])
    # plt.gca().invert_yaxis()
    # plt.show()

    # nonsquare_coords = nonsquare_queries.standard().cpu().numpy()[display_view]
    # plt.imshow(img[display_view], extent=[-1, 1, -1, 1])
    # plt.scatter(nonsquare_coords[:, 1], nonsquare_coords[:, 0])
    # plt.gca().invert_yaxis()
    # plt.show()

    # sample subset of queries
    num_samples = 72
    sampled_coords = queries.sample(num_samples).standard().cpu().numpy()[display_view]
    plt.imshow(img[display_view], extent=[-1, 1, -1, 1])
    plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0])
    plt.gca().invert_yaxis()
    plt.show()

    # # sample subset of queries with manual queries
    # # manual_queries = Query(torch.rand(views, 100, 2) * 2 - 1)
    # manual_queries = click_queries(img, num_samples, device, "manual_72")
    # sampled_coords = queries.sample(num_samples, manual_queries).standard().cpu().numpy()[display_view]
    # manual_coords = manual_queries.standard().cpu().numpy()[display_view]
    # plt.imshow(img[display_view], extent=[-1, 1, -1, 1])
    # plt.scatter(sampled_coords[:, 1], sampled_coords[:, 0])
    # plt.scatter(manual_coords[:, 1], manual_coords[:, 0], c='r')
    # plt.legend(["Sampled", "Target (Manual)"])
    # plt.gca().invert_yaxis()
    # plt.show()