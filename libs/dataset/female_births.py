"""A pytorch dataset to load daily-total-female-births dataset."""
import os

import numpy as np
import pandas as pd
import urllib3
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class DailyTotalFemaleBirthDataset(Dataset):
    """Time series dataset."""

    BASE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    CSV_FILE = "daily-total-female-births.csv"
    NAMES_FILE = "daily-total-female-births.names"

    def __init__(self, data_path: str, look_back: int = 6, train: bool = False):
        """Initialize class."""
        csv_path = os.path.join(data_path, self.CSV_FILE)
        names_path = os.path.join(data_path, self.NAMES_FILE)

        if not os.path.exists(csv_path):
            os.makedirs(data_path, exist_ok=True)
            self.__download_file(
                self.BASE_URL + self.CSV_FILE, csv_path, chunk_size=1024
            )

        if not os.path.exists(names_path):
            os.makedirs(data_path, exist_ok=True)
            self.__download_file(
                self.BASE_URL + self.NAMES_FILE, names_path, chunk_size=1024
            )

        df = pd.read_csv(csv_path, header=0)

        x = df.values

        # create a rolling window of length (look_back + 1)
        x = np.array(
            [x[i : i + (look_back + 1)] for i in range(len(x) - (look_back + 1))]
        )

        # make the last element of each instance as its label
        y = x[:, -1]
        x = x[:, :-1]

        # split into train and test
        train_size = int(len(x) * 0.8)
        if train:
            self.X = x[:train_size]
            self.y = y[:train_size]
        else:
            self.X = x[train_size:]
            self.y = y[train_size:]

    @staticmethod
    def __download_file(url: str, path: str, chunk_size: int = 1024):
        """Download a file.

        Args:
            url (str): The url of the file.
            path (str): The directory to save the file.
            chunk_size (int, optional): The chunk size to save the file.
            Defaults to 1024.
        """
        print(f"Downloading {url}")
        http = urllib3.PoolManager()
        with http.request("GET", url, preload_content=False) as r:
            with open(path, "wb") as out:
                content_bytes = r.headers.get("Content-Length")
                content_bytes = None if content_bytes is None else float(content_bytes)
                progress_bar = (
                    None
                    if content_bytes is None
                    else tqdm(total=content_bytes, unit="iB", unit_scale=True)
                )
                while True:
                    data = r.read(chunk_size)
                    if not data:
                        break
                    if progress_bar is not None:
                        progress_bar.update(len(data))
                    out.write(data)
                if progress_bar is not None:
                    progress_bar.close()

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        return self.X[idx], self.y[idx]
