
import pandas as pd
import numpy as np
import torch

def npsample_batch(x, y, size=None, sort=True):
    """Sample from numpy arrays along 2nd dim."""
    inds = np.random.choice(range(x.shape[1]), size=size, replace=False)
    if sort:
        inds.sort()
    return x[:, inds], y[:, inds]

def collate_fns(max_num_context, max_num_extra_target, sample, sort=True):
    def collate_fn(batch):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(4, max_num_context)
        num_extra_target = np.random.randint(4, max_num_extra_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x_context = x[:, :max_num_context]
        y_context = y[:, :max_num_context]

        if sample:
            x_target_extra = x[:, max_num_context:]
            y_target_extra = y[:, max_num_context:]

            # This is slightly differen't than normal, we are ensuring that out target point are in the future, to mimic deployment
            x_context, y_context = npsample_batch(
                x_context, y_context, size=num_context, sort=sort
            )
            x_target_extra, y_target_extra = npsample_batch(
                x_target_extra, y_target_extra, size=num_extra_target, sort=sort
            )

            x = torch.cat([x_context, x_target_extra], 1)
            y = torch.cat([y_context, y_target_extra], 1)
        return x_context, y_context, x, y

    return collate_fn


class SmartMeterDataSet(torch.utils.data.Dataset):
    def __init__(self, df, num_context=40, num_extra_target=10, label_names=['energy(kWh/hh)']):
        self.df = df
        self.num_context = num_context
        self.num_extra_target = num_extra_target
        self.label_names = label_names

    def __getitem__(self, i):
        rows = self.df.iloc[i : i + (self.num_context + self.num_extra_target)].copy()
# (df['tstp'] -  df['tstp'].iloc[0]).dt.total_seconds()
        rows['tstp'] = (rows['tstp'] - rows['tstp'].iloc[0]).dt.total_seconds() /  86400.0
        x = rows.drop(columns=self.label_names).values
        y = rows[self.label_names].values
        return x, y
        
    def __len__(self):
        return len(self.df) - (self.num_context + self.num_extra_target)
