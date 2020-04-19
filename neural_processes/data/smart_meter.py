from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from diskcache import Cache

cache = Cache(".cache")

def npsample_batch(x, y, size=None, sort=True):
    """Sample from numpy arrays along 2nd dim."""
    inds = np.random.choice(range(x.shape[1]), size=size, replace=False)
    if sort:
        inds.sort()
    return x[:, inds], y[:, inds]

def collate_fns(max_num_context, max_num_extra_target, sample, sort=True, context_in_target=True):
    def collate_fn(batch, sample=sample):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(4, max_num_context)
        num_extra_target = np.random.randint(4, max_num_extra_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Last feature will show how far in time a point is from our last context
        assert (np.diff(x[:, :, 0], 1)>=0).all(), 'first features should be ordered e.g. seconds'
        assert (x[:, max_num_context, -1]==0.).all(), 'last features should be empty'
        time = x[:, :, 0]
        t0 = x[:, max_num_context, 0][:, None]
        x[:, :, -1] = time - t0  # Feature to let the model know this is past data
        
        x_context = x[:, :max_num_context]
        y_context = y[:, :max_num_context]
    
        x_target_extra = x[:, max_num_context:]
        y_target_extra = y[:, max_num_context:]
        
        if sample:

            # This is slightly differen't than normal, we are ensuring that our target point are in the future, to mimic deployment
            x_context, y_context = npsample_batch(
                x_context, y_context, size=num_context, sort=sort
            )

            x_target_extra, y_target_extra = npsample_batch(
                x_target_extra, y_target_extra, size=num_extra_target, sort=sort
            )

        # do we want to compute loss over context+target_extra, or focus in on only target_extra?
        if context_in_target is True:
            x_target = torch.cat([x_context, x_target_extra], 1)
            y_target = torch.cat([y_context, y_target_extra], 1)
        else:
            x_target = x_target_extra
            y_target = y_target_extra

        assert (x[:, -1, -1] > 0).all()
        assert (x[:, 0, -1] < 0).all()
        
        return x_context, y_context, x_target, y_target

    return collate_fn


class SmartMeterDataSet(torch.utils.data.Dataset):
    def __init__(self, df, num_context=40, num_extra_target=10, label_names=['energy(kWh/hh)']):
        self.df = df
        self.num_context = num_context
        self.num_extra_target = num_extra_target
        self.label_names = label_names

    def get_rows(self, i):
        rows = self.df.iloc[i : i + (self.num_context + self.num_extra_target)].copy()
        rows['tstp'] = (rows['tstp'] - rows['tstp'].iloc[0]).dt.total_seconds() / 86400.0
        rows = rows.sort_values('tstp')

        # make sure tstp, which is our x axis, is the first value
        columns = ['tstp'] + list(set(rows.columns) - set(['tstp', 'block'])) + ['future']
        rows['future'] = 0.
        rows = rows[columns]

        # This will be the last row, and will change it upon sample to let the model know some points are in the future

        x = rows.drop(columns=self.label_names).copy()
        y = rows[self.label_names].copy()
        return x, y


    def __getitem__(self, i):
        x, y = self.get_rows(i)
        return x.values, y.values
        
    def __len__(self):
        return len(self.df) - (self.num_context + self.num_extra_target)


def load_weather_csv(infile):
           
    # Load weather data
    df_weather = pd.read_csv(infile, parse_dates=[3])
    use_cols = ['visibility', 'windBearing', 'temperature', 'time', 'dewPoint',
        'pressure', 'apparentTemperature', 'windSpeed', 
        'humidity']
    df_weather = df_weather[use_cols].set_index('time')

    # Resample to match energy data    
    df_weather = df_weather.resample('30T').ffill()

    # Normalise
    weather_norms=dict(mean={'visibility': 11.2,
    'windBearing': 195.7,
    'temperature': 10.5,
    'dewPoint': 6.5,
    'pressure': 1014.1,
    'apparentTemperature': 9.2,
    'windSpeed': 3.9,
    'humidity': 0.8},
    std={'visibility': 3.1,
    'windBearing': 90.6,
    'temperature': 5.8,
    'dewPoint': 5.0,
    'pressure': 11.4,
    'apparentTemperature': 6.9,
    'windSpeed': 2.0,
    'humidity': 0.1})

    for col in df_weather.columns:
        df_weather[col] -= weather_norms['mean'][col]
        df_weather[col] /= weather_norms['std'][col]
    return df_weather

def f2i(f: Path) -> int:
    """block_2.csv->2"""
    return int(f.stem.split('_')[-1])

def is_test(f):
    return f2i(f) % 8 == 1

def is_val(f):
    return f2i(f) % 7==1

@cache.memoize()
def get_smartmeter_df(indir=Path('./data/smart-meters-in-london'), max_files=60, use_logy=False):
    
    df_weather = load_weather_csv(indir/'weather_hourly_darksky.csv')    

    # Also find bank holidays
    df_hols = pd.read_csv(indir/'uk_bank_holidays.csv', parse_dates=[0])
    holidays = set(df_hols['Bank holidays'].dt.round('D'))  

    def load_csv(f):
        df = pd.read_csv(f, parse_dates=[1], na_values=['Null'])

        # Do a whole block as one series
        df = df.groupby('tstp').mean()
        df = df.sort_values('tstp')

        df['block'] = f2i(f)

        # Drop nan and 0's
        df = df[df['energy(kWh/hh)'] != 0]
        df = df.dropna()
        # df.index.name = 'tstp'
        df['tstp'] = df.index

        # join weather and holidays
        df = pd.concat([df, df_weather], 1).dropna()
        df['holiday'] = df.tstp.apply(lambda dt: dt.floor('D') in holidays).astype(int)

        # Add time features
        time = df.tstp
        df["month"] = time.dt.month / 12.0
        df['day'] = time.dt.day / 310.0
        df['week'] = time.dt.week / 52.0
        df['hour'] = time.dt.hour / 24.0
        df['minute'] = time.dt.minute / 24.0
        df['dayofweek'] = time.dt.dayofweek / 7.0

        if use_logy:
            df['energy(kWh/hh)'] = np.log(df['energy(kWh/hh)']+1e-4)
        return df
    
    csv_files = list((indir / 'halfhourly_dataset').glob('*.csv'))
    csv_files.sort(key=f2i)
    csv_files = csv_files[:max_files]
    
    test_files = [f for f in csv_files if is_test(f)]
    val_files = [f for f in csv_files if is_val(f) and (not is_test(f))]
    train_files = [f for f in csv_files if (not is_val(f)) and (not is_test(f))]
    print(len(train_files), len(val_files), len(test_files))
    print(train_files, val_files, test_files)
    assert not set(train_files).intersection(set(test_files), set(val_files))
    assert not set(test_files).intersection(set(val_files))

    df_test = pd.concat([load_csv(f) for f in tqdm(test_files, desc='test csv')], 0)
    df_val = pd.concat([load_csv(f) for f in tqdm(val_files, desc='val csv')], 0)
    df_train = pd.concat([load_csv(f) for f in tqdm(train_files, desc='train csv')], 0)
    return df_train, df_val, df_test
