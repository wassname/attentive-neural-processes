from pathlib import Path
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
    def collate_fn(batch, sample=sample):
        # Collate
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(4, max_num_context)
        num_extra_target = np.random.randint(4, max_num_extra_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x[:,:max_num_context, -1] = 0  # Feature to let the model know this is past data
        n=x[:, max_num_context:, -1].shape[1]
        x[:, max_num_context:, -1] = torch.arange(1, n+1)/1.0/n  # Feature to let the model know this is past data
        x_context = x[:, :max_num_context]
        y_context = y[:, :max_num_context]
    
        if sample:
            x_target_extra = x[:, max_num_context:]
            y_target_extra = y[:, max_num_context:]

            # This is slightly differen't than normal, we are ensuring that our target point are in the future, to mimic deployment
            x_context, y_context = npsample_batch(
                x_context, y_context, size=num_context, sort=sort
            )

            x_target_extra, y_target_extra = npsample_batch(
                x_target_extra, y_target_extra, size=num_extra_target, sort=sort
            )

            x = torch.cat([x_context, x_target_extra], 1)
            y = torch.cat([y_context, y_target_extra], 1)            

        assert (x_context[:, :, -1]==0).all()
        assert (x[:, -1, -1] > 0).all()
        assert (x[:, 0, -1] == 0).all()
        
        return x_context, y_context, x, y

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
        columns = ['tstp'] + list(set(rows.columns) - set(['tstp']))
        rows = rows[columns]

        # This will be the last row, and will change it upon sample to let the model know some points are in the future
        rows['future']=1

        x = rows.drop(columns=self.label_names)
        y = rows[self.label_names]
        return x, y


    def __getitem__(self, i):
        x,y = self.get_rows(i)
        return x.values, y.values
        
    def __len__(self):
        return len(self.df) - (self.num_context + self.num_extra_target)

def get_smartmeter_df(indir=Path('./data/smart-meters-in-london'), use_logy=False):
    csv_files = sorted(Path('data/smart-meters-in-london/halfhourly_dataset').glob('*.csv'))[:1]
    df = pd.concat([pd.read_csv(f, parse_dates=[1], na_values=['Null']) for f in csv_files])
#     print(df.info())

    df = df.groupby('tstp').mean()
    df['tstp'] = df.index
    df.index.name = ''

    # Load weather data
    df_weather = pd.read_csv(indir/'weather_hourly_darksky.csv', parse_dates=[3])

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

    df = pd.concat([df, df_weather], 1).dropna()
    
    
    # Also find bank holidays
    df_hols = pd.read_csv(indir/'uk_bank_holidays.csv', parse_dates=[0])
    holidays = set(df_hols['Bank holidays'].dt.round('D'))

    df['holiday'] = df.tstp.apply(lambda dt:dt.floor('D') in holidays).astype(int)

    # Add time features
    time = df.tstp
    df["month"] = time.dt.month / 12.0
    df['day'] = time.dt.day / 310.0
    df['week'] = time.dt.week / 52.0
    df['hour'] = time.dt.hour / 24.0
    df['minute'] = time.dt.minute / 24.0
    df['dayofweek'] = time.dt.dayofweek / 7.0

    # Drop nan and 0's
    df = df[df['energy(kWh/hh)']!=0]
    df = df.dropna()

    if use_logy:
        df['energy(kWh/hh)'] = np.log(df['energy(kWh/hh)']+eps)
    df = df.sort_values('tstp')
    
    # split data
    n_split = -int(len(df)*0.1)
    df_train = df[:n_split]
    df_test = df[n_split:]
    return df_train, df_test
