import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import vectorbtpro as vbt
from dateutil.parser import parse
from polygon import CryptoClient, ReferenceClient, StocksClient
from retry import retry
from vctr.data.labeling import label_data_extrema_multi
from vctr.features.feature_engineering import add_features, clean_data
from vctr.utils.cache import cache_plz

reference_client = ReferenceClient(os.environ['POLYGON_API_KEY'])
crypto_client = CryptoClient(os.environ['POLYGON_API_KEY'])
stocks_client = StocksClient(os.environ['POLYGON_API_KEY'])


def get_data(symbol: str, timeframe: str, start=None, end=None, crypto=True) -> pd.DataFrame:
    n_workers = os.cpu_count() * 7

    binance_symbol = symbol

    # Ensure that `USD` is appended to the symbol.
    if crypto is True:
        if not symbol.endswith('USD'):
            symbol += 'USD'
            binance_symbol = symbol

        # Ensure that `USD` is appended to the symbol.
        if not symbol.startswith('X:'):
            symbol = f'X:{symbol}'

    if start is None:
        start = '2017-01-01'
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set a maximum chunk size of 30 days
    max_chunk_size = timedelta(days=30)

    def chunk_date_range(start: datetime, end: datetime, delta: timedelta) -> Tuple[datetime, datetime]:
        while start < end:
            next_start = min(start + delta, end)
            yield start, next_start
            start = next_start

    # Convert input date strings to datetime objects.
    start_date = parse(start) if isinstance(start, str) else start
    end_date = parse(end) if isinstance(end, str) else end

    # Calculate the chunk size. If necessary, adjust to ensure no chunk is larger than 30 days.
    total_days = (end_date - start_date).days
    if total_days > max_chunk_size.days:
        chunk_size = max_chunk_size.days
    else:
        chunk_size = max(1, total_days // n_workers)

    # Generate chunks of date ranges.
    date_chunks = list(chunk_date_range(start_date, end_date, timedelta(days=chunk_size)))

    # Define a function to fetch data for a specific date range.
    def fetch_data_chunk(symbol: str, timeframe: str, date_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        start, end = date_range
        return fetch_polygon_aggs(
            symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            crypto=crypto,
        )

    # Create a partial function with the symbol and timeframe arguments pre-filled.
    fetch_data_chunk_partial = partial(fetch_data_chunk, symbol, timeframe)

    # Fetch data for all date range chunks in parallel using a ThreadPoolExecutor.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        data_frames = list(executor.map(fetch_data_chunk_partial, date_chunks))

    if not data_frames or all(df.empty for df in data_frames):
        return pd.DataFrame()

    if crypto:
        # Now also fetch the last 10 bars from Binance because Polygon doesn't have
        # the latest data. Go through vectorbtpro to get the latest bars.
        binance_data, _ = vbt.BinanceData.fetch_symbol(
            symbol=binance_symbol,
            timeframe=timeframe,
            start=pd.Timestamp.now() - pd.Timedelta(hours=6),
            show_progress=False,
            silence_warnings=True,
        )

        binance_data = binance_data.rename(
            columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        )
        data_frames.append(binance_data[['open', 'high', 'low', 'close', 'volume']])

    # Concatenate the fetched data, remove duplicates, and sort by time.
    result = pd.concat(data_frames).drop_duplicates().sort_index()

    return pd.DataFrame(result)


@cache_plz('/Users/vince/vctr/vctr/features/feature_engineering.py')
def get_data_with_features(symbol: str, timeframes: Union[str, List[str]], start=None, end=None, crypto=True):
    # Ensure timeframes is a list
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    dataframes = []

    original_timeframe = timeframes[0]

    for timeframe in timeframes:
        data = get_data(symbol, timeframe, start, end, crypto=crypto)

        if data.empty:
            # Throw an error if no data was fetched
            raise ValueError(f'No data for {symbol} in {timeframe} timeframe')

        data = add_features(data)
        data = clean_data(data)

        # Rename all the column names by appending `_{timeframe}`
        if timeframe is not None and timeframe != original_timeframe:
            print('Changing column names...')
            data = data.add_suffix(f'_{timeframe}')

        dataframes.append(data)

    # Combine all the DataFrames by broadcasting their features
    combined_data = dataframes[0]

    for i in range(1, len(dataframes)):
        smaller_timeframe_df = dataframes[i - 1]
        larger_timeframe_df = dataframes[i]

        # Drop duplicates before resampling
        smaller_timeframe_df = smaller_timeframe_df.loc[~smaller_timeframe_df.index.duplicated(keep='first')]
        larger_timeframe_df = larger_timeframe_df.loc[~larger_timeframe_df.index.duplicated(keep='first')]

        frequency = (
            smaller_timeframe_df.index.to_series().diff().mode()[0]
        )  # Get the frequency of the smaller timeframe
        resampled_larger_timeframe_df = larger_timeframe_df.resample(
            frequency
        ).ffill()  # Resample and forward fill the larger timeframe

        combined_data = smaller_timeframe_df.join(
            resampled_larger_timeframe_df, how='left', lsuffix='_smaller', rsuffix='_larger'
        )

    if end is not None:
        combined_data = combined_data.loc[:end]

    return combined_data


@cache_plz('/Users/vince/vctr/vctr/features/feature_engineering.py')
def get_data_with_features_and_labels(
    symbol: str,
    timeframes: Union[str, List[str]],
    label_args=None,
    start=None,
    end=None,
    separate=True,
    crypto=True,
):
    # Ensure timeframes is a list
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    if label_args is None:
        label_args = (0.05, 0.01)

    data = get_data_with_features(symbol, timeframes, start, end, crypto=crypto)
    data = label_data_extrema_multi(data, *label_args)

    # cat_clf = CatBoostClassifier().load_model('trend_clf.model')
    # X_clf = StandardScaler().fit_transform(data.copy())
    # data['trend_clf'] = pd.Series(cat_clf.predict(X_clf), index=data.index)
    # data = label_data_trends_binary(data)

    return (data.drop(columns=['label']), data['label']) if separate else data


def scale_all_but_ohlc(data):
    ohlc = data[['open', 'high', 'low', 'close']]
    data = data.drop(columns=['open', 'high', 'low', 'close'])

    # Store column names before transforming the data
    column_names = data.columns

    data = StandardScaler().fit_transform(data)
    data = pd.DataFrame(data, index=ohlc.index, columns=column_names)
    data = pd.concat([data, ohlc], axis=1)
    return data


# @retry(delay=1, backoff=2, max_delay=4)
def fetch_polygon_aggs(symbol, timeframe, start, end, crypto=True):
    multiplier, timespan = int(timeframe[:-1]), timeframe[-1].lower()
    timespan = dict(zip('mhd', ['minute', 'hour', 'day']))[timespan]

    if crypto:
        ohlcv = crypto_client.get_full_range_aggregate_bars(
            symbol,
            start,
            end,
            multiplier=multiplier,
            timespan=timespan,
            adjusted=False,
            warnings=False,
            run_parallel=False,
            high_volatility=True,
        )
    else:
        ohlcv = stocks_client.get_aggregate_bars(
            symbol,
            start,
            end,
            multiplier=multiplier,
            timespan=timespan,
            adjusted=True,
            warnings=False,
            run_parallel=False,
            high_volatility=True,
            full_range=True,
        )

    # Convert to DataFrame and rename columns
    df = pd.DataFrame(ohlcv)

    # If there is no data, return an empty DataFrame
    if df.empty:
        return df

    for col in ['vw', 'n']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.rename(
        columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            't': 'date',
        },
        inplace=True,
    )

    df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df.set_index('date', inplace=True)
    df.index.name = 'date'
    df.attrs['symbol'] = symbol
    df.attrs['timeframe'] = timeframe

    return df


def fetch_polygon_symbols_list():
    tickers = reference_client.get_tickers(market='crypto')['results']
    return [x['base_currency_symbol'] for x in tickers]
