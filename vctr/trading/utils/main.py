import pandas as pd
from vctr.trading.client import binance_client


def get_usd_trade_symbols():
    # get exchange info from the API
    exchange_info = binance_client.get_exchange_info()

    # filter symbols by quote asset equal to USD
    usd_pairs = [symbol['baseAsset'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USD4']

    # print the list of USD pairs
    return usd_pairs


def get_daily_volume(symbol='BTC'):
    # call the get_ticker() method with the specified symbol
    ticker = binance_client.get_ticker(symbol=f'{symbol}USD')

    # extract the 24 hour volume from the ticker data
    return float(ticker['volume']) * float(ticker['lastPrice'])


def get_symbols_with_min_volume(min_volume=250000):
    # get all symbols
    symbols = get_usd_trade_symbols()

    # filter symbols by minimum daily volume
    symbols_with_min_volume = [symbol for symbol in symbols if get_daily_volume(symbol) > min_volume]

    # print the list of symbols with minimum daily volume
    return symbols_with_min_volume


def get_symbols_and_volumes():
    # get all symbols
    symbols = get_usd_trade_symbols()

    # get the volume for each symbol
    volumes = [get_daily_volume(symbol) for symbol in symbols]

    # create a DataFrame of symbols and volumes
    df = pd.DataFrame({'symbol': symbols, 'volume': volumes})

    # sort the DataFrame by volume
    df = df.sort_values('volume', ascending=False)

    # print the DataFrame
    return df
