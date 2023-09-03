import os

import ccxt
from binance.client import Client


# Binance API credentials (source from env vars)
binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')

# Initialize the Binance API client
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret, tld='us')
ccxt_client = ccxt.binanceus(
    {
        'apiKey': binance_api_key,
        'secret': binance_api_secret,
        'enableRateLimit': True,
    }
)
