import math

from binance.helpers import round_step_size
from vctr.trading.client import binance_client


def get_lot_size_filter(symbol_info):
    for f in symbol_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return f


def get_price_filter(symbol_info):
    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            return f


def calculate_quantity(symbol_info, amount):
    lot_size_filter = get_lot_size_filter(symbol_info)
    step_size = lot_size_filter['stepSize']
    min_qty = float(lot_size_filter['minQty'])
    qty = max(min_qty, amount)
    return round_step_size(qty, step_size)


def calculate_price(symbol_info, price):
    price_filter = get_price_filter(symbol_info)
    tick_size = price_filter['tickSize']
    return round_step_size(price, tick_size)


def create_limit_buy_order(symbol, quantity, price):
    symbol_info = binance_client.get_symbol_info(symbol)
    order = binance_client.create_order(
        symbol=symbol,
        side='BUY',
        type='LIMIT',
        timeInForce='GTC',
        quantity=calculate_quantity(symbol_info, quantity),
        price=calculate_price(symbol_info, price),
    )
    return order


def create_limit_sell_order(symbol, quantity, price):
    symbol_info = binance_client.get_symbol_info(symbol)
    order = binance_client.create_order(
        symbol=symbol,
        side='SELL',
        type='LIMIT',
        timeInForce='GTC',
        quantity=calculate_quantity(symbol_info, quantity),
        price=calculate_price(symbol_info, price),
    )
    return order


def get_current_price(symbol_info):
    ticker = binance_client.get_symbol_ticker(symbol=symbol_info['symbol'])
    return float(ticker['price'])
