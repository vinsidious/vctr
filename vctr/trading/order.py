from vctr.trading.utils.orders import calculate_price, calculate_quantity
from vctr.trading.client import binance_client
from binance.helpers import round_step_size


class Order:
    def __init__(self, symbol, order_type):
        self.symbol = symbol
        self.order_type = order_type
        self.sub_orders = []
        self.total_quantity = 0
        self.weighted_average_price = 0

    def add_sub_order(self, sub_order):
        sub_order['origQty'] = float(sub_order['origQty'])
        sub_order['price'] = float(sub_order['price'])

        self.sub_orders.append(sub_order)
        self.total_quantity += sub_order['origQty']
        self.weighted_average_price = (
            self.weighted_average_price * (self.total_quantity - sub_order['origQty'])
            + sub_order['price'] * sub_order['origQty']
        ) / self.total_quantity

    def consolidate_sub_orders(self, price=None):
        if price is None:
            # Calculate the average price of the constituent sub-orders
            price = self.weighted_average_price
        else:
            # Get the current market price and compare it to the specified price
            current_market_price = binance_client.get_ticker(self.symbol)['lastPrice']
            if self.order_type == 'buy' and current_market_price < price:
                price = current_market_price
            elif self.order_type == 'sell' and current_market_price > price:
                price = current_market_price

        for sub_order in self.sub_orders:
            # Cancel each sub-order
            binance_client.cancel_order(symbol=self.symbol, orderId=sub_order['orderId'])

        symbol_info = binance_client.get_symbol_info(self.symbol)
        quantity = calculate_quantity(symbol_info, self.total_quantity)
        price = calculate_price(symbol_info, price)

        # Create a new, consolidated order at the specified/calculated price
        binance_client.create_order(
            symbol=self.symbol,
            side=self.order_type,
            quantity=quantity,
            price=price,
            type='LIMIT',
            timeInForce='GTC',
        )

    @staticmethod
    def get_all_orders():
        orders = {}

        open_orders = binance_client.get_open_orders()
        for order in open_orders:
            key = (order['symbol'], order['side'])
            if key not in orders:
                orders[key] = Order(order['symbol'], order['side'])
            orders[key].add_sub_order(order)

        return orders
