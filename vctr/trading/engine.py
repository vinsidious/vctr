import logging
import time
import sys
from datetime import datetime, timedelta
from typing import Literal

import schedule
from termcolor import colored
from vctr.models.lstm.actions import predict
from vctr.models.lstm.defaults import SEQUENCE_LENGTH
from vctr.trading.client import ccxt_client  # import the ccxt client
from vctr.trading.coins import tradable_coins


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create an instance of the TradeEngine
TIMEFRAME = '1h'
RUN_INTERVAL_MINUTES = 30
STOP_LOSS_PCT = 0.05
DUST_AMOUNT_USD = 20
TOTAL_STAKE_AMOUNT = 10 * 1000
TRADABLE_SYMBOLS = tradable_coins
BUY_ORDER_AMOUNT_USD = TOTAL_STAKE_AMOUNT / len(TRADABLE_SYMBOLS)
CURRENT_MODEL = 'lstm-mk-936-XTZ'
LABEL_ARGS = (0.035, 0.005)
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds


class TradeEngine:
    def __init__(self):
        self._balance_info = ccxt_client.fetch_balance()
        if len(sys.argv) > 1 and sys.argv[1] == 'close_all':
            self._close_all_open_trades()  # Close all open trades if the command line argument is 'close_all'

        self._run_trading_strategies()
        schedule.every(RUN_INTERVAL_MINUTES).minutes.do(self._run_trading_strategies)
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.exception('Unexpected error occurred while running scheduled tasks: %s', e)

    def _run_trading_strategies(self):
        # Get the current balance info for each run
        retries = 0
        while retries < MAX_RETRIES:
            try:
                self._balance_info = ccxt_client.fetch_balance()
                break
            except Exception as e:
                logger.error(colored('Error fetching balance: %s', 'red'), e)
                retries += 1
                time.sleep(RETRY_DELAY)

        if retries == MAX_RETRIES:
            logger.error(
                colored('Failed to fetch balance after %s retries. Skipping this run.', 'red'), MAX_RETRIES
            )
            return

        logger.info(colored('Checking for new buy/sell signals...', 'blue'))

        for coin in TRADABLE_SYMBOLS:
            try:
                signal = self._get_signal_for_symbol(coin)

                if signal == 'BUY':
                    self._open_trade(coin)
                elif signal == 'SELL':
                    self._close_trade(coin)
            except Exception as e:
                logger.error(colored('Error processing %s: %s', 'red'), coin, e)

    def _get_signal_for_symbol(self, symbol) -> Literal['BUY', 'SELL', 'HOLD']:
        # Set the `start` date to be 1 month ago.
        data = predict(
            CURRENT_MODEL,
            symbol=symbol,
            timeframes=TIMEFRAME,
            start=datetime.now() - timedelta(days=60),
            label_args=LABEL_ARGS,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=12,
        )
        # Check the last two predictions and send a signal if either of them is
        # a buy or sell
        last_pred = data['pred'][-1]

        if last_pred == 1:
            return 'BUY'
        elif last_pred == 2:
            return 'SELL'
        else:
            return 'HOLD'

    def _open_trade(self, symbol):
        if self._has_open_position(symbol):
            logger.info(colored('Already have an open position for %s', 'yellow'), symbol)
            return

        try:
            buy_amount_usd = BUY_ORDER_AMOUNT_USD / self._last_price(symbol)
            # open a buy market order
            order = ccxt_client.create_order(
                symbol=self._pair(symbol),
                type='market',
                side='buy',
                amount=buy_amount_usd,
            )
            logger.info(colored('Opened a BUY trade for %s with %.2f USD', 'green'), symbol, BUY_ORDER_AMOUNT_USD)
        except Exception as e:
            logger.error(colored('Error opening trade for %s: %s', 'red'), symbol, e)

    def _close_trade(self, symbol):
        if not self._has_open_position(symbol):
            logger.info(colored('No open position to close for %s', 'yellow'), symbol)
            return

        try:
            sell_amount = self._coin_bal(symbol, as_string=True)
            # sell market order
            order = ccxt_client.create_order(
                side='sell',
                type='market',
                symbol=self._pair(symbol),
                amount=sell_amount,
            )
            logger.info(
                colored('Closed a SELL trade for %s with %.2f USD', 'red'),
                symbol,
                float(sell_amount) * self._last_price(symbol),
            )
        except Exception as e:
            logger.error(colored('Error closing trade for %s: %s', 'red'), symbol, e)

    # Helpers

    def _close_all_open_trades(self):
        for coin in TRADABLE_SYMBOLS:
            if self._has_open_position(coin):
                self._close_trade(coin)

    def _has_open_position(self, symbol):
        return self._coin_bal_usd(symbol) > DUST_AMOUNT_USD

    def _coin_bal(self, symbol, as_string=False):
        bal_str = self._balance_info[symbol]['free']
        return bal_str if as_string else float(bal_str)

    def _coin_bal_usd(self, symbol):
        coin_bal = float(self._balance_info[symbol]['free'])
        return coin_bal * self._last_price(symbol)

    def _last_price(self, symbol):
        try:
            ticker = ccxt_client.fetch_ticker(self._pair(symbol))
            return float(ticker['last'])
        except Exception as e:
            logger.error('Error fetching last price for %s: %s', symbol, e)
            raise

    def _pair(self, pair):
        return pair.replace('USD', '') + 'USD'


if __name__ == '__main__':
    TradeEngine()
