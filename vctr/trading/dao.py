import sqlite3
import json
from uuid import uuid4


class TradesDAO:
    def __init__(self, db_file='trades.db'):
        self.conn = sqlite3.connect(db_file)
        self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def create_trade(self, symbol, quantity, buy_price_usd, buy_order_ids):
        sql = 'INSERT INTO trades(id, symbol, quantity, buy_price_usd, buy_order_ids) VALUES (?, ?, ?, ?, ?)'
        vals = (str(uuid4()), symbol, quantity, buy_price_usd, json.dumps(buy_order_ids))
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Update Trade
    def update_trade(self, id, sell_price_usd, sell_order_ids):
        sql = 'UPDATE trades SET sell_price_usd=?, sell_order_ids=? WHERE id=?'
        vals = (sell_price_usd, sell_order_ids, id)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Close Trade
    def close_trade(self, id, closed_at):
        sql = 'UPDATE trades SET closed_at=? WHERE id=?'
        vals = (closed_at, id)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Get Open Trades
    def get_open_trades(self):
        sql = 'SELECT * FROM trades WHERE closed_at IS NULL'
        self.cur.execute(sql)
        return self.cur.fetchall()

    def symbol_has_open_trades(self, symbol):
        sql = 'SELECT * FROM trades WHERE closed_at IS NULL AND symbol=?'
        vals = (symbol,)
        self.cur.execute(sql, vals)
        return len(self.cur.fetchall()) > 0

    # Avg Trade Profit
    def avg_trade_profit(self):
        sql = 'SELECT AVG(sell_price_usd - buy_price_usd) FROM trades WHERE closed_at IS NOT NULL'
        self.cur.execute(sql)
        return self.cur.fetchone()[0]

    # Remove Order IDs
    def remove_order_ids(self, order_ids):
        sql = 'DELETE FROM trades WHERE buy_order_ids IN ? OR sell_order_ids IN ?'
        vals = (order_ids, order_ids)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Remove Buy Order IDs
    def remove_buy_order_ids(self, order_ids):
        sql = 'DELETE FROM trades WHERE buy_order_ids IN ?'
        vals = (order_ids,)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Remove Sell Order IDs
    def remove_sell_order_ids(self, order_ids):
        sql = 'DELETE FROM trades WHERE sell_order_ids IN ?'
        vals = (order_ids,)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Add Buy Order IDs
    def add_buy_order_ids(self, id, order_ids):
        sql = 'UPDATE trades SET buy_order_ids = JSON_SET(buy_order_ids, ?, ?) WHERE id = ?'
        vals = (order_ids, id)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Add Sell Order IDs
    def add_sell_order_ids(self, id, order_ids):
        sql = 'UPDATE trades SET sell_order_ids = JSON_SET(sell_order_ids, ?, ?) WHERE id = ?'
        vals = (order_ids, id)
        self.cur.execute(sql, vals)
        self.conn.commit()

    # Create the trades table
    def create_table(self):
        sql = """CREATE TABLE trades ( 
                id UUID PRIMARY KEY,
                symbol TEXT,
                quantity INTEGER,
                buy_price_usd REAL,
                buy_order_ids TEXT,
                opened_at TIMESTAMP,
                sell_price_usd REAL,
                sell_order_ids TEXT,
                closed_at TIMESTAMP
            );"""
        self.cur.execute(sql)
        self.conn.commit()
