import contextlib
import io
import pandas as pd
from dateutil import parser
from tabulate import tabulate
from termcolor import colored
from vctr.models.lstm.actions import predict
from vctr.trading.utils.misc import get_red_green_shapes, get_return
from tqdm import tqdm
from pqdm.threads import pqdm


def parallel_runner(
    model, symbol, timeframe, start, batch_size, label_args, pf_params, end=None, crypto=True, **kwargs
):
    data = predict_and_get_stats(
        model=model,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        batch_size=batch_size,
        label_args=label_args,
        pf_params=pf_params,
        end=end,
        crypto=crypto,
        **kwargs,
    )
    data['symbol'] = symbol
    return data


def predict_and_agg_stats(
    model=None,
    symbols=None,
    timeframe=None,
    start=None,
    label_args=None,
    batch_size=32,
    pf_params=None,
    end=None,
    crypto=True,
):
    args = [
        (model, symbol, timeframe, start, batch_size, label_args, pf_params, end, crypto) for symbol in symbols
    ]

    datasets = pqdm(args, parallel_runner, n_jobs=10, argument_type='args', tqdm_class=tqdm)
    data = pd.DataFrame(datasets)

    # Aggregate the stats, suppress warnings
    with contextlib.redirect_stderr(io.StringIO()):
        df = data.mean()

    df['model'] = model
    df['start'] = start
    df['label_args'] = label_args

    # Formatting and coloring
    colored_sortino = color_value(f'{df.sortino:.2f}', df.sortino < 4, df.sortino < 6)
    colored_sharpe = color_value(f'{df.sharpe:.2f}', df.sharpe < 0.5, df.sharpe < 1)

    formatted_return_pct = format_percentage(df.return_pct)
    colored_return_pct = color_value(
        formatted_return_pct,
        (df.return_pct < abs(df.bm_return_pct)) | (df.return_pct < 0),
        df.return_pct <= 1.25 * abs(df.bm_return_pct),
    )
    colored_annual_return_pct = color_value(
        format_percentage(df.annualized_return_pct),
        df.annualized_return_pct < 1,
        df.annualized_return_pct < 20,
    )

    formatted_win_rate = format_percentage(df.win_rate)
    colored_win_rate = color_value(formatted_win_rate, df.win_rate <= 0.6, df.win_rate <= 0.79)

    formatted_max_dd = format_percentage(df.max_dd)
    colored_max_dd = color_value(formatted_max_dd, df.max_dd < -0.2, df.max_dd < -0.12)

    if end is None:
        end = pd.Timestamp.now()
    else:
        end = parser.parse(end)

    if start is None:
        start = end - pd.Timedelta(days=365)
    else:
        start = parser.parse(start)

    # Tabulate data
    table = [
        ['Duration', format_duration(end - start)],
        ['Total Return', colored_return_pct],
        ['Benchmark Return', format_percentage(df.bm_return_pct)],
        ['-------------------------', '-------'],
        ['Annualized Return', colored_annual_return_pct],
        ['Sortino', colored_sortino],
        ['Sharpe', colored_sharpe],
        ['-------------------------', '-------'],
        ['Number of Trades', df.num_trades],
        ['Average Trade Duration', format_duration(df.avg_trade_duration)],
        ['Win Rate', colored_win_rate],
        ['-------------------------', '-------'],
        ['Max Drawdown', colored_max_dd],
        ['Average Drawdown Duration', format_duration(df.avg_dd_duration)],
        ['Max Drawdown Duration', format_duration(df.max_dd_duration)],
    ]

    return tabulate(table)


def predict_and_get_stats(
    model,
    symbol,
    timeframe,
    start,
    label_args,
    batch_size,
    pf_params,
    end=None,
    crypto=True,
):
    data = predict(
        model,
        symbol=symbol,
        timeframe=timeframe,
        batch_size=batch_size,
        start=start,
        end=end,
        label_args=label_args,
        crypto=crypto,
    )
    pf = get_return(data, '1h', 'pred', sell_signal=True, pf_params=pf_params)

    data = {
        'return_pct': pf.total_return,
        'bm_return_pct': pf.total_market_return,
        'annualized_return_pct': pf.returns_acc.annualized(),
        'sortino': pf.sortino_ratio,
        'sharpe': pf.sharpe_ratio,
        'num_trades': len(pf.trades),
        'avg_trade_duration': pf.trades.avg_duration,
        'win_rate': pf.trades.win_rate,
        'avg_dd_duration': pf.drawdowns.avg_duration,
        'max_dd_duration': pf.drawdowns.max_duration,
        'max_dd': pf.max_drawdown,
    }

    return data


def format_duration(duration):
    if duration < pd.Timedelta(days=1):
        return f'{duration.seconds // 3600} hours'
    elif duration < pd.Timedelta(days=30):
        return f'{duration.days} days'
    elif duration < pd.Timedelta(days=90):
        return f'{duration.days // 7} weeks'
    else:
        return f'{duration.days // 30} months'


def format_percentage(percentage):
    return f'{percentage * 100:,.0f}%'


def color_value(value, red_cond, yellow_cond):
    if red_cond:
        return colored(value, 'red')
    elif yellow_cond:
        return colored(value, 'yellow')
    else:
        return colored(value, 'green')


def print_stats(pf):
    # Data
    return_pct = pf.total_return
    annualized_return_pct = pf.returns_acc.annualized()
    bm_return_pct = pf.total_market_return

    num_trades = len(pf.trades)
    avg_trade_duration = pf.trades.avg_duration
    win_rate = pf.trades.win_rate

    avg_dd_duration = pf.drawdowns.avg_duration
    max_dd_duration = pf.drawdowns.max_duration
    max_dd = pf.max_drawdown

    # Formatting and coloring
    colored_sortino = color_value(f'{pf.sortino_ratio:.2f}', pf.sortino_ratio < 4, pf.sortino_ratio < 6)
    colored_sharpe = color_value(f'{pf.sharpe_ratio:.2f}', pf.sharpe_ratio < 0.5, pf.sharpe_ratio < 1)

    formatted_return_pct = format_percentage(return_pct)
    colored_return_pct = color_value(
        formatted_return_pct,
        (return_pct < abs(bm_return_pct)) | (return_pct < 0),
        return_pct <= 1.25 * abs(bm_return_pct),
    )
    colored_annual_return_pct = color_value(
        format_percentage(annualized_return_pct),
        annualized_return_pct < 1,
        annualized_return_pct < 20,
    )

    formatted_win_rate = format_percentage(win_rate)
    colored_win_rate = color_value(formatted_win_rate, win_rate <= 0.6, win_rate <= 0.79)

    formatted_max_dd = format_percentage(max_dd)
    colored_max_dd = color_value(formatted_max_dd, max_dd < -0.2, max_dd < -0.15)

    # Tabulate data
    table = [
        ['Duration', format_duration(pf.wrapper.index[-1] - pf.wrapper.index[0])],
        ['Total Return', colored_return_pct],
        ['Benchmark Return', format_percentage(bm_return_pct)],
        ['-------------------------', '-------'],
        ['Annualized Return', colored_annual_return_pct],
        ['Sortino', colored_sortino],
        ['Sharpe', colored_sharpe],
        ['-------------------------', '-------'],
        ['Number of Trades', num_trades],
        ['Average Trade Duration', format_duration(avg_trade_duration)],
        ['Win Rate', colored_win_rate],
        ['-------------------------', '-------'],
        ['Max Drawdown', colored_max_dd],
        ['Average Drawdown Duration', format_duration(avg_dd_duration)],
        ['Max Drawdown Duration', format_duration(max_dd_duration)],
    ]

    print(tabulate(table))


def add_shape(shape, price, min_max):
    shape.update(
        {
            'opacity': 0.35,
            'yref': 'y',
            'y0': min_max,
            'y1': price,
            'line': {'width': 1, **shape['line']},
        }
    )
    return shape


def get_plot_and_pf(data, key='pred', pf_params=None):
    if pf_params is None:
        pf_params = {}
    pf = get_return(data, '1h', key, sell_signal=True, pf_params=pf_params)

    pf.trades.plot


def add_shapes(fig, inputs, key):
    buy_mask = inputs[key] == 1
    buy_mask.vbt.signals.ranges.plot_shapes(
        plot_close=False, fig=fig, shape_kwargs=dict(fillcolor='#30c77b', opacity=0.28)
    )
    sell = inputs[key] == 2
    sell.vbt.signals.ranges.plot_shapes(
        plot_close=False, fig=fig, shape_kwargs=dict(fillcolor='#c43338', opacity=0.28)
    )


def get_plot_and_pf(data, key='pred', pf_params=None, plot_label=False, plot_pred=False, width=1200, height=500):
    if pf_params is None:
        pf_params = {}
    pf = get_return(data, '1h', key, sell_signal=True, pf_params=pf_params)

    # fig = data.vbt.ohlcv.plot(plot_volume=False)
    # fig.data[0].opacity = 0.5
    # fig.data[0].increasing.fillcolor = '#0ead69'
    # fig.data[0].increasing.line.color = '#0ead69'
    # fig.data[0].decreasing.fillcolor = '#ee4266'
    # fig.data[0].decreasing.line.color = '#ee4266'
    fig = pf.trades.plot()

    axis_args = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.05)', griddash='dot')
    fig.update_xaxes(**axis_args)
    fig.update_yaxes(**axis_args)

    fig.update_layout(dict(width=width, height=height))

    if plot_label:
        add_shapes(fig, data, 'label')
    if plot_pred:
        add_shapes(fig, data, 'pred')

    return fig, pf
