import contextlib

import plotly.graph_objects as go
import vectorbtpro as vbt
from vctr.models.lstm.actions import predict
from vctr.trading.coins import tradable_coins


def get_red_green_shapes(data, pred_key='pred'):
    green_bars = data[data[pred_key] == 1]['close']
    red_bars = data[data[pred_key] == 2]['close']

    return (
        *(
            (
                dict(
                    type='rect',
                    xref='x',
                    yref='y domain',
                    x0=timestamp,
                    x1=timestamp,
                    y0=0,
                    y1=30000,
                    opacity=0.1,
                    line=dict(width=1, color='#30c77b'),
                ),
                price,
            )
            for timestamp, price in green_bars.items()
        ),
        *(
            (
                dict(
                    type='rect',
                    xref='x',
                    yref='y domain',
                    x0=timestamp,
                    x1=timestamp,
                    y0=0,
                    y1=30000,
                    opacity=0.1,
                    line=dict(width=1, color='#c43338'),
                ),
                price,
            )
            for timestamp, price in red_bars.items()
        ),
    )


def plot_price_preds(data, pred_key='pred'):
    shapes = get_red_green_shapes(data, pred_key)
    price_trace = go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price', line=dict(width=2))
    # Combine all traces and create a plot
    layout = go.Layout(
        shapes=shapes,
        title=dict(text='Price with Buy/Sell Predictions', font=dict(color='white')),
        xaxis=dict(
            title='Timestamp',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
        ),
        plot_bgcolor='rgba(32, 32, 32, 1)',
        paper_bgcolor='rgba(32, 32, 32, 1)',
        legend=dict(font=dict(color='white')),
        width=1200,
        height=700,
        yaxis_autorange=True,
    )

    fig = go.Figure(data=[price_trace], layout=layout)
    fig.update_layout(yaxis_autorange=True)
    fig.show()


def plot_price_preds_actual(data):
    # data = data[-2000:]

    # Get the max price for the y-axis
    max_price = data['close'].max()

    # Extract data for green and red bars
    green_bars_actual = data[data['label'] == 1]['close']
    red_bars_actual = data[data['label'] == 2]['close']

    green_bars = data[data['pred'] == 1]['close']
    red_bars = data[data['pred'] == 2]['close']

    # Create line plot for close prices
    price_trace = go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price', line=dict(width=1))

    # Create green and red vertical bar shapes
    green_shapes = [
        *(
            dict(
                type='rect',
                xref='x',
                yref='y',
                x0=timestamp,
                x1=timestamp,
                y0=0,
                y1=price,
                opacity=0.2,
                line=dict(width=3, color='#30c77b'),
            )
            for timestamp, price in green_bars.items()
        ),
        *(
            dict(
                type='rect',
                xref='x',
                yref='y',
                x0=timestamp,
                x1=timestamp,
                y0=price,
                y1=max_price * 2,
                opacity=0.2,
                line=dict(width=3, color='#30c77b'),
            )
            for timestamp, price in green_bars_actual.items()
        ),
    ]

    red_shapes = [
        *(
            dict(
                type='rect',
                xref='x',
                yref='y',
                x0=timestamp,
                x1=timestamp,
                y0=0,
                y1=price,
                opacity=0.2,
                line=dict(width=3, color='#c43338'),
            )
            for timestamp, price in red_bars.items()
        ),
        *(
            dict(
                type='rect',
                xref='x',
                yref='y',
                x0=timestamp,
                x1=timestamp,
                y0=price,
                y1=max_price * 2,
                opacity=0.2,
                line=dict(width=3, color='#c43338'),
            )
            for timestamp, price in red_bars_actual.items()
        ),
    ]

    # Combine all traces and create a plot
    layout = go.Layout(
        shapes=green_shapes + red_shapes,
        title=dict(text='Price with Buy/Sell Labels', font=dict(color='white')),
        xaxis=dict(
            title='Timestamp',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
        ),
        plot_bgcolor='rgba(32, 32, 32, 1)',
        paper_bgcolor='rgba(32, 32, 32, 1)',
        legend=dict(font=dict(color='white')),
        width=1200,
        height=700,
    )

    fig = go.Figure(data=[price_trace], layout=layout)
    fig.show()


def get_return(
    df,
    freq='30T',
    pred_key='pred',
    sell_signal=True,
    pf_params=dict(
        tsl_stop=0.005,
        sl_stop=0.05,
    ),
):
    pf = vbt.Portfolio.from_signals(
        df['close'],
        df[pred_key] == 1,
        df[pred_key] == 2 if sell_signal else None,
        freq=freq,
        init_cash=10000,
        fees=0.0006,
        slippage=0.001,
        log=True,
        **pf_params,
    )
    return pf


def print_returns(model, timeframe='1H', start=None, end=None, pred_key='pred'):
    returns = []
    for coin in tradable_coins:
        with contextlib.suppress(IndexError):
            data = predict(model, coin, timeframe, start=start, end=end)
            pf = vbt.Portfolio.from_signals(
                data['close'],
                data[pred_key] == 1,
                data[pred_key] == 2,
                freq=timeframe.replace('m', 'T').upper(),
                init_cash=10000,
                slippage=0.001,
                log=True,
            )
            coin_return = pf.stats('total_return').values[0]
            returns.append(coin_return)
            print(coin, f'{coin_return:.0f}%')
    print(f'Average ROI: {sum(returns) / len(returns):.0f}%')
