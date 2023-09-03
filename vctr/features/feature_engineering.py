import contextlib
import hashlib
import io

import numpy as np
import pandas as pd
import pandas_ta as ta
import pywt
import requests
import talib
import vectorbtpro as vbt
from pyemd.EMD import EMD
from vctr.features.correlated_feats import correlated_feats
from vctr.models.lstm.defaults import SEQUENCE_LENGTH


def wavelet_transform(data, wavelet='db4', level=None, mode='zero'):
    if level is None:
        level = pywt.dwt_max_level(data_len=len(data), filter_len=pywt.Wavelet(wavelet).dec_len)

    coeff = pywt.wavedec(data, wavelet=wavelet, level=level, mode=mode)
    return coeff


def add_multiscale_features(data, columns_to_transform, wavelet='db4', level=None, mode='zero'):
    # Apply wavelet transform to each column and store the results in a dictionary
    transformed_data = {}
    for column in columns_to_transform:
        transformed_data[column] = wavelet_transform(data[column], wavelet, level, mode)

    # Create a new dataframe to hold the transformed data
    multiscale_data = pd.DataFrame()

    # Combine the transformed data into the new dataframe
    for column, coeff in transformed_data.items():
        for i, details in enumerate(coeff):
            multiscale_data[f'{column}_scale_{i}'] = details

    # You may need to handle NaN values that may appear after wavelet transformation
    multiscale_data = multiscale_data.dropna()

    return multiscale_data


def add_emd_features(df: pd.DataFrame, column: str, n_imfs: int) -> pd.DataFrame:
    """
    Adds EMD features to the given dataframe with a consistent number of IMFs.

    :param df: A pandas DataFrame containing OHLCV data.
    :param column: The column name for which EMD features should be computed.
    :param n_imfs: The fixed number of IMFs to include in the output.
    :return: DataFrame with EMD features added.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe.")

    # Perform EMD on the specified column
    emd = EMD()
    imfs = emd(df[column].values)

    # Ensure a consistent number of IMFs
    if len(imfs) < n_imfs:
        padding = np.zeros((n_imfs - len(imfs), len(df[column])))
        imfs = np.vstack((imfs, padding))
    elif len(imfs) > n_imfs:
        # Combine extra IMFs or discard them
        imfs = imfs[:n_imfs]

    # Add IMFs to the dataframe
    for i, imf in enumerate(imfs):
        df[f'{column}_IMF{i+1}'] = imf

    return df


def calc_smma(src, length):
    smma = np.full_like(src, np.nan)
    for i in range(length, len(src)):
        if np.isnan(smma[i - 1]):
            smma[i] = talib.SMA(src[: i + 1], timeperiod=length)[-1]
        else:
            smma[i] = (smma[i - 1] * (length - 1) + src[i]) / length
    return smma


def calc_zlema(src, length):
    ema1 = talib.EMA(src, timeperiod=length)
    ema2 = talib.EMA(ema1, timeperiod=length)
    d = ema1 - ema2
    return ema1 + d


def calc_impulse_macd(high, low, close, period_ma=34, period_signal=9):
    src = (high + low + close) / 3
    hi = calc_smma(high, period_ma)
    lo = calc_smma(low, period_ma)
    mi = calc_zlema(src, period_ma)

    imacd = pd.Series(np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0)))
    imacd_signal = pd.Series(talib.SMA(imacd, timeperiod=period_signal))
    imacd_histo = pd.Series(imacd - imacd_signal)

    return imacd, imacd_histo, imacd_signal


def add_imacd(df: pd.DataFrame, period_ma: int = 34, period_signal: int = 9) -> pd.DataFrame:
    # sourcery skip: avoid-builtin-shadow
    high, low, close = df['high'], df['low'], df['close']
    imacd, imacd_histo, imacd_signal = calc_impulse_macd(high, low, close, period_ma, period_signal)

    id = f'imacd_{period_ma}_{period_signal}'

    imacd.index = df.index
    imacd_histo.index = df.index
    imacd_signal.index = df.index

    df[f'{id}_abv_sig'] = np.where(imacd > imacd_signal, 1, 0)
    df[f'{id}_histo_abv_0'] = np.where(imacd_histo > 0, 1, 0)
    df[f'{id}_abv_2'] = np.where(imacd > 2, 1, 0)
    df[f'{id}_und_2'] = np.where(imacd < 2, 1, 0)

    return df


def hash_df(args, _=None):
    df = args[0]
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
    # we build a tuple, and not a list, as a cache key
    # because tuples are immutable, and thus hashable
    return df_hash


def calc_ma(type, series, length):
    ma = ta.ma(type, series, length=length)
    # Calculate the expanding mean (simple average) up to each point in the 'close' series
    expanding_mean = series.expanding().mean()
    # Replace the NaN values in the SMA with the corresponding expanding mean values
    if ma is not None:
        ma = ma.combine_first(expanding_mean)
    else:
        ma = pd.Series(expanding_mean)
    ma.index = series.index

    return ma


def add_bband_features(df, period, std):
    try:
        bb_df = ta.bbands(df['close'], length=period, std=std)
        above_upper_band = (
            df[['open', 'high', 'low', 'close']].ge(bb_df[f'BBU_{period}_{std}.0'], axis=0).any(axis=1)
        )
        df[f'prc_abv_bbu_{period}_{std}'] = np.where(above_upper_band, 1, 0)

        below_lower_band = (
            df[['open', 'high', 'low', 'close']].le(bb_df[f'BBL_{period}_{std}.0'], axis=0).any(axis=1)
        )
        df[f'prc_und_bbl_{period}_{std}'] = np.where(below_lower_band, 1, 0)

        return df
    except:
        df[f'prc_abv_bbu_{period}_{std}'] = 0
        df[f'prc_und_bbl_{period}_{std}'] = 0
        return df


def add_rsi_features(df, rsi_period, ema_period, lower_thresh=30, upper_thresh=70):
    rsi = ta.rsi(df['close'], length=rsi_period)
    rsi_ema = calc_ma('ema', rsi, ema_period)

    rsi_id = f'rsi{rsi_period}'
    ema_id = f'{rsi_id}_ema{ema_period}'

    rsi_abv_upper = pd.Series(np.where(rsi > upper_thresh, 1, 0), index=df.index)
    rsi_und_lower = pd.Series(np.where(rsi < lower_thresh, 1, 0), index=df.index)
    ema_abv_upper = pd.Series(np.where(rsi_ema > upper_thresh, 1, 0), index=df.index)
    ema_und_lower = pd.Series(np.where(rsi_ema < lower_thresh, 1, 0), index=df.index)

    df[f'{rsi_id}_cross_{upper_thresh}'] = rsi_abv_upper.diff().apply(lambda x: 1 if x == 1 else 0)
    df[f'{rsi_id}_cross_{lower_thresh}'] = rsi_und_lower.diff().apply(lambda x: 1 if x == 1 else 0)
    df[f'{ema_id}_cross_{upper_thresh}'] = ema_abv_upper.diff().apply(lambda x: 1 if x == 1 else 0)
    df[f'{ema_id}_cross_{lower_thresh}'] = ema_und_lower.diff().apply(lambda x: 1 if x == 1 else 0)

    return df


def add_stationary_features(df):
    price_keys = ['open', 'high', 'low', 'close']
    for key in price_keys:
        key_log_diff = pd.Series(np.log(df[key]).diff(), name=f'{key}_log_diff', index=df.index)
    return pd.concat([df, key_log_diff], axis=1)


def add_hist_comp_features(df, period=24):
    col_name = f'prc_abv_{period}_prd'
    # Check whether the current price is above or below the price from `period`
    # periods ago.
    df[col_name] = np.where(df['close'] > df['close'].shift(period), 1, 0)

    return df


def add_n_pct_from_prev_minima(data: pd.DataFrame, percentage: int, window: int) -> pd.DataFrame:
    minima = data.iloc[:, 0].rolling(window=window).min()
    n_pct_above_minima = data.iloc[:, 0] > (1 + percentage / 100) * minima.shift(1)
    data[f'{percentage}_pct_above_{window}_pd_minima'] = n_pct_above_minima.astype(int)
    return data


def add_last_n_bars_color(df, n, color):
    # Define the name of the new column
    new_column_name = f'last_{n}_bars_{color}'

    # If color is not red or green, raise an error
    if color not in ['red', 'green']:
        raise ValueError("Color must be 'red' or 'green'.")

    # Create a new column with default value as False
    df[new_column_name] = False

    # Get the close and open prices for the last n bars using numpy arrays
    close_prices = df['close'].to_numpy()
    open_prices = df['open'].to_numpy()
    last_n_bars_close = np.lib.stride_tricks.sliding_window_view(close_prices, n)
    last_n_bars_open = np.lib.stride_tricks.sliding_window_view(open_prices, n)

    with contextlib.redirect_stderr(io.StringIO()):
        # Check if the close price is less than the open price for all bars in last_n_bars
        if color == 'red':
            all_less = np.all(last_n_bars_close < last_n_bars_open, axis=1)
            df[new_column_name].iloc[n - 1 :] = all_less
        # Check if the close price is greater than or equal to the open price for all bars in last_n_bars
        elif color == 'green':
            all_greater = np.all(last_n_bars_close >= last_n_bars_open, axis=1)
            df[new_column_name].iloc[n - 1 :] = all_greater

        # Cast the column to int
        df[new_column_name] = df[new_column_name].astype(int)

        return df


def add_trend_magic(df, params):
    period = params['period']
    coeff = params['coeff']
    atr_period = params['atr_period']
    suffix = f'_p{period}_c{coeff}_atr{atr_period}'

    src = df['close']
    high = df['high']
    low = df['low']

    atr = talib.SMA(talib.TRANGE(high, low, src), atr_period)
    up_t = low - atr * coeff
    down_t = high + atr * coeff
    magic_trend = np.zeros_like(src)
    tm_signal_up = np.zeros_like(src)
    tm_signal_dn = np.zeros_like(src)

    cci_values = talib.CCI(high, low, src, period)

    magic_trend[cci_values >= 0] = np.minimum(up_t[cci_values >= 0], magic_trend[cci_values >= 0])
    magic_trend[cci_values < 0] = np.maximum(down_t[cci_values < 0], magic_trend[cci_values < 0])

    tm_signal_up[cci_values > 0] = 1
    tm_signal_dn[cci_values < 0] = 1

    df[f'tm_signal{suffix}'] = magic_trend
    df[f'tm_signal_up{suffix}'] = tm_signal_up
    df[f'tm_signal_dn{suffix}'] = tm_signal_dn

    return df


def add_supertrend(df, params):
    atr_period = params['atr_period']
    atr_multiplier = params['atr_multiplier']
    change_atr = params['change_atr']
    suffix = f'_atrP{atr_period}_atrM{atr_multiplier}_cAtr{change_atr}'

    hl2 = (df['high'] + df['low']) / 2
    atr = (
        talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        if change_atr
        else talib.TRANGE(df['high'], df['low'], df['close']).rolling(window=atr_period).mean()
    )

    up = hl2 - (atr_multiplier * atr)
    dn = hl2 + (atr_multiplier * atr)

    up1 = up.shift(1).fillna(up)
    dn1 = dn.shift(1).fillna(dn)

    df[f'st_up{suffix}'] = np.where(df['close'].shift(1) > up1, np.maximum(up, up1), up)
    df[f'st_dn{suffix}'] = np.where(df['close'].shift(1) < dn1, np.minimum(dn, dn1), dn)

    df[f'st_trend{suffix}'] = np.where(
        df['close'] > df[f'st_dn{suffix}'].shift(1),
        1,
        np.where(df['close'] < df[f'st_up{suffix}'].shift(1), -1, np.nan),
    )
    df[f'st_trend{suffix}'].fillna(method='ffill', inplace=True)

    df[f'st_buy{suffix}'] = (df[f'st_trend{suffix}'] == 1) & (df[f'st_trend{suffix}'].shift(1) == -1).astype(int)
    df[f'st_sell{suffix}'] = (df[f'st_trend{suffix}'] == -1) & (df[f'st_trend{suffix}'].shift(1) == 1).astype(int)

    return df


def add_wave_trend(df, params):
    n1 = params['n1']
    n2 = params['n2']
    obLevel1 = params['obLevel1']
    obLevel2 = params['obLevel2']
    osLevel1 = params['osLevel1']
    osLevel2 = params['osLevel2']
    suffix = f'_n1{n1}_n2{n2}_obL1{obLevel1}_obL2{obLevel2}_osL1{osLevel1}_osL2{osLevel2}'

    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    esa = talib.EMA(hlc3, n1)
    d = talib.EMA(np.abs(hlc3 - esa), n1)
    ci = (hlc3 - esa) / (0.015 * d)
    tci = talib.EMA(ci, n2)

    wt1 = tci
    wt2 = talib.SMA(wt1, 4)

    wt2_cross_und_wt1 = np.where(
        (wt2.shift(1) >= wt1.shift(1)) & (wt2 < wt1) & (wt1 < osLevel1) & (wt2 < osLevel1), 1, 0
    )
    wt2_cross_abv_wt1 = np.where((wt2.shift(1) <= wt1.shift(1)) & (wt2 > wt1) & (wt1 > 0) & (wt2 > 0), 1, 0)

    df[f'wt1{suffix}'] = wt1
    df[f'wt2{suffix}'] = wt2
    df[f'wt2_cross_und_wt1{suffix}'] = wt2_cross_und_wt1
    df[f'wt2_cross_abv_wt1{suffix}'] = wt2_cross_abv_wt1

    return df


def add_ma_features(df, short_period=1, long_period=5, ma_type_short='sma', ma_type_long='swma'):
    if short_period > long_period:
        short_period, long_period = long_period, short_period

    close = df['close']
    short_id = f'{ma_type_short}{short_period}'
    long_id = f'{ma_type_long}{long_period}'

    short_ma = calc_ma(ma_type_short, close, short_period)
    long_ma = calc_ma(ma_type_long, close, long_period)

    short_abv_long = pd.Series(
        np.where(short_ma > long_ma, 1, 0), index=close.index, name=f'{short_id}_abv_{long_id}'
    )

    # Create a new series to record cross over/under
    cross_over_under = short_abv_long.diff()
    cross_over_under = cross_over_under.apply(lambda x: 1 if abs(x) == 1 else 0)

    cross_over_under.name = f'{short_id}_cross_{long_id}'

    return pd.concat([df, cross_over_under], axis=1)


def add_higher_highs(df, period_length, num_periods):
    col_name = f'higher_highs_{period_length}_{num_periods}'
    highs = df['high'].rolling(window=num_periods, min_periods=num_periods).max()
    prev_highs = df['high'].shift(period_length * (num_periods - 1))
    is_higher_high = highs > prev_highs
    df[col_name] = is_higher_high.astype(int)
    return df


def add_lower_lows(df, period_length, num_periods):
    col_name = f'lower_lows_{period_length}_{num_periods}'
    lows = df['low'].rolling(window=num_periods, min_periods=num_periods).min()
    prev_lows = df['low'].shift(period_length * (num_periods - 1))
    is_lower_low = lows < prev_lows
    df[col_name] = is_lower_low.astype(int)
    return df


def add_williams_vix_fix(df, params):
    pd, bbl, mult, lb, ph, pl = (
        params['pd'],
        params['bbl'],
        params['mult'],
        params['lb'],
        params['ph'],
        params['pl'],
    )
    suffix = f'_pd{pd}_bbl{bbl}_mult{mult}_lb{lb}_ph{ph}_pl{pl}'

    wvf = ((df['close'].rolling(window=pd).max() - df['low']) / df['close'].rolling(window=pd).max()) * 100

    s_dev = mult * wvf.rolling(window=bbl).std()
    mid_line = wvf.rolling(window=bbl).mean()
    wvf_lower_band = mid_line - s_dev
    wvf_upper_band = mid_line + s_dev

    wvf_range_high = wvf.rolling(window=lb).max() * ph
    wvf_range_low = wvf.rolling(window=lb).min() * pl

    with contextlib.redirect_stderr(io.StringIO()):
        df[f'wvf{suffix}'] = wvf
        df[f'wvf_lower_band{suffix}'] = wvf_lower_band
        df[f'wvf_upper_band{suffix}'] = wvf_upper_band
        df[f'wvf_range_high{suffix}'] = wvf_range_high
        df[f'wvf_range_low{suffix}'] = wvf_range_low

        df[f'wvf_abv_th{suffix}'] = np.where(
            (df[f'wvf{suffix}'] > df[f'wvf_upper_band{suffix}'])
            | (df[f'wvf{suffix}'] > df[f'wvf_range_high{suffix}']),
            1,
            0,
        )

    return df


def add_squeeze_mom(df, params):
    kc_length = params['kc_length']
    suffix = f'_kc{kc_length}'

    val = talib.LINEARREG(
        df['close']
        - (
            (df['high'].rolling(window=kc_length).max() + df['low'].rolling(window=kc_length).min()) / 2
            + talib.SMA(df['close'], timeperiod=kc_length)
        )
        / 2,
        timeperiod=kc_length,
    )

    df[f'squeeze_up{suffix}'] = np.where((val > 0) & (val > val.shift(1)), 1, 0)
    df[f'squeeze_dn{suffix}'] = np.where((val <= 0) & (val >= val.shift(1)), 1, 0)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop([col for col in correlated_feats if col in df.columns], axis=1)

    # Remove columns containing only NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all')

    # Impute NaN or infinite values with column mean
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

    for col in ['real_gdp', 'inflation', 'federal_funds_rate']:
        df[col] = df[col].interpolate(method='linear') if col in df.columns else 0

    return df.astype('float32')


def add_vbt_indicators(df):
    with contextlib.redirect_stderr(io.StringIO()):
        features_talib = df.vc.run('talib_all', periods=vbt.run_func_dict(talib_mavp=14))
        features_ta = df.vc.run('ta_all', window=vbt.run_func_dict(ta_smaindicator=40))
        features_wqa101 = df.vc.run('wqa101_all')

        return df.join(features_talib).join(features_ta).join(features_wqa101)


def add_sigdet(df, params):
    bb_factor = params['bb_factor']
    macd_fastperiod = params['macd_fastperiod']
    macd_slowperiod = params['macd_slowperiod']
    macd_signalperiod = params['macd_signalperiod']
    suffix = f'_bbf{bb_factor}_macdf{macd_fastperiod}_macds{macd_slowperiod}_macdsp{macd_signalperiod}'

    sd_bb = vbt.SIGDET.run(df.vc.run('bbands').bandwidth, factor=bb_factor)

    df[f'sd_bb_dn{suffix}'] = (sd_bb.signal == -1).astype(int)
    df[f'sd_bb_up{suffix}'] = (sd_bb.signal == 1).astype(int)

    sd_macd = vbt.SIGDET.run(
        vbt.talib('MACD')
        .run(df.close, fastperiod=macd_fastperiod, slowperiod=macd_slowperiod, signalperiod=macd_signalperiod)
        .macd,
        factor=bb_factor,
    )

    df[f'sd_macd_dn{suffix}'] = (sd_macd.signal == -1).astype(int)
    df[f'sd_macd_up{suffix}'] = (sd_macd.signal == 1).astype(int)

    return df


def rsi_swing_indicator(
    df: pd.DataFrame, rsi_length: int = 12, rsi_overbought: int = 80, rsi_oversold: int = 20
) -> pd.DataFrame:
    def moveOverBoughtLabel(label_hh, line_up, i):
        label_hh[i] = label_hh[i - 1]
        line_up[i] = line_up[i - 1]

    def moveOversoldLabel(label_ll, line_down, i):
        label_ll[i] = label_ll[i - 1]
        line_down[i] = line_down[i - 1]

    rsi_values = talib.RSI(df['close'], timeperiod=rsi_length)
    is_overbought = rsi_values >= rsi_overbought
    is_oversold = rsi_values <= rsi_oversold

    # State of the last extreme 0 for initialization, 1 = overbought, 2 = oversold
    laststate = 0

    # Highest and Lowest prices since the last state change
    hh = df['low']
    ll = df['high']

    # Labels
    label_ll = np.full_like(df['close'], np.nan)
    label_hh = np.full_like(df['close'], np.nan)

    # Swing lines
    line_up = np.full_like(df['close'], np.nan)
    line_down = np.full_like(df['close'], np.nan)

    for i in range(1, len(df)):
        # We go from oversold straight to overbought NEW DRAWINGS CREATED HERE
        if laststate == 2 and is_overbought[i]:
            hh[i] = df['high'][i]
            label_hh[i] = df['high'][i]
            label_ll[i - 1]
            i - 1
            label_ll[i - 1]
            line_up[i] = df['high'][i]

        # We go from overbought straight to oversold  NEW DRAWINGS CREATED HERE
        if laststate == 1 and is_oversold[i]:
            ll[i] = df['low'][i]
            label_ll[i] = df['low'][i]
            label_hh[i - 1]
            i - 1
            label_hh[i - 1]
            line_down[i] = df['low'][i]

        # If we are overbought
        if is_overbought[i]:
            if df['high'][i] >= hh[i - 1]:
                hh[i] = df['high'][i]
                label_hh[i] = label_hh[i - 1]
                moveOverBoughtLabel(label_hh, line_up, i)
            laststate = 1

        # If we are oversold
        if is_oversold[i] and df['low'][i] <= ll[i - 1]:
            ll[i] = df['low'][i]
            label_ll[i] = label_ll[i - 1]
            moveOversoldLabel(label_ll, line_down, i)

        # If last state was overbought
        if laststate == 1:
            if hh[i] <= df['high'][i]:
                hh[i] = df['high'][i]
                moveOverBoughtLabel(label_hh, line_up, i)

        elif laststate == 2:
            if ll[i] >= df['low'][i]:
                ll[i] = df['low'][i]
                moveOversoldLabel(label_ll, line_down, i)

    # create columns for signals
    df['rsi_swing_sell'] = is_overbought.astype(int)
    df['rsi_swing_buy'] = is_oversold.astype(int)

    return df


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    close = df['close']
    rsi = talib.RSI(close, timeperiod=period)

    col_id = f'rsi_{period}'
    rsi.index = df.index

    df[f'{col_id}_abv_70'] = np.where(rsi > 70, 1, 0)
    df[f'{col_id}_bel_30'] = np.where(rsi < 30, 1, 0)

    return df


def add_macd(df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
    close = df['close']
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period
    )

    col_id = f'macd_{fast_period}_{slow_period}_{signal_period}'
    macd.index = df.index
    macd_signal.index = df.index
    macd_hist.index = df.index

    df[f'{col_id}_abv_sig'] = np.where(macd > macd_signal, 1, 0)
    df[f'{col_id}_hist_abv_0'] = np.where(macd_hist > 0, 1, 0)

    return df


def add_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    high, low, close = df['high'], df['low'], df['close']
    adx = talib.ADX(high, low, close, timeperiod=period)

    col_id = f'adx_{period}'
    adx.index = df.index

    df[f'{col_id}_abv_25'] = np.where(adx > 25, 1, 0)

    return df


def add_bollinger_bands(df: pd.DataFrame, period: int, std_dev_multiplier: float) -> pd.DataFrame:
    close = df['close']
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=period, nbdevup=std_dev_multiplier, nbdevdn=std_dev_multiplier
    )

    col_id = f'bb_{period}_{std_dev_multiplier}'
    upper.index = df.index
    middle.index = df.index
    lower.index = df.index

    df[f'{col_id}_abv_up'] = np.where(close > upper, 1, 0)
    df[f'{col_id}_bel_low'] = np.where(close < lower, 1, 0)

    return df


def add_obv(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    close, volume = df['close'], df['volume']
    obv = talib.OBV(close, volume)

    col_id = f'obv_{threshold}'
    obv.index = df.index

    df[f'{col_id}_abv_thresh'] = np.where(obv > threshold, 1, 0)
    df[f'{col_id}_bel_thresh'] = np.where(obv < threshold, 1, 0)

    return df


def compute_Hc(data, kind='price', simplified=True):
    """Compute the Hurst exponent of data."""
    # Convert data to numpy array for generality
    data = np.asarray(data)

    # Ensure data is a price series
    if kind == 'price':
        data = np.diff(np.log(data))

    # Create a range of lag values
    lags = range(2, 20)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0, poly[1]


def compute_Hc_rolling(series, window):
    """Compute Hurst exponent of a rolling window of a series."""
    return series.rolling(window).apply(lambda x: compute_Hc(x)[0], raw=True), series.rolling(window).apply(
        lambda x: compute_Hc(x)[1], raw=True
    )


def get_economic_indicator(indicator):
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function={indicator}&interval=daily&apikey=TMP0580D0ZSFILSH'
    r = requests.get(url)
    return r.json()['data']


def add_data_column(dataframe, economic_data, col_name):
    economic_data_df = pd.DataFrame(economic_data)
    economic_data_df['date'] = pd.to_datetime(economic_data_df['date'])
    economic_data_df['value'] = pd.to_numeric(economic_data_df['value'])
    economic_data_df.set_index('date', inplace=True)
    economic_data_df.index = economic_data_df.index.tz_localize('UTC')
    if dataframe.index.tz is None or dataframe.index.tz.zone != 'UTC':
        dataframe = dataframe.tz_convert('UTC')
    frequency = dataframe.index.to_series().diff().mode().iloc[0]
    dataframe = dataframe.loc[~dataframe.index.duplicated(keep='first')]
    economic_data_df = economic_data_df.loc[~economic_data_df.index.duplicated(keep='first')]
    resampled_economic_data = economic_data_df.resample(frequency).last().ffill()
    merged_df = dataframe.merge(resampled_economic_data, how='left', left_index=True, right_index=True)
    merged_df['value'] = merged_df['value'].fillna(method='ffill')
    merged_df.rename(columns={'value': col_name}, inplace=True)

    return merged_df


def add_features(df, timeframe=None):  # sourcery skip: low-code-quality
    with contextlib.redirect_stderr(io.StringIO()):
        df = add_vbt_indicators(df)

        rsi_config = [
            (2, 2, 30, 70),
        ]

        for rsi_params in rsi_config:
            df = add_rsi_features(df, *rsi_params)

        # bband_config = [
        #     (15, 2),
        #     (30, 2),
        #     (45, 2),
        #     (60, 2),
        #     (12, 2),
        #     (8, 2),
        #     (5, 2),
        # ]

        # for bband_params in bband_config:
        #     df = add_bband_features(df, *bband_params)

        # Modify extrema configuration for 15m-1h charts
        extrema_config = [
            (1, 200),  # 15-min chart
        ]

        for extrema_params in extrema_config:
            df = add_n_pct_from_prev_minima(df, *extrema_params)

        imacd_config = [
            (4, 2),
        ]

        for imacd_params in imacd_config:
            df = add_imacd(df, *imacd_params)

        ma_config = [
            (2, 100, 'vidya', 't3'),
            (2, 50, 'dema', 'dema'),
        ]

        for ma_params in ma_config:
            df = add_ma_features(df, *ma_params)

        last_n_color_config = [
            2,
            6,
            10,
        ]
        for color in ['green', 'red']:
            for n in last_n_color_config:
                df = add_last_n_bars_color(df, n, color)

        # Add MACD features
        macd_config = [
            (12, 26, 9),
            (10, 20, 7),
            (15, 30, 10),
        ]
        for macd_params in macd_config:
            df = add_macd(df, *macd_params)

        # Add ADX features
        adx_config = [
            7,
            14,
            28,
        ]
        for period in adx_config:
            df = add_adx(df, period)

        # Add Bollinger Bands features
        bollinger_bands_config = [
            (20, 2),
            (14, 2),
            (30, 1.5),
        ]
        for bb_params in bollinger_bands_config:
            df = add_bollinger_bands(df, *bb_params)

        # Add OBV features
        obv_config = [
            1000000,
            5000000,
            2000000,
            7500000,
            2500000,
        ]
        for threshold in obv_config:
            df = add_obv(df, threshold)

        params_list = [
            {'pd': 14, 'bbl': 11, 'mult': 1.5, 'lb': 40, 'ph': 0.75, 'pl': 1.00},
            {'pd': 22, 'bbl': 20, 'mult': 2.0, 'lb': 50, 'ph': 0.85, 'pl': 1.01},
            {'pd': 30, 'bbl': 25, 'mult': 1.5, 'lb': 60, 'ph': 0.9, 'pl': 1.02},
            {'pd': 28, 'bbl': 22, 'mult': 2.5, 'lb': 55, 'ph': 0.87, 'pl': 1.03},
            {'pd': 25, 'bbl': 18, 'mult': 2.2, 'lb': 45, 'ph': 0.83, 'pl': 1.00},
            {'pd': 35, 'bbl': 30, 'mult': 1.7, 'lb': 65, 'ph': 0.95, 'pl': 1.04},
            {'pd': 40, 'bbl': 35, 'mult': 2.3, 'lb': 70, 'ph': 0.88, 'pl': 1.05},
            {'pd': 45, 'bbl': 40, 'mult': 1.8, 'lb': 75, 'ph': 0.92, 'pl': 1.06},
        ]
        for params in params_list:
            df = add_williams_vix_fix(df, params)

        params_list = [
            {'kc_length': 40},
            {'kc_length': 50},
            {'kc_length': 60},
            {'kc_length': 70},
        ]

        for params in params_list:
            df = add_squeeze_mom(df, params)

        params_list = [
            {'bb_factor': 2, 'macd_fastperiod': 10, 'macd_slowperiod': 24, 'macd_signalperiod': 7},
            {'bb_factor': 2, 'macd_fastperiod': 10, 'macd_slowperiod': 30, 'macd_signalperiod': 8},
            {'bb_factor': 3, 'macd_fastperiod': 8, 'macd_slowperiod': 28, 'macd_signalperiod': 7},
        ]

        for params in params_list:
            df = add_sigdet(df, params)

        params_list = [
            {'period': 20, 'coeff': 1, 'atr_period': 5},
            {'period': 30, 'coeff': 1.5, 'atr_period': 10},
            {'period': 15, 'coeff': 1, 'atr_period': 5},
            {'period': 20, 'coeff': 1.5, 'atr_period': 7},
        ]

        for params in params_list:
            df = add_trend_magic(df, params)

        params_list = [
            {'n1': 10, 'n2': 21, 'obLevel1': 60, 'obLevel2': 53, 'osLevel1': -60, 'osLevel2': -53},
            {'n1': 10, 'n2': 30, 'obLevel1': 55, 'obLevel2': 50, 'osLevel1': -55, 'osLevel2': -50},
            {'n1': 15, 'n2': 35, 'obLevel1': 75, 'obLevel2': 68, 'osLevel1': -75, 'osLevel2': -68},
        ]

        for params in params_list:
            df = add_wave_trend(df, params)

        params_list = [
            {'atr_period': 10, 'atr_multiplier': 2.0, 'change_atr': True},
            {'atr_period': 14, 'atr_multiplier': 3.0, 'change_atr': False},
            {'atr_period': 14, 'atr_multiplier': 1.0, 'change_atr': True},
        ]
        for params in params_list:
            df = add_supertrend(df, params)

        df = rsi_swing_indicator(df)

        # Calculate the rolling Hurst Exponent for each of the data's columns
        for col in ['close', 'volume']:
            H_rolling, c_rolling = compute_Hc_rolling(df[col], SEQUENCE_LENGTH)
            df[col + '_hurst_exp'] = H_rolling
            df[col + '_hurst_const'] = c_rolling

        federal_funds_rate = get_economic_indicator('FEDERAL_FUNDS_RATE')
        df = add_data_column(df, federal_funds_rate, 'federal_funds_rate')

        gdp = get_economic_indicator('REAL_GDP')
        df = add_data_column(df, gdp, 'real_gdp')

        inflation = get_economic_indicator('INFLATION')
        df = add_data_column(df, inflation, 'inflation')

    # Add stationary features for some reason
    df = add_stationary_features(df)

    return df
