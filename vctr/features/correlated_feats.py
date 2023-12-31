correlated_feats = [
    '1_pct_above_48_pd_minima',
    '2_pct_above_24_pd_minima',
    '3_pct_above_12_pd_minima',
    '3_pct_above_48_pd_minima',
    'close_log_diff',
    'dema50_abv_zlema150',
    'ema75_abv_zlema225',
    'higher_highs_48_3',
    'imacd_20_6_abv_sig',
    'imacd_20_6_histo_abv_0',
    'imacd_20_6_und_1',
    'imacd_20_6_und_1p5',
    'imacd_20_6_und_2',
    'imacd_20_6_und_p20',
    'imacd_20_6_up_trnd',
    'imacd_34_10_histo_abv_0',
    'imacd_34_10_und_1',
    'imacd_34_10_und_1p5',
    'imacd_34_10_und_2',
    'imacd_34_10_und_p20',
    'imacd_34_10_up_trnd',
    'imacd_54_14_histo_abv_0',
    'imacd_54_14_und_1',
    'imacd_54_14_und_1p5',
    'imacd_54_14_und_2',
    'imacd_54_14_und_p20',
    'imacd_66_18_abv_1',
    'imacd_66_18_abv_1p5',
    'imacd_66_18_abv_p20',
    'imacd_66_18_histo_abv_0',
    'imacd_66_18_und_1',
    'imacd_66_18_und_1p5',
    'imacd_66_18_und_2',
    'imacd_66_18_und_p20',
    'lower_lows_24_3',
    'lower_lows_24_9',
    'lower_lows_48_3',
    'lower_lows_48_9',
    'prc_abv_12_prd',
    'prc_abv_84_prd',
    'prc_abv_bbu_60_2',
    'prc_abv_dema20_and_hma80',
    'prc_abv_dema50',
    'prc_abv_ema75_and_zlema225',
    'prc_abv_ema75',
    'prc_abv_hma300',
    'prc_abv_sma15_and_dema50',
    'prc_abv_wma35_and_hma110',
    'prc_abv_zlema225',
    'prc_abv_zlma80_and_hma300',
    'prc_abv_zlma80',
    'prc_und_bbl_30_2',
    'prc_und_bbl_60_2',
    'rsi2_ema7_both_und_40',
    'rsi5_btw_20_80',
    'rsi5_btw_40_80',
    'rsi5_und_20',
    'rsi5_und_40',
    'ta_accdistindexindicator_acc_dist_index',
    'ta_adxindicator_adx_neg',
    'ta_adxindicator_adx_pos',
    'ta_adxindicator_adx',
    'ta_aroonindicator_aroon_down',
    'ta_aroonindicator_aroon_indicator',
    'ta_averagetruerange_average_true_range',
    'ta_awesomeoscillatorindicator_awesome_oscillator',
    'ta_bollingerbands_bollinger_hband',
    'ta_bollingerbands_bollinger_lband',
    'ta_bollingerbands_bollinger_mavg',
    'ta_bollingerbands_bollinger_pband',
    'ta_bollingerbands_bollinger_wband',
    'ta_cciindicator_cci',
    'ta_cumulativereturnindicator_cumulative_return',
    'ta_dailylogreturnindicator_daily_log_return',
    'ta_dailyreturnindicator_daily_return',
    'ta_donchianchannel_donchian_channel_hband',
    'ta_donchianchannel_donchian_channel_lband',
    'ta_donchianchannel_donchian_channel_mband',
    'ta_donchianchannel_donchian_channel_pband',
    'ta_donchianchannel_donchian_channel_wband',
    'ta_emaindicator_ema_indicator',
    'ta_ichimokuindicator_ichimoku_a',
    'ta_ichimokuindicator_ichimoku_b',
    'ta_ichimokuindicator_ichimoku_base_line',
    'ta_ichimokuindicator_ichimoku_conversion_line',
    'ta_kamaindicator_kama',
    'ta_keltnerchannel_keltner_channel_hband',
    'ta_keltnerchannel_keltner_channel_lband',
    'ta_keltnerchannel_keltner_channel_mband',
    'ta_keltnerchannel_keltner_channel_pband',
    'ta_keltnerchannel_keltner_channel_wband',
    'ta_kstindicator_kst_sig',
    'ta_macd_macd_diff',
    'ta_macd_macd_signal',
    'ta_macd_macd',
    'ta_mfiindicator_money_flow_index',
    'ta_negativevolumeindexindicator_negative_volume_index',
    'ta_onbalancevolumeindicator_on_balance_volume',
    'ta_percentagepriceoscillator_ppo_hist',
    'ta_percentagepriceoscillator_ppo_signal',
    'ta_percentagepriceoscillator_ppo',
    'ta_percentagevolumeoscillator_pvo_signal',
    'ta_psarindicator_psar_down_indicator',
    'ta_psarindicator_psar_down',
    'ta_psarindicator_psar_up',
    'ta_psarindicator_psar',
    'ta_rocindicator_roc',
    'ta_rsiindicator_rsi',
    'ta_smaindicator_sma_indicator',
    'ta_stcindicator_stc',
    'ta_stochasticoscillator_stoch_signal',
    'ta_stochasticoscillator_stoch',
    'ta_stochrsiindicator_stochrsi_d',
    'ta_stochrsiindicator_stochrsi_k',
    'ta_stochrsiindicator_stochrsi',
    'ta_trixindicator_trix',
    'ta_tsiindicator_tsi',
    'ta_ultimateoscillator_ultimate_oscillator',
    'ta_volumeweightedaverageprice_volume_weighted_average_price',
    'ta_vortexindicator_vortex_indicator_diff',
    'ta_vortexindicator_vortex_indicator_neg',
    'ta_vortexindicator_vortex_indicator_pos',
    'ta_williamsrindicator_williams_r',
    'ta_wmaindicator_wma',
    'talib_acos_real',
    'talib_add_real',
    'talib_adxr_real',
    'talib_asin_real',
    'talib_atan_real',
    'talib_avgprice_real',
    'talib_bbands_lowerband',
    'talib_bbands_middleband',
    'talib_bbands_upperband',
    'talib_cdlharamicross_integer',
    'talib_cdlhighwave_integer',
    'talib_cdllongleggeddoji_integer',
    'talib_cdlrickshawman_integer',
    'talib_cdltakuri_integer',
    'talib_ceil_real',
    'talib_cmo_real',
    'talib_cos_real',
    'talib_cosh_real',
    'talib_dema_real',
    'talib_dx_real',
    'talib_ema_real',
    'talib_exp_real',
    'talib_floor_real',
    'talib_ht_dcphase_real',
    'talib_ht_sine_leadsine',
    'talib_ht_sine_sine',
    'talib_ht_trendline_real',
    'talib_kama_real',
    'talib_linearreg_intercept_real',
    'talib_linearreg_real',
    'talib_linearreg_slope_real',
    'talib_ln_real',
    'talib_log10_real',
    'talib_ma_real',
    'talib_macd_macd',
    'talib_macd_macdsignal',
    'talib_macdext_macd',
    'talib_macdext_macdsignal',
    'talib_macdfix_macd',
    'talib_macdfix_macdhist',
    'talib_macdfix_macdsignal',
    'talib_mama_fama',
    'talib_mama_mama',
    'talib_mavp_real',
    'talib_max_real',
    'talib_medprice_real',
    'talib_midpoint_real',
    'talib_midprice_real',
    'talib_min_real',
    'talib_minindex_integer',
    'talib_minmax_max',
    'talib_minmax_min',
    'talib_minmaxindex_maxidx',
    'talib_minmaxindex_minidx',
    'talib_minus_di_real',
    'talib_minus_dm_real',
    'talib_mom_real',
    'talib_mult_real',
    'talib_natr_real',
    'talib_obv_real',
    'talib_plus_dm_real',
    'talib_ppo_real',
    'talib_roc_real',
    'talib_rocp_real',
    'talib_rocr_real',
    'talib_rocr100_real',
    'talib_rsi_real',
    'talib_sar_real',
    'talib_sarext_real',
    'talib_sin_real',
    'talib_sinh_real',
    'talib_sma_real',
    'talib_sqrt_real',
    'talib_stoch_slowd',
    'talib_stochf_fastd',
    'talib_stochrsi_fastd',
    'talib_stochrsi_fastk',
    'talib_sub_real',
    'talib_sum_real',
    'talib_t3_real',
    'talib_tan_real',
    'talib_tanh_real',
    'talib_tema_real',
    'talib_trange_real',
    'talib_trima_real',
    'talib_trix_real',
    'talib_tsf_real',
    'talib_typprice_real',
    'talib_var_real',
    'talib_wclprice_real',
    'talib_willr_real',
    'talib_wma_real',
    'wqa101_101_out',
    'wqa101_11_out',
    'wqa101_14_out',
    'wqa101_19_out',
    'wqa101_22_out',
    'wqa101_25_out',
    'wqa101_26_out',
    'wqa101_27_out',
    'wqa101_32_out',
    'wqa101_36_out',
    'wqa101_37_out',
    'wqa101_39_out',
    'wqa101_40_out',
    'wqa101_41_out',
    'wqa101_42_out',
    'wqa101_43_out',
    'wqa101_47_out',
    'wqa101_49_out',
    'wqa101_5_out',
    'wqa101_51_out',
    'wqa101_52_out',
    'wqa101_57_out',
    'wqa101_61_out',
    'wqa101_62_out',
    'wqa101_64_out',
    'wqa101_65_out',
    'wqa101_66_out',
    'wqa101_71_out',
    'wqa101_72_out',
    'wqa101_73_out',
    'wqa101_74_out',
    'wqa101_75_out',
    'wqa101_77_out',
    'wqa101_78_out',
    'wqa101_81_out',
    'wqa101_83_out',
    'wqa101_84_out',
    'wqa101_86_out',
    'wqa101_88_out',
    'wqa101_94_out',
    'wqa101_98_out',
]
