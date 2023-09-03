all_coins = [
    '1INCH',
    'AAVE',
    'ACH',
    'ADA',
    'ALGO',
    'ANKR',
    'ANT',
    'APE',
    'API3',
    'ATOM',
    'AVAX',
    'AXS',
    'BAL',
    'BAND',
    'BAT',
    'BCH',
    'BNT',
    'BOND',
    'BOSON',
    'BTC',
    'BTRST',
    'CHZ',
    'CLV',
    'COMP',
    'COTI',
    'CRV',
    'CTSI',
    'DAI',
    'DASH',
    'DGB',
    'DOGE',
    'DOT',
    'EGLD',
    'ENJ',
    'EOS',
    'ETC',
    'ETH',
    'FET',
    'FIL',
    'FLOW',
    'FORTH',
    'FTM',
    'GALA',
    'GLM',
    'GRT',
    'GTC',
    'HBAR',
    'ICP',
    'ICX',
    'IMX',
    'JASMY',
    'KAVA',
    'KNC',
    'KSM',
    'LINK',
    'LOOM',
    'LPT',
    'LRC',
    'LSK',
    'LTC',
    'MANA',
    'MASK',
    'MATIC',
    'MKR',
    'NEAR',
    'NEO',
    'NMR',
    'OCEAN',
    'OGN',
    'OMG',
    'OXT',
    'PAXG',
    'QNT',
    'QTUM',
    'REEF',
    'REN',
    'REQ',
    'RLC',
    'RNDR',
    'ROSE',
    'SAND',
    'SHIB',
    'SKL',
    'SNX',
    'SOL',
    'SPELL',
    'STORJ',
    'SUSHI',
    'TRAC',
    'TRX',
    'UNI',
    'VET',
    'WAVES',
    'XLM',
    'XTZ',
    'YFI',
    'ZEC',
    'ZEN',
    'ZIL',
    'ZRX',
]

tradable_coins = [
    'ACH',
    'ADA',
    'ANKR',
    'AVAX',
    'BCH',
    'BTC',
    'DOGE',
    'DOT',
    'ETC',
    'ETH',
    'FET',
    'FIL',
    'FTM',
    'GALA',
    'GRT',
    'HBAR',
    'ICX',
    'LINK',
    'LTC',
    'MASK',
    'MATIC',
    'NEAR',
    'RNDR',
    'SOL',
    'VET',
    'WAVES',
    'XLM',
    'ZIL',
]

trainable_coins = list(set(all_coins) - set(tradable_coins))

no_coins = ['QNT', 'ZRX', 'MATIC', 'BNT']