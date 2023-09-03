from datetime import datetime
import pandas as pd
import vectorbtpro as vbt

vbt.settings.set_theme('dark')

vbt.BinanceData.set_custom_settings(
    client_config=dict(
        api_key='szjZZtbsoyZRQupFSCjVCbZjzIzykqhvKIBP7w721M31CKJB3Lm0LRckoMeaqJbK',
        api_secret='t7oWLVw6HHEQF4oXVQlwFWGnqRO89QP0H4TEJy2uzDZAt7ng3SO0VWwylvqyfEC0',
        tld='us',
    )
)

vbt.PolygonData.set_custom_settings(client_config=dict(api_key='lwdFpiZdZLX3RNGgjnpEpGzlSNPnfwmH'))


class VCDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__vc__ = None


def vc(self):
    if '__vc__' not in self.__dict__ or len(self) != len(next(iter(self.__vc__.data.values()))):
        dummy = vbt.BinanceData.fetch(
            'MATICUSD',
            timeframe='1h',
            start=datetime.strptime('2022-01-01 12:00:00', '%Y-%m-%d %H:%M:%S'),
            end=datetime.strptime('2022-01-01 12:01:00', '%Y-%m-%d %H:%M:%S'),
            show_progress=False,
        )
        self.__vc__ = vbt.BinanceData.from_data(self)
        df = next(iter(self.__vc__.data.values()))
        df.rename(columns={k: k.capitalize() for k in df.columns}, inplace=True)

        dummy._data = self.__vc__.data
        self.__vc__ = dummy
    return self.__vc__


pd.DataFrame.vc = property(vc)
