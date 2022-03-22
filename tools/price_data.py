# %%
from abc import ABCMeta
import pandas as pd


# %%


class PriceDataBase(object, metaclass=ABCMeta):
    def __init__(self, data=None):
        self.data = data

    def split_data(self, time_constraint_from, time_constraint_to):
        self.data = self.data.loc[time_constraint_from:time_constraint_to, :, :]
        return None

    def get_table(self, field_name):
        return self.data[field_name].unstack()

    def stack_and_insert(self, table, field_name):
        self.data[field_name] = table.stack()
        return None

    def get_sub_data(self, columns):
        return self.data[columns]

    def __repr__(self):
        try:
            return str(self.data.head(10))
        except:
            return str(self.data)
