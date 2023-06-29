import pandas as pd
import numpy as np
import os
import func
from datetime import datetime
import copy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from numpy import VisibleDeprecationWarning
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


class Base:
    def __init__(self, data: pd.DataFrame) -> None:
        data.reset_index(drop=True, inplace=True)

        # Check các cột bắt buộc phải có và không được coi là biến
        dropped_cols = ["TIME", "PROFIT", "SYMBOL", "VALUEARG"]
        for col in dropped_cols:
            if col not in data.columns:
                raise Exception(f"Thiếu cột {col}")

        # Check kiểu dữ liệu của cột TIME và cột PROFIT
        if data["TIME"].dtype != "int64": raise
        if data["PROFIT"].dtype != "float64": raise

        # Check tính giảm dần của cột TIME
        if data["TIME"].diff().max() > 0:
            raise Exception("Các giá trị trong cột TIME phải được sắp xếp giảm dần")

        # Check các chu kì và lấy index
        time_unique = data["TIME"].unique()
        index = []
        for i in range(data["TIME"].max(), data["TIME"].min()-1, -1):
            if i not in time_unique:
                raise Exception(f"Thiếu chu kì {i}")

            index.append(data[data["TIME"] == i].index[0])

        index.append(data.shape[0])
        self.INDEX = np.array(index)

        # Loại các cột có kiểu dữ liệu không phải là số nguyên (int64) và số thực (float64)
        for col in data.columns:
            if col not in dropped_cols and data[col].dtype != "int64" and data[col].dtype != "float64":
                dropped_cols.append(col)

        print(f"Các cột không được coi là biến: {dropped_cols}")
        self.dropped_cols = dropped_cols

        # Các thuộc tính
        self.data = data
        self.PROFIT = np.array(data["PROFIT"], dtype=np.float64)

        operand_data = data.drop(columns=dropped_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = np.transpose(np.array(operand_data, dtype=np.float64))

        symbol_name = data["SYMBOL"].unique()
        self.symbol_name = {symbol_name[i]:i for i in range(len(symbol_name))}
        self.SYMBOL = np.array([self.symbol_name[s] for s in data["SYMBOL"]])
        self.symbol_name = {v:k for k,v in self.symbol_name.items()}

        # Thuộc tính bổ sung cho các phương thức
        self.__string_operator = "+-*/"

    def convert_arrF_to_strF(self, arrF):
        strF = ""
        for i in range(len(arrF)):
            if i % 2 == 1:
                strF += str(arrF[i])
            else:
                strF += self.__string_operator[arrF[i]]

        return strF

    def convert_strF_to_arrF(self, strF):
        f_len = sum(strF.count(c) for c in self.__string_operator) * 2
        str_len = len(strF)
        arrF = np.full(f_len, 0)

        idx = 0
        for i in range(f_len):
            if i % 2 == 1:
                t_ = 0
                while True:
                    t_ = 10*t_ + int(strF[idx])
                    idx += 1
                    if idx == str_len or strF[idx] in self.__string_operator:
                        break

                arrF[i] = t_
            else:
                arrF[i] = self.__string_operator.index(strF[idx])
                idx += 1

        return arrF
