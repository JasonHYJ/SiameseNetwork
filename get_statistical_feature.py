import numpy as np
import csv
import tsfresh as tsf
import pandas as pd
import random as rd

def calculate(List):
    t = pd.Series(List)
    min = tsf.feature_extraction.feature_calculators.minimum(t)  # 最小值
    max = tsf.feature_extraction.feature_calculators.maximum(t)  # 最大值
    mean = tsf.feature_extraction.feature_calculators.mean(t)  # 平均值
    median = tsf.feature_extraction.feature_calculators.median(t)  # 中位数
    deviation = tsf.feature_extraction.feature_calculators.standard_deviation(t)  # 标准差
    variance = tsf.feature_extraction.feature_calculators.variance(t)  # 方差

    return min, max, mean, median, deviation, variance
