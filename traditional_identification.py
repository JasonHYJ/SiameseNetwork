# -*- coding: utf-8 -*-

import os
import csv
from get_statistical_feature import *
from Split import *



LabDir = os.path.join('Laboratory', 'Lab_Devices')
Lab_SessionDir = os.path.join('Laboratory', 'Lab_Session')
Lab_RawdataDir = os.path.join('Laboratory', 'Lab_rawData')
Lab_Devices = ["baidu_speaker", "baidu_speaker_play_v1", "baidu_speaker_play_v2", "haique_h1-ipc", "haique_q1_lite-ipc",
               "honor_speaker", "huawei-reading_lamp", "huawei_gateway", "huawei_speaker", "samsung_gateway",
               "TP-Link_cam_ipc-43an", "TP-Link_cam_ipc-64c", "TP-Link_cam_ipc-633-d4", "xiaomi_gateway", "xiaomi_lamp",
               "xiaomi_lamp-1s", "xiaomi_speaker", "xiaomi_speaker_play", "xiaomi-ipc", "xiaomi-ipc_yuntai"]

unswDir = os.path.join('unsw2016', 'unsw2016_Devices')
unsw_SessionDir = os.path.join('unsw2016', 'unsw2016_Session')
unsw_RawdataDir = os.path.join('unsw2016', 'unsw2016_rawData')
unsw_Devices = ["Smart_Things", "Amazon_Echo", "Netatmo_Welcome", "TP-Link_Day_Night_Cloud_camera", "Samsung_SmartCam",
                "Dropcam", "Withings_Smart_Baby_Monitor", "Belkin_wemo_switch", "TP-Link_Smart_plug", "iHome", "HP_Printer",
                "Belkin_wemo_motion_sensor", "NEST_Protect_smoke_alarm", "Netatmo_weather_station", "Withings_Smart_scale",
                "Insteon_Camera", "Blipcare_Blood_Pressure_meter", "Withings_Aura_smart_sleep_sensor", "LiFX_Bulb",
                "Triby_Speaker", "PIX-STAR_Photo-frame", "Samsung_Galaxy_Tab", "Nest_Dropcam"]


unsw_Devices = ["Smart_Things", "Amazon_Echo", "Netatmo_Welcome", "TP-Link_Day_Night_Cloud_camera", "Samsung_SmartCam",
                "Withings_Smart_Baby_Monitor", "Belkin_wemo_switch", "TP-Link_Smart_plug", "Belkin_wemo_motion_sensor",
                "NEST_Protect_smoke_alarm", "Netatmo_weather_station", "Withings_Aura_smart_sleep_sensor", "LiFX_Bulb",
                "PIX-STAR_Photo-frame", "Nest_Dropcam"]




savefile_train = 'train.csv'
savefile_test = 'test.csv'
labelheader = ['length_min', 'length_max', 'length_mean', 'length_median', 'length_deviation', 'length_variance',
               'IAT_min', 'IAT_max', 'IAT_mean', 'IAT_median', 'IAT_deviation', 'IAT_variance',
               'dtype']

with open(savefile_train, "a+", newline='') as ftr:
    writer = csv.writer(ftr)
    writer.writerow(labelheader)
with open(savefile_test, "a+", newline='') as fte:
    writer = csv.writer(fte)
    writer.writerow(labelheader)




unsw_SessionCsv = os.path.join('unsw2016', 'unsw2016_Csv')
"""
for dev in unsw_Devices:
    OriginalPath = os.path.join(unsw_SessionDir, dev)
    OriginalCsvPath = os.path.join(unsw_SessionCsv, dev)
    pcapfiles = list()

    if os.path.exists(OriginalPath):
        if not os.path.exists(OriginalCsvPath):
            os.mkdir(OriginalCsvPath)

        for root, dirs, files in os.walk(OriginalPath):
            pcapfiles = files

    for pcapfile in pcapfiles:
        Session_pcapfile = os.path.join(OriginalPath, pcapfile)
        csvfile = pcapfile[:-5] + '.csv'
        Session_Csvfile = os.path.join(OriginalCsvPath, csvfile)
        cmd_ession = "tshark -r " + Session_pcapfile + " -Y \"!tcp.analysis.flags\" -T fields -e frame.number -e frame.time_epoch " \
                                                       "-e frame.time_delta_displayed -e ip.src -e ip.dst -e ip.len " \
                                                       "-e _ws.col.Info -E header=y -E separator=, -E aggregator=\">\" > " + Session_Csvfile
        r = os.popen(cmd_ession)
        r.readlines()
"""




for dev in unsw_Devices:
    OriginalCsvPath = os.path.join(unsw_SessionCsv, dev)
    csvfiles = list()

    if os.path.exists(OriginalCsvPath):
        for root, dirs, files in os.walk(OriginalCsvPath):
            csvfiles = files
    train_csvfiles, test_csvfiles = data_split(csvfiles, 0.25, shuffle=True)


    # 训练集
    for csvfile in train_csvfiles:
        Session_csvfile = os.path.join(OriginalCsvPath, csvfile)
        with open(Session_csvfile, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            IAT = [row[2] for row in reader]
            del IAT[0]

        with open(Session_csvfile, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            packetlen = [row[5] for row in reader]
            del packetlen[0]

        # 预处理过程
        for i in range(len(packetlen)):
            if packetlen[i] == "":
                packetlen[i] = 0.0
            elif '>' in packetlen[i]:
                packetlen[i] = float(packetlen[i].split('>')[0])
            else:
                packetlen[i] = float(packetlen[i])

            if IAT[i] == "":
                IAT[i] = 0.0
            else:
                IAT[i] = float(IAT[i])

        # 数据包大小的统计特征
        length_min, length_max, length_mean, length_median, length_deviation, length_variance = calculate(packetlen[0:10])
        # 到达时间间隔IAT的统计特征
        iat_min, iat_max, iat_mean, iat_median, iat_deviation, iat_variance = calculate(IAT[0:10])
        # 将统计特征写入csv
        labelvalue = [length_min, length_max, length_mean, length_median, length_deviation, length_variance,
                      iat_min, iat_max, iat_mean, iat_median, iat_deviation, iat_variance,
                      dev]
        with open(savefile_train, "a+", newline='', encoding='utf-8') as file:
            Writer = csv.writer(file)
            Writer.writerow(labelvalue)






    # 测试集
    for csvfile in test_csvfiles:
        Session_csvfile = os.path.join(OriginalCsvPath, csvfile)
        with open(Session_csvfile, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            IAT = [row[2] for row in reader]
            del IAT[0]

        with open(Session_csvfile, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            packetlen = [row[5] for row in reader]
            del packetlen[0]

        # 预处理过程
        for i in range(len(packetlen)):
            if packetlen[i] == "":
                packetlen[i] = 0.0
            elif '>' in packetlen[i]:
                packetlen[i] = float(packetlen[i].split('>')[0])
            else:
                packetlen[i] = float(packetlen[i])

            if IAT[i] == "":
                IAT[i] = 0.0
            else:
                IAT[i] = float(IAT[i])

        # 数据包大小的统计特征
        length_min, length_max, length_mean, length_median, length_deviation, length_variance = calculate(packetlen[0:10])
        # 到达时间间隔IAT的统计特征
        iat_min, iat_max, iat_mean, iat_median, iat_deviation, iat_variance = calculate(IAT[0:10])
        # 将统计特征写入csv
        labelvalue = [length_min, length_max, length_mean, length_median, length_deviation, length_variance,
                      iat_min, iat_max, iat_mean, iat_median, iat_deviation, iat_variance,
                      dev]
        with open(savefile_test, "a+", newline='', encoding='utf-8') as file:
            Writer = csv.writer(file)
            Writer.writerow(labelvalue)
