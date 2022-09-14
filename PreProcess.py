# -*- coding: utf-8 -*-

import os
import csv


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




savefile = 'dataOrigin.csv'



dev_IndexLabel = -1
# 让标签为设备类型(字符串)替换为对应的数字, 每次遍历一个新设备时 + 1

for dev in unsw_Devices:
    OriginalPath = os.path.join(unsw_RawdataDir, dev)       # unsw数据集
    txtfiles = list()

    if os.path.exists(OriginalPath):
        dev_IndexLabel += 1
        print(dev_IndexLabel, '  ', dev)
        for root, dirs, files in os.walk(OriginalPath):
            txtfiles = files


    for txtfile in txtfiles:
        Session_txtfile = os.path.join(OriginalPath, txtfile)

        raw_packets = []
        with open(Session_txtfile, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                raw_packets.append(lines)
                if not lines:
                    break

        # 读取会话的前10个分组数据包, 每个分组数据包读取前100个字节
        # （1个字节表示2个16进制数）
        # m 表示分组的个数, n 表示每个分组截取前 n 个字节
        m = 10
        n = 100

        packet_count = 0
        payload_feature = ""

        if len(raw_packets) >= m:   # 如果数据包分组的数目大于等于10再进行操作。
            for raw_packet in raw_packets:  # 对于每个数据包，提取前200个字符，如果少于200则补充0，如果大于200则截取前200个。一共取前10个数据包。
                if len(raw_packet) < n * 2:
                    payload_feature += str(raw_packet).ljust(n * 2, '0')
                else:
                    payload_feature += str(raw_packet[0 : n * 2])
                packet_count += 1
                if packet_count >= m:
                    break

        index = 0
        payload_features = []
        while index < len(payload_feature):
            payload_features.append(int(payload_feature[index:index+2], 16))    # 16进制数需要2位2位的加，将16进制的数转换成整数
            index += 2

        payload_features.append(dev_IndexLabel)                # 加入标签

        with open(savefile, "a+", newline='') as f:
            Writer = csv.writer(f)
            Writer.writerow(payload_features)
