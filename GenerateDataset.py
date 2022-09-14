# -*- coding: utf-8 -*-

import os
import pyshark
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




'''
for dev in unsw_Devices:
    devpcap = dev + '.pcap'
    devpcapfile = os.path.join(unswDir, devpcap)
    TargetPath = os.path.join(unsw_SessionDir, dev)
    if not os.path.exists(TargetPath):
        if os.path.exists(devpcapfile):
            os.makedirs(TargetPath)

    if os.path.exists(devpcapfile) and os.path.exists(TargetPath):
        cmd = "SplitCap -r " + devpcapfile + " -s flow -o " + TargetPath        # 按流分割样本
        r = os.popen(cmd)
        r.readlines()
'''


# 生成原始 16 进制数据
for dev in unsw_Devices:
    TargetPath = os.path.join(unsw_RawdataDir, dev)
    if not os.path.exists(TargetPath):
        os.makedirs(TargetPath)

    OriginalPath = os.path.join(unsw_SessionDir, dev)
    pcapfiles = list()
    for root, dirs, files in os.walk(OriginalPath):
        pcapfiles = files

    for pacpfile in pcapfiles:
        Session_Pcapfile = os.path.join(OriginalPath, pacpfile)
        txtfile = pacpfile[:-5] + '.txt'
        txtfile = os.path.join(TargetPath, txtfile)

        size = os.path.getsize(Session_Pcapfile)
        if size < 1024:
            os.remove(Session_Pcapfile)
            continue

        payload_list = []
        cap = pyshark.FileCapture(Session_Pcapfile, use_json=True, include_raw=True)
        for packet in cap:
            payload = str(packet.frame_raw.value)
            payload = payload[28: 52] + payload[68:]                # 去除物联层、链路层和 IP 地址
            payload_list.append(payload)
        cap.close()                                                 # 必须关闭 FileCapture, 否则会报错

        if len(payload_list) < 10:                                  # 删除分组数小于 10 的流样本
            os.remove(Session_Pcapfile)
            continue

        with open(txtfile, 'a') as file_to_write:
            for payload in payload_list:
                file_to_write.write(payload+'\n')
                # file_to_write.write(str(packet.get_raw_packet())+'\n')
        file_to_write.close()

    print(str(dev) + ' is complete...')
