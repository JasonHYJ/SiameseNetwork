# -*- coding: utf-8 -*-

import os
import csv
from Attack_Time import Attack_startstop_Times
from Entropy import *



rootdir = "D:\\PycharmProjects\\DatasetLabeling"
abnormal_path = "Attack_dataset"
benign_path = "Benign_dataset"

DevList = ["Wemo_Motion_Sensor", "Wemo_Power_Switch", "Samsung_Camera", "TP-Link_Plug", "Netatmo_Camera", "Hue_bulb",
           "Amazon_Echo", "Chromecast", "ihome", "Lifx"]



for dev in DevList:
    devpcap = dev + ".pcap"
    devpcapfile_abnormal = os.path.join(rootdir, abnormal_path, devpcap)
    devpcapfile_benign = os.path.join(rootdir, benign_path, devpcap)

    devcsv = dev + ".csv"
    devcsvfile_abnormal = os.path.join(rootdir, abnormal_path, devcsv)
    devcsvfile_benign = os.path.join(rootdir, benign_path, devcsv)

    '''
    cmd_feature = "tshark -r " + devpcapfile_abnormal + " -Y \"!tcp.analysis.flags\" -T fields -e frame.number -e frame.time_epoch " \
                                                        "-e frame.time_delta_displayed -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport " \
                                                        "-e udp.srcport -e udp.dstport -e frame.len -e _ws.col.Info " \
                                                        "-E header=y -E separator=, -E aggregator=\">\" > " + devcsvfile_abnormal
    r_feature = os.popen(cmd_feature)
    r_feature.readlines()

    
    cmd_feature = "tshark -r " + devpcapfile_benign + " -Y \"!tcp.analysis.flags\" -T fields -e frame.number -e frame.time_epoch " \
                                                      "-e frame.time_delta_displayed -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport " \
                                                      "-e udp.srcport -e udp.dstport -e frame.len -e _ws.col.Info " \
                                                      "-E header=y -E separator=, -E aggregator=\">\" > " + devcsvfile_benign
    r_feature = os.popen(cmd_feature)
    r_feature.readlines()
    '''


for dev in DevList:
    moment, srcIP, dstIP, srcPort, dstPort, packetLen, dtype = [], [], [], [], [], [], []
    E_iat, E_srcIP, E_dstIP, E_srcPort, E_dstPort, E_packetLen = [], [], [], [], [], []
    EList_Count, EList_iat, EList_srcIP, EList_dstIP, EList_srcPort, EList_dstPort, EList_packetLen = [], [], [], [], [], [], []
    packetNum = []

    Entropy_srcIP, Entropy_dstIP, Entropy_srcPort, Entropy_dstPort, Entropy_packetlen = [], [], [], [], []

    beginTime = []
    endTime = []

    devCsvfile = dev + ".csv"
    abnormal_or_benign_path = "Attack_dataset"  # ???????????? abnormal ??? benign
    devfile = os.path.join(rootdir, abnormal_or_benign_path, devCsvfile)

    window_size = 300   # ??????

    csv.field_size_limit(500 * 1024 * 1024)
    with open(devfile, 'r', encoding='utf-8')as f:
        reader = csv.reader(f)
        packets = [row for row in reader]
        del packets[0]

    for i in range(len(packets)):
        moment.append(packets[i][1])
        srcIP.append(packets[i][3])
        dstIP.append(packets[i][4])

        tcp_srcport = packets[i][5]     # tcp ?????????
        udp_srcport = packets[i][7]     # udp ?????????
        if tcp_srcport != "" and udp_srcport == "":
            srcPort.append(tcp_srcport)
        elif udp_srcport != "" and tcp_srcport == "":
            srcPort.append(udp_srcport)
        else:
            srcPort.append(0)

        tcp_dstport = packets[i][6]     # tcp ????????????
        udp_dstport = packets[i][8]     # udp ????????????
        if tcp_dstport != "" and udp_dstport == "":
            dstPort.append(tcp_dstport)
        elif udp_dstport != "" and tcp_dstport == "":
            dstPort.append(udp_dstport)
        else:
            dstPort.append(0)

        if packets[i][9] == "":         # ??????
            packetLen.append(0)
        elif '>' in packets[i][9]:
            packetLen.append(float(packets[i][9].split('>')[0]))
        else:
            packetLen.append(float(packets[i][9]))


    time_start = float(moment[0])
    begin_flag = 1      # ??????????????????????????????????????????
    begin_moment, end_moment = 0, 0

    packet_num = 0
    for i in range(len(packets)):
        moment[i] = float(moment[i])
        if begin_flag:
            begin_moment = moment[i]    # ??????????????????
            packet_num = 0
            begin_flag = 0
        if moment[i] <= time_start + window_size:
            E_iat.append(packets[i][2])         # ?????? IAT
            E_srcIP.append(srcIP[i])
            E_dstIP.append(dstIP[i])
            if srcPort[i] != 0:
                E_srcPort.append(srcPort[i])
            if dstPort[i] != 0:
                E_dstPort.append(dstPort[i])
            if packetLen[i] != 0:
                E_packetLen.append(packetLen[i])
            packet_num += 1
        else:
            time_start = moment[i]      # ????????????????????????????????????
            end_moment = moment[i-1]    # ??????????????????
            # begin_moment ??? end_moment, ????????????????????????????????????????????????
            # ?????????????????????????????????????????????, ???????????????????????????????????????????????????; ?????????????????????????????????
            begin_end_moment = Attack_startstop_Times[dev]

            # ????????????????????????????????????
            if begin_moment > begin_end_moment[-1]:
                break

            bem = 0
            while bem < len(begin_end_moment):
                abnormal_beginTime = begin_end_moment[bem]
                abnormal_endTime = begin_end_moment[bem+1]
                if begin_moment > abnormal_endTime or end_moment < abnormal_beginTime:
                    bem += 2
                else:
                    dtype.append("abnormal")
                    break
            if bem >= len(begin_end_moment):
                dtype.append("normal")

            # dtype.append("normal")

            begin_flag = 1
            # ??????????????????
            beginTime.append(begin_moment)
            endTime.append(end_moment)
            packetNum.append(packet_num)      # ?????????????????????
            EList_iat.append(E_iat)
            EList_srcIP.append(E_srcIP)
            EList_dstIP.append(E_dstIP)
            EList_srcPort.append(E_srcPort)
            EList_dstPort.append(E_dstPort)
            EList_packetLen.append(E_packetLen)
            E_srcIP, E_dstIP, E_srcPort, E_dstPort, E_packetLen = [], [], [], [], []
            i = i - 1       # ????????????????????????????????????


    for i in range(len(EList_iat)):
        Entropy_srcIP.append(calcInfoShannon(EList_srcIP[i]))
        Entropy_dstIP.append(calcInfoShannon(EList_dstIP[i]))
        Entropy_srcPort.append(calcInfoShannon(EList_srcPort[i]))
        Entropy_dstPort.append(calcInfoShannon(EList_dstPort[i]))
        Entropy_packetlen.append(calcInfoShannon(EList_packetLen[i]))


    savefile = dev + "_attack.csv"
    savefile = os.path.join("devices", savefile)
    labelheader = ['beginTime', 'endTime', 'Entropy_srcIP', 'Entropy_dstIP','Entropy_packetlen',
                   'Count', 'dtype']
    with open(savefile, "a+", newline='') as ftr:
        writer = csv.writer(ftr)
        writer.writerow(labelheader)
    print(savefile)

    for i in range(len(dtype)):
        # ????????????????????? csv
        labelvalue = [beginTime[i], endTime[i], Entropy_srcIP[i], Entropy_dstIP[i], Entropy_packetlen[i],
                      packetNum[i], dtype[i]]

        if dtype[i] == "abnormal":
            with open(savefile, "a+", newline='') as file:
                Writer = csv.writer(file)
                Writer.writerow(labelvalue)
