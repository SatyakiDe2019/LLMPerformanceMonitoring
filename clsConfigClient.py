################################################
####                                        ####
#### Written By: SATYAKI DE                 ####
#### Written On:  15-May-2020               ####
#### Modified On: 05-Jan-2025               ####
####                                        ####
#### Objective: This script is a config     ####
#### file, contains all the keys for        ####
#### Performance tracking of LLM evaluation ####
#### solution to fetch the KPIs to tune it. ####
####                                        ####
################################################

import os
import platform as pl

class clsConfigClient(object):
    Curr_Path = os.path.dirname(os.path.realpath(__file__))

    os_det = pl.system()
    if os_det == "Windows":
        sep = '\\'
    else:
        sep = '/'

    conf = {
        'APP_ID': 1,
        'ARCH_DIR': Curr_Path + sep + 'arch' + sep,
        'PROFILE_PATH': Curr_Path + sep + 'profile' + sep,
        'LOG_PATH': Curr_Path + sep + 'log' + sep,
        'DATA_PATH': Curr_Path + sep + 'data' + sep,
        'OUTPUT_PATH': Curr_Path + sep + 'Output' + sep,
        'TEMP_PATH': Curr_Path + sep + 'temp' + sep,
        'IMAGE_PATH': Curr_Path + sep + 'Image' + sep,
        'MODEL_PATH': Curr_Path + sep + 'Model' + sep,
        'AUDIO_PATH': Curr_Path + sep + 'audio' + sep,
        'SESSION_PATH': Curr_Path + sep + 'my-app' + sep + 'src' + sep + 'session' + sep,
        'APP_DESC': 'LLM Performance Comparison!',
        'DEBUG_IND': 'Y',
        'INIT_PATH': Curr_Path,
        'MODEL_NAME_1': 'gpt-4o',
        'MODEL_NAME_2': 'claude-3-5-sonnet-20241022',
        'MODEL_NAME_3': 'deepseek-chat',
        'MODEL_NAME_4': 'CoRover/BharatGPT-3B-Indic',
        'OPEN_AI_KEY': "sk-fz1Kkdsjdj848484Y9Pt2238993jrfjfjfjfjfjf89383h",
        'ANTHROPIC_AI_KEY': "sk-ant-api03-i-gHYT__Jdjudidriri4948jrtfjfjufhhaAHHFJUFJFJFIOOIf0R0h0X3iUlyBKJUI8rCQ-0g5JDJ87JU",
        'DEEPSEEK_AI_KEY': "sk-e8383893939AKJKFHF5c30NJ98",
        'DEEPSEEK_URL': "https://api.deepseek.com/v1/chat/completions",
        'TEMP_VAL': 0.2,
        'PATH' : Curr_Path,
        'MAX_TOKEN' : 1000,
        'MAX_CNT' : 5,
        'OUT_DIR': 'data',
        'OUTPUT_DIR': 'output',
        "DB_PATH": Curr_Path + sep + 'data' + sep
    }
