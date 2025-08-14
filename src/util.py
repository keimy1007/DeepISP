import pandas as pd 
import numpy as np
import torch
import yaml 

class ExamConfig:
    def __init__(self):
        self.config_hfa24 = pd.read_csv("./datasets/solid/coord_hfa24.csv")
        self.config_hfa10 = pd.read_csv("./datasets/solid/coord_hfa10.csv")
        self.config_isp = pd.read_csv("./datasets/solid/coord_isp.csv")

        self.dict_pos_isp_2_hfa24 = dict(zip(self.config_isp["index"], self.config_isp["index24"]))
        self.dict_pos_hfa24_2_isp = dict(zip(self.config_isp["index24"], self.config_isp["index"]))

        self.dict_pos_isp_2_hfa10 = dict(zip(self.config_isp["index"], self.config_isp["index10"]))
        self.dict_pos_hfa10_2_isp = dict(zip(self.config_isp["index10"], self.config_isp["index"]))

        self.dict_pos_isp_2_hfatype = dict(zip(self.config_isp["index"], self.config_isp["kensa"]))

        self.dict_pos_hfa24_coord = dict(zip(self.config_hfa24["index_24"], zip(self.config_hfa24["x"], self.config_hfa24["y"])))
        self.dict_pos_hfa10_coord = dict(zip(self.config_hfa10["index_10"], zip(self.config_hfa10["x"], self.config_hfa10["y"])))
        self.dict_pos_isp_coord = dict(zip(self.config_isp["index"], zip(self.config_isp["x"], self.config_isp["y"])))

        return


    def get_config_hfa24(self):
        return self.config_hfa24
    
    def get_config_hfa10(self):
        return self.config_hfa10
    
    def get_config_isp(self):
        return self.config_isp
    
    def convert_pos_hfa2isp(self, pos_hfa, hfa_type):
        if hfa_type == "hfa24":
            return int(self.dict_pos_hfa24_2_isp[pos_hfa])
        elif hfa_type == "hfa10":
            return int(self.dict_pos_hfa10_2_isp[pos_hfa])
        else:
            raise KeyError(f"{hfa_type} must be hfa24 or hfa10")

    def convert_pos_isp2hfa(self, pos_isp):
        hfa_type = self.dict_pos_isp_2_hfatype[pos_isp]
        if hfa_type == "24-2":
            return int(self.dict_pos_isp_2_hfa24[pos_isp]), "hfa24"
        elif hfa_type == "10-2":
            return int(self.dict_pos_isp_2_hfa10[pos_isp]), "hfa10"
        else:
            raise KeyError(f"{pos_isp} is not in the ISP position list")

    def get_isp_coord(self, pos_isp):
        return list(self.dict_pos_isp_coord[pos_isp])

    def get_hfa24_coord(self, pos_hfa24):
        return list(self.dict_pos_hfa24_coord[pos_hfa24])
    
    def get_hfa10_coord(self, pos_hfa10):
        return list(self.dict_pos_hfa10_coord[pos_hfa10])


    def get_isp_threshold(self, pos_isp, age):        
        age_decade = int(age / 10) * 10
        if age_decade >= 70:
            age_decade = "70over"
        elif age_decade <= 10:
            raise ValueError(f"age must be greater than 20, but {age} was given")
        
        return int(self.config_isp[self.config_isp["index"] == pos_isp][f"th{str(age_decade)}"])
    
exam_cfg = ExamConfig()


def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
