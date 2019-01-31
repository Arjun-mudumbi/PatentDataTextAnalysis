
import os
for file in os.listdir("/home/msarjun/Desktop/Fall-2018/DAEN-690/Patent Data_PD"):
    if file.endswith(".json"):
        print(os.path.join("/home/msarjun/Desktop/Fall-2018/DAEN-690/Patent Data_PD", file))
