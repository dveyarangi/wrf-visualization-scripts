import os
from pathlib import Path

folder_to_clean = r'E:\meteo\urban-wrf\wrfout'

for filename in list(Path(folder_to_clean).rglob("*")):
    if os.path.isdir(filename):
        continue

    new_filename = str(filename).replace("%3A", "^%")
    new_filename = new_filename[0:-2] + "00"

    if not new_filename == str(filename):
        print(filename)
        os.rename(str(filename), new_filename)
