# This file is the data preprocessing function

import os
import pandas as pd

def pre_process(raw_data, gaze_out, fixation_out, observe = False):
    """Preprocess the raw gaze data from Tobii eye-tracker"""

    #Get the start timestamp of whole session
    with open(raw_data) as data:
            line = data.readline()
            l = line.strip("\n").split("\t")
            while line != "":
                if l[0] == "EyePosLeftX " and len(l) == 26:
                    time_start = int(l[l.index("timestamp  ") +1])
                    data.close()
                    break
                line = data.readline()
                l = line.strip("\n").split("\t")

    #Only keep gaze x and y coordinates
    with open(raw_data) as data:
        with open("./gaze_temp.txt", 'w') as out, open("./fix_temp.txt", 'w') as fix:
            line = data.readline()
            while line is not "":
                l = line.strip("\n").split("\t")
                if (l[0] == 'Fixation_Data ' or l[0] == 'Fixation_Begin ' or l[0] == 'Fixation_End ') and (len(l)==7):
                    fix_x = l[l.index("FixationX ") +1]
                    fix_y = l[l.index("FixationY ") +1]
                    # Got a issue here:  I just manually add the first gaze data time stamp here, absolutely we can
                    # obtain this value, maybe do it later.
                    fix_time = (int(l[l.index("Timestamp ") +1])-time_start)/1000.0
                    if fix_x == "-nan(ind) ":
                        fix_x = "loss"
                    if fix_y == "-nan(ind) ":
                        fix_y = "loss"
                    if fix_time > 0:
                        fix.write("Fixation_X"+ "\t"+str(fix_x) + "\t"+ "Fixation_Y"+ "\t"+str(fix_y)+"\t"+"Timestamp"+"\t"+str(fix_time) +"\n")
                if l[0] == 'GazeX ' and len(l) == 8:
                    gaze_x = l[l.index("GazeX ") +1]
                    gaze_y = l[l.index("GazeY ") +1]
                    gaze_time = (int(l[l.index("Timestamp ") +1])-time_start)/1000.0
                    if gaze_x == "-nan(ind) ":
                        gaze_x = "loss"
                    if gaze_y == "-nan(ind) ":
                        gaze_y = "loss"
                    if gaze_time > 0:
                        out.write("Gaze_X"+ "\t"+str(gaze_x) + "\t"+ "Gaze_Y"+ "\t"+str(gaze_y)+"\t"+"Timestamp"+"\t"+str(gaze_time) +"\n")
                line = data.readline()
    gaze_vec = {}
    fix_vec = {}

    # Create a new file for processed data and replace na data with "loss", save fixation points from eyetracker and normal coordinates into two files
    with open(gaze_out,'w') as output_gaze, open(fixation_out, 'w') as output_fixation:
        output_gaze.write("x_coordinate"+"\t"+"y_coordinate"+"\t"+"time_stamp"+"\n")
        output_fixation.write("x_coordinate"+"\t"+"y_coordinate"+"\t"+"time_stamp"+"\n")
        with open("./gaze_temp.txt") as gaze, open("fix_temp.txt") as fixation:
            for line in gaze:
                gaze_vec[float(line.strip("\n").split("\t")[5])] = line.strip("\n").split("\t")[1]+"\t"+line.strip("\n").split("\t")[3]
            for line in fixation:
                fix_vec[float(line.strip("\n").split("\t")[5])] = line.strip("\n").split("\t")[1]+"\t"+line.strip("\n").split("\t")[3]
        gaze_list = sorted(gaze_vec.items(), key=lambda item: item[0])
        fix_list = sorted(fix_vec.items(), key=lambda item: item[0])
        for item in gaze_list:
            output_gaze.write(item[1] + "\t" + str(item[0]) + "\n")
        for item in fix_list:
            output_fixation.write(item[1] + "\t" + str(item[0]) + "\n")
    os.remove("./gaze_temp.txt")
    os.remove("./fix_temp.txt")
    sort_gaze = pd.read_csv(gaze_out, sep="\t")
    sort_fix = pd.read_csv(fixation_out, sep="\t")
    if observe:
        display(sort_gaze.head(20))
        display(sort_fix.head(20))
    return sort_gaze

def test():
    print("e")