#This file contains four fixation identification methods, as well as a function to visualize the fixations

import os
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd

def fixation_detection(gaze_data , sort_fix, fix_intervals, threshold = 0, method = 'frequency'):

    #Identifying fixations by eyetrackers frequency, wuthin a valid fixation, the timestamp difference from two sequential points shouldn't be longer than 15 miliseconds
    if method == 'frequency':
        with open(sort_fix) as data: 
            with open(fix_intervals, 'w') as fix_int:
                with open("./fq_rec.csv", 'w') as fix_rec:
                    fix_int.write("start_timestamp" + "\t" + "end_timestamp" + "\n")
                    fix_rec.write("fix_time"+"\n")
                    data.readline()
                    line = data.readline()
                    nextLine = data.readline()
                    while (line is not "") and (nextLine is not ""):
                        #print(line)
                        #print(nextLine)
                        l = line.strip("\n").split("\t")
                        nL = nextLine.strip("\n").split("\t")
                        sl = l
                        el = l
                        if l[0] == 'loss' and not(nl[0] == 'loss'):
                            l = nL
                            nL = data.readline().strip("\n").split("\t")
                            sl = l
                            el = l
                        if nL[0] == 'loss':
                            l = data.readline().strip("\n").split("\t")
                            nl = data.readline().strip("\n").split("\t")
                            sl = l
                            el = l
                        while float(nL[2])-float(l[2]) < 0.015:
                            el = nL
                            l=nL
                            nL = data.readline().strip("\n").split("\t")
                            if nL[0] == '' or nL[0] == 'loss':
                                    #print("blank")
                                    break
                        if float(len(nL) == 3):
                            if float(sl[2]) != float(el[2]):
                                fix_int.write(sl[2] + "\t" + el[2] + "\n")
                                i = float(sl[2])
                                while(i <= float(el[2])):
                                    i = round(i, 3)
                                    fix_rec.write(str(i)+"\n")
                                    i += 0.001
                                #fixation_bnd.append([float(sl[2]),float(el[2])])
                        line = nextLine
                        nextLine = data.readline()
                    # else:
                    #     break

    # Speed based method, in a Gaze points whose velocity is faster than a threshold will be considered as saccades, the others will be considered as fixations
    if method == "speed":
        with open("./speed.txt", 'w') as output:
            with open(gaze_data) as data: 
                data.readline()
                line = data.readline()
                nextLine = data.readline()
                while (line is not "") and (nextLine is not ""):
                    l = line.strip("\n").split("\t")
                    nL = nextLine.strip("\n").split("\t")
                    if l[0] == 'loss' and not(nl[0] == 'loss'):
                        l = nL
                        nL = data.readline().strip("\n").split("\t")
                    if nL[0] == 'loss':
                        l = data.readline().strip("\n").split("\t")
                        nl = data.readline().strip("\n").split("\t")
                    while (float(nL[2])-float(l[2]) > 0.015):
                        l = nL
                        nL = data.readline().strip("\n").split("\t")
                    while (nL[2] == l[2]):
                        nL = data.readline().strip("\n").split("\t")
                    distance = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))
                    speed = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))/(float(nL[2])-float(l[2]))
                    output.write(str(distance) + "\t"+ str(speed)+"\t" + nL[2] + "\t" + nL[0] + "\t" + nL[1] + "\n")
                    line = data.readline()
                    nextLine = data.readline()
        fixation_bnd = []
        with open("./speed.txt") as speed, open(fix_intervals, "w") as fix, open("sp_rec.csv", "w") as fix_rec:
            fix_rec.write("fix_time" + "\n")
            write_time = 0
            fix.write("start_timestamp" + "\t" + "end_timestamp" + "\n")
            line = speed.readline()
            l = line.strip("\n").split("\t")
            start_time = l[2]
            end_time = l[2]
            new_start = False
            while line is not "":
                if float(line.strip("\n").split("\t")[1]) <= threshold:
                    if new_start:
                        start_time = line.strip("\n").split("\t")[2]
                        new_start = False
                    end_time = line.strip("\n").split("\t")[2]
                    line = speed.readline()
                elif line is not "" and float(line.strip("\n").split("\t")[1]) > threshold:
                    if (start_time != end_time):
                        if write_time != start_time:
                            fix.write(start_time + "\t" + end_time + "\n")
                            write_time = start_time
                        i = float(start_time)
                        while (i <= float(end_time)):
                            i = round(i, 3)
                            fix_rec.write(str(i) + "\n")
                            i += 0.001
                    line = speed.readline()
                    new_start = True

    #Within a valid fixation, gaze points should not be further than a threshold from the centroid.
    if method == "distance":
        from scipy.spatial import distance
        with open(gaze_data) as gaze, open(fix_intervals, "w") as fix, open("ds_rec.csv", "w") as fix_rec:
            fix_rec.write("fix_time" + "\n")
            coordinates = []
            gaze.readline()
            line = gaze.readline()
            l = line.strip("\n").split("\t")
            start_time = l[2]
            end_time = l[2]
            new_start = False
            while line is not "":
                coordinates.append((float(line.strip("\n").split("\t")[0]), float(line.strip("\n").split("\t")[1])))
                #print(coordinates)
                if distance.cdist(coordinates, coordinates, 'euclidean').max() <= threshold:
                    #print(distance.cdist(coordinates, coordinates, 'euclidean').max())
                    if new_start:
                        start_time = line.strip("\n").split("\t")[2]
                        new_start = False
                    end_time = line.strip("\n").split("\t")[2]
                    line = gaze.readline()
                    #print(line)
                elif line is not "" and distance.cdist(coordinates, coordinates, 'euclidean').max() > threshold:
                    fix.write(start_time + "\t" + end_time + "\n")
                    i = float(start_time)
                    while (i <= float(end_time)):
                        i = round(i, 3)
                        fix_rec.write(str(i) + "\n")
                        i += 0.001
                    coordinates = []
                    #coordinates.append((float(line.strip("\n").split("\t")[0]), float(line.strip("\n").split("\t")[1])))
                    line = gaze.readline()
                    #print(line)
                    new_start = True

    #In valid fixations, the maximal distance plus maximal vertical distance less than given threshold.
    if method == "salvucci":
        with open(gaze_data) as gaze, open(fix_intervals, "w") as fix, open("sv_rec.csv", "w") as fix_rec:
            fix_rec.write("fix_time" + "\n")
            x_cor = []
            y_cor = []
            gaze.readline()
            line = gaze.readline()
            l = line.strip("\n").split("\t")
            start_time = l[2]
            end_time = l[2]
            new_start = False
            while line is not "":
                x_cor.append(float(line.strip("\n").split("\t")[0]))
                y_cor.append(float(line.strip("\n").split("\t")[1]))
                #print(coordinates)
                if  (max(x_cor)-min(x_cor)) + (max(y_cor)-min(y_cor)) <= threshold:
                    #print(distance.cdist(coordinates, coordinates, 'euclidean').max())
                    if new_start:
                        start_time = line.strip("\n").split("\t")[2]
                        new_start = False
                    end_time = line.strip("\n").split("\t")[2]
                    line = gaze.readline()
                    #print(line)
                elif line is not "" and (max(x_cor)-min(x_cor)) + (max(y_cor)-min(y_cor)) > threshold:
                    fix.write(start_time + "\t" + end_time + "\n")
                    i = float(start_time)
                    while (i <= float(end_time)):
                        i = round(i, 3)
                        fix_rec.write(str(i) + "\n")
                        i += 0.001
                    x_cor = []
                    y_cor = []
                    #coordinates.append((float(line.strip("\n").split("\t")[0]), float(line.strip("\n").split("\t")[1])))
                    line = gaze.readline()
                    #print(line)
                    new_start = True




# Calculate the eye gaze speed for each timestamp and save the result as a separate file
def get_speed(gaze_data):
    with open("./speed.txt", 'w') as output:
        with open(gaze_data) as data:
            data.readline()
            line = data.readline()
            nextLine = data.readline()
            while (line is not "") and (nextLine is not ""):
                l = line.strip("\n").split("\t")
                nL = nextLine.strip("\n").split("\t")
                if l[0] == 'loss' and not(nl[0] == 'loss'):
                    l = nL
                    nL = data.readline().strip("\n").split("\t")
                if nL[0] == 'loss':
                    l = data.readline().strip("\n").split("\t")
                    nl = data.readline().strip("\n").split("\t")
                while (float(nL[2])-float(l[2]) > 0.015):
                    l = nL
                    nL = data.readline().strip("\n").split("\t")
                while (nL[2] == l[2]):
                    nL = data.readline().strip("\n").split("\t")
                distance = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))
                speed = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))/(float(nL[2])-float(l[2]))
                output.write(str(distance) + "\t"+ str(speed)+"\t" + nL[2] + "\t" + nL[0] + "\t" + nL[1] + "\n")
                line = data.readline()
                nextLine = data.readline()
    df_speed = pd.read_csv("./speed.txt", sep="\t")
    return df_speed



    
        
# Use this function in jupyternotebook. Could visualize the gaze speed within any timewindow with fixation intervals as a line chart.
def visualize_fixation(gaze_data, fix_intervals, start_timestamp, end_timestamp):
    fixation_bnd = []
    tm = []
    sp = []
    xpos = []
    with open("./speed.txt", 'w') as output:
        with open(gaze_data) as data: 
            data.readline()
            line = data.readline()
            nextLine = data.readline()
            while (line is not "") and (nextLine is not ""):
                l = line.strip("\n").split("\t")
                nL = nextLine.strip("\n").split("\t")
                if l[0] == 'loss' and not(nl[0] == 'loss'):
                    l = nL
                    nL = data.readline().strip("\n").split("\t")
                if nL[0] == 'loss':
                    l = data.readline().strip("\n").split("\t")
                    nl = data.readline().strip("\n").split("\t")
                while (float(nL[2])-float(l[2]) > 0.015):
                    l = nL
                    nL = data.readline().strip("\n").split("\t")
                while (nL[2] == l[2]):
                    nL = data.readline().strip("\n").split("\t")
                distance = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))
                speed = math.sqrt((float(nL[0])-float(l[0])) * (float(nL[0])-float(l[0])) + (float(nL[1])-float(l[1])) * (float(nL[1])-float(l[1])))/(float(nL[2])-float(l[2]))
                output.write(str(distance) + "\t"+ str(speed)+"\t" + nL[2] + "\t" + nL[0]+ "\t" + nL[1] + "\n")
                line = data.readline()
                nextLine = data.readline()
    with open("./speed.txt") as speed, open(fix_intervals) as fix:
        for line in speed:
            if float(line.strip("\n").split("\t")[2]) < end_timestamp and float(line.strip("\n").split("\t")[2]) > start_timestamp:
                tm.append(float(line.strip("\n").split("\t")[2]))
                sp.append(float(line.strip("\n").split("\t")[1]))
                xpos.append(float(line.strip("\n").split("\t")[3]))
            elif float(line.strip("\n").split("\t")[2]) > end_timestamp:
                break
        fix.readline()
        line = fix.readline()
        while line is not "" and float(line.strip("\n").split("\t")[0]) < start_timestamp:
            #print(line)
            line = fix.readline()
        while line is not "" and float(line.strip("\n").split("\t")[1]) < end_timestamp and float(line.strip("\n").split("\t")[0]) > start_timestamp:
            #print(line)
            fixation_bnd.append([float(line.strip("\n").split("\t")[0]),float(line.strip("\n").split("\t")[1])])
            line = fix.readline()
    #print(fixation_bnd)
    x = tm
    y1 = sp
    y2 = xpos
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x,y1,'r-')
    ax2.plot(x,y2,'b-')
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Speed(Pixels/s)",color='r')
    ax1.set_ylim(0,50000)
    ax2.set_ylim(0,1600)
    for pair in fixation_bnd:
        if pair[1] - pair[0] > 0:
            rect = plt.Rectangle((pair[0],0.1),pair[1]-pair[0],50000, color = 'y')
            ax1.add_patch(rect)
        
    ax2.set_ylabel("Gaze Position along the x-axis",color='b')
    #plt.savefig('./asd_gaze.jpg',quality = 100,optimize = True)
    plt.show()
    #os.remove("./speed.txt")