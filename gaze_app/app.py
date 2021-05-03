# This is the code of the dashboard part
#Import the libraries

import math
import glob
import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_table
import PeyeTracking
import pandas as pd
import flask
from os import path
from PeyeTracking.data_preprocess import pre_process
import os, shutil
from PeyeTracking.fixation_classification import fixation_detection, visualize_fixation, get_speed
import cv2
import moviepy.editor as moviepy
import time

raw_file = 'file'

#Load the stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Set the path that will store the frames and video
image_directory = '/Users/shiyaoli/2019Vanderbilt/FinalS/Capstone/gaze_app/frames/'
video_directory = '/Users/shiyaoli/2019Vanderbilt/FinalS/Capstone/gaze_app/'

# Set image list and static path(will be used later)
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

#Set your color for fixation circles
color_1 = (0, 255, 0)
color_2 = (255, 0, 0)
color_3 = (0, 0, 255)
color_4 = (255, 255, 255)
color_5 = (100, 100, 100)

#Initialize the dashboard
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Hide the exceptions from website
app.config.suppress_callback_exceptions = True

# Arrange the layout

#Create the data upload component
app.layout = html.Div([
    html.H2(children='Upload your raw gaze data file here', id='title',style={'background-image': 'url(https://immersed.io/wp-content/uploads/2017/06/11-e1498743729956.jpg)',
                                                                              'color': 'white'}),
    dcc.Upload(
            id='upload_raw',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '90%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
    html.Div(id='file_info'),
    html.Button('Raw gaze data preprocess', id='pre_process', n_clicks=0),
    html.Div(id='container-buttons',
                 children='Enter a value and press submit'),
    html.Button('View the data', id='view_data', n_clicks=0),
    html.Div(id='intermediate_value', style={'display': 'none'}),
    dcc.Store(id = 'data'),
    html.Br(),
    html.I("Please input the start and end timestamp of the session you'd like to visualize:"),
    html.Br(),
    dcc.Input(id="start_time", type="number", placeholder="", debounce=True),
    dcc.Input(id="end_time", type="number", placeholder="", debounce=True),
    html.Button('Visualize gaze speed', id='visual_data', n_clicks=0),
    html.Br(),

    #Linechart y variable selection
    html.Div([
        dcc.Dropdown(
            id='line_chart_dropdown',
            options=[
                {'label': 'gaze speed', 'value': 'speed'},
                {'label': 'x position', 'value': 'x_position'},
                {'label': 'y position', 'value': 'y_position'},
                {'label': 'distance', 'value': 'distance'}
            ],
            value='speed',
            placeholder='Please select the variable you\'d like to visualize here ',
        ),
        ],style={"width": "48%", 'display': 'inline-block'},),

    #Linechart fixation method selection
    html.Div([
        dcc.Dropdown(
            id='fixation_dropdown',
            options=[
                {'label': 'speed based', 'value': 'speed'},
                {'label': 'salvucci method', 'value': 'salvucci'},
                {'label': 'frequency based', 'value': 'frequency'},
                {'label': 'distance from center based', 'value': 'distance'}
            ],
            value='frequency based',
            multi = False,
            placeholder='Please select the fixation identification methods here ',
        ),
        ],style={"width": "48%", 'display': 'inline-block', 'float': 'right'},),

    #Threshold for the selected fixation method
    dcc.Input(id="threshold", type="number", placeholder="Please enter the threshold here", debounce=True),
    html.Button('Visualize fixations', id='visual_fix', n_clicks=0),
    html.Div(id='dd-output-container'),

    #Show the gaze data on dashboard
    dash_table.DataTable(
            id='gaze_table',
            columns=[{"name": 'x_coordinate', "id": 'x_coordinate'},
                     {"name": 'y_coordinate', "id": 'y_coordinate'},
                     {"name": 'time_stamp', "id": 'time_stamp'}],
            style_cell=dict(textAlign='left'),
            style_header=dict(backgroundColor="paleturquoise"),
            style_data=dict(backgroundColor="lavender")
        ),
    html.Div([], id = 'graph_container'),
    html.I("Please input the start and end timestamp of the session you'd like to generate video and frames:", style={'padding': 10}),
    dcc.Input(id="start_frame", type="number", placeholder="", debounce=True, style={'padding': 10}),
    dcc.Input(id="end_frame", type="number", placeholder="", debounce=True),
    html.Br(),
    html.I("Please input the fixation classification method for generating video and frames:", style={'padding': 10}),

    #Fixation method for generating video and frames selection
    dcc.Checklist(
        id = "fix_method",
        options=[
            {'label': 'Frequency based method', 'value': 'Frequency based method'},
            {'label': 'Speed based method', 'value': 'Speed based method'},
            {'label': 'Distance based method', 'value': 'Distance based method'},
            {'label': 'Salvucci method', 'value': 'Salvucci method'}
        ],
        value=['']
    ),

    #Instruction for color and fixation methods
    html.I("Green for original gaze points, blue for frequency method, red for speed method, white for centroid method, gray for Salvucci method.", style={'padding': 10}),
    html.Br(),
    dcc.Input(id="sp_thres", type="number", placeholder="Speed threshold"),
    dcc.Input(id="dis_thres", type="number", placeholder="Distance threshold"),
    dcc.Input(id="sav_thres", type="number", placeholder="Salvucci threshold"),
    html.Button('Get video and all the frames', id='get_frame', n_clicks=0),

    #The html video component for playing generated video
    html.Video(id = 'video', src= static_image_route + 'movie_18_2nd.mp4', controls=True),

    #Selecting the image based on a given timestamp
    dcc.Dropdown(
        id='image_dropdown',
        options=[],
        value='Select the image'
    ),

    #The html video component for showing generated frames
    html.Img(id = 'image', src = static_image_route + '6283.png')


])

def set_filename(raw_file_input):
    raw_file = raw_file_input


#Showing file name after uploading
@app.callback(Output('file_info', 'children'),
              Input('upload_raw', 'filename'),)
def update_file(filename):
    if filename:
        raw_file = '../sample_data/' + str(filename[0])
        set_filename(raw_file)
        return 'You have uploaded this raw gaze data file: ' + str(filename[0])


#Informing if the data has been preprocessed succeesfully and also provide the session length
@app.callback(
    [Output('container-buttons', 'children'),
     Output('data', 'data'),
     Output('start_time', 'value'),
     Output('end_time', 'value')],
    [Input('pre_process', 'n_clicks'),
    Input('upload_raw', 'filename')])
def update_process(n_clicks, filename):
    if n_clicks == 1:
        processed_gaze = pre_process('../sample_data/' + str(filename[0]), gaze_out="../out_sample.txt", fixation_out="./fixation_sample.txt", observe=False)
        duration = processed_gaze.iloc[-1]['time_stamp']
        return 'The raw data has been processed sucessfully! The duration of this session is ' + str(duration) + " seconds", processed_gaze.head(50).to_dict('records'), 0, duration
    else:
        return '', [], 0, 0

#The button which shows the data
@app.callback(
    Output('gaze_table', 'data'),
    Input('view_data', 'n_clicks'),
    Input('data', 'data'),)
def show_data(button, children):
    if button % 2 == 0:
        children = []
    return children

#Interactive linechart with fixation identification method selection
@app.callback(
    Output('graph_container', 'children'),
    [Input('visual_data', 'n_clicks'),
    Input('line_chart_dropdown', 'value'),
    Input('fixation_dropdown', 'value'),
    Input('visual_fix', 'n_clicks'),
    State('threshold','value'),
    State('start_time', 'value'),
    State('end_time', 'value')],
    State('graph_container', 'children'))
def show_data(button, drop_title, fix_method, fix_butt, threshold, start, end, children):
    #count = fix_butt-1
    speed_data = get_speed("../out_sample.txt")
    speed_data.columns = ['distance', 'speed', 'timestamp', 'x_position', 'y_position']
    filtered_data = speed_data[(speed_data['timestamp'] >= start) & (speed_data['timestamp'] <= end)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x = filtered_data['timestamp'], y = filtered_data[drop_title], mode = 'lines', showlegend=False)
    )
    fixation_bnd = []
    fixation_detection(gaze_data = "../out_sample.txt", sort_fix = "./fixation_sample.txt", fix_intervals = "./fixation_intervals.txt", method = fix_method, threshold = threshold)
    if path.exists("./fixation_intervals.txt"):
        with open("./fixation_intervals.txt") as fix:
            fix.readline()
            line = fix.readline()
            while line is not "" and float(line.strip("\n").split("\t")[0]) < start:
                # print(line)
                line = fix.readline()
            while line is not "" and float(line.strip("\n").split("\t")[1]) < end and float(line.strip("\n").split("\t")[0]) > start:
                s = float(line.strip("\n").split("\t")[0])
                e = float(line.strip("\n").split("\t")[1])
                if [s, e] not in fixation_bnd:
                    fig.add_trace(go.Scatter(x=[s, e], y = [150000, 150000], fill='tozeroy',
                                             mode='none', fillcolor='pink', opacity=0.1, showlegend=False  # override default markers+lines
                                             ))
                line = fix.readline()
        os.remove("./fixation_intervals.txt")
    if button > 0:
        children = []
        children.append(
            dcc.Graph(
                figure = fig
            )

        )
        return children
    return children


# Generating video and frames for a given time window
@app.callback(
    [Output('image_dropdown', 'options'),
     Output('video', 'src')],
    [Input('get_frame', 'n_clicks'),
     Input('image_dropdown', 'options'),
    State('start_frame', 'value'),
    State('end_frame', 'value'),
    State('fix_method', 'value'),
    State('sp_thres', 'value'),
    State('dis_thres', 'value'),
    State('sav_thres', 'value')],)
def arrange_frames(n_clicks, options, start, end, ck_value, sp_thres, dis_thres, sav_thres):
    # cap = cv2.VideoCapture("./movie_18_2nd")
    # start_frame_num =
    gaze_data = pd.read_csv('../out_sample.txt', sep='\t')
    gaze_data = gaze_data.round({'time_stamp': 3})
    filter_data = gaze_data = gaze_data[(gaze_data['time_stamp'] >= start) & (gaze_data['time_stamp'] <= end)]
    filter_data.to_csv("piece_data.csv", sep="\t", index=False)
    gaze_data = gaze_data.astype({'time_stamp': 'str'})
    if n_clicks == 0:
        clear_dir("./frames")
        return  options, static_image_route + 'movie_18_2nd.mp4'
    clear_dir("./frames")
    start_frame_num = math.ceil(((start-56.5)*1000)/40)
    end_frame_num = math.ceil(((end - 56.5) * 1000) / 40)
    vid = cv2.VideoCapture('movie_18_2nd.mp4')
    print(vid.get(cv2.CAP_PROP_FPS))
    vid.set(cv2.CAP_PROP_FPS, 25)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    out = cv2.VideoWriter('period.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (frame_width, frame_height))
    time_curframe = round(start, 3)
    gaze_values = gaze_data.time_stamp.values
    for i in range(start_frame_num, end_frame_num+1):
        vid.set(1, i)
        ret, still = vid.read()
        #print(time_curframe)
        for s in range(40):
            if str(time_curframe) in gaze_values:
                x = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["x_coordinate"].values[0]
                y = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["y_coordinate"].values[0]
                if x != "loss" and y != "loss":
                    gaze_x = float(x) * (1092 / 1600)
                    gaze_y = float(y) * (614 / 900)
                    cv2.circle(still, (int(gaze_x), int(gaze_y)), 15, color_1, 2)
                if "Frequency based method" in ck_value:
                    f_data = pd.read_csv('fixation_sample.txt', sep='\t')
                    f_data = f_data.round({'time_stamp': 3})
                    filter_fdata = f_data[(f_data['time_stamp'] >= start) & (f_data['time_stamp'] <= end)]
                    filter_fdata.to_csv("piece_fdata.csv", sep="\t", index=False)
                    fixation_detection(gaze_data = "piece_data.csv", sort_fix = "piece_fdata.csv", fix_intervals = "./fixation_intervals.txt", method = "frequency")
                    frec_data = pd.read_csv('fq_rec.csv', sep='\t')
                    frec_data = frec_data.round({'fix_time': 3})
                    frec_data = frec_data.astype({'fix_time': 'str'})
                    os.remove("./fixation_intervals.txt")
                    f_values = frec_data.fix_time.values
                    if str(time_curframe) in f_values:
                        print(time_curframe)
                        x = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["x_coordinate"].values[0]
                        y = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["y_coordinate"].values[0]
                        if x != "loss" and y != "loss":
                            gaze_x = float(x) * (1092 / 1600)
                            gaze_y = float(y) * (614 / 900)
                            cv2.circle(still, (int(gaze_x), int(gaze_y)), 20, color_2, 2)
                if "Speed based method" in ck_value:
                    fixation_detection(gaze_data = "piece_data.csv", sort_fix = "piece_fdata.csv", fix_intervals = "./fixation_intervals.txt", method = "speed", threshold=sp_thres)
                    sp_data = pd.read_csv('sp_rec.csv', sep='\t')
                    sp_data = sp_data.round({'fix_time': 3})
                    sp_data = sp_data.astype({'fix_time': 'str'})
                    os.remove("./fixation_intervals.txt")
                    sp_values = sp_data.fix_time.values
                    if str(time_curframe) in sp_values:
                        #print(time_curframe)
                        x = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["x_coordinate"].values[0]
                        y = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["y_coordinate"].values[0]
                        if x != "loss" and y != "loss":
                            gaze_x = float(x) * (1092 / 1600)
                            gaze_y = float(y) * (614 / 900)
                            cv2.circle(still, (int(gaze_x), int(gaze_y)), 25, color_3, 2)
                if "Distance based method" in ck_value:
                    fixation_detection(gaze_data = "piece_data.csv", sort_fix = "piece_fdata.csv", fix_intervals = "./fixation_intervals.txt", method = "distance", threshold=dis_thres)
                    dis_data = pd.read_csv('ds_rec.csv', sep='\t')
                    dis_data = dis_data.round({'fix_time': 3})
                    dis_data = dis_data.astype({'fix_time': 'str'})
                    os.remove("./fixation_intervals.txt")
                    dis_values = dis_data.fix_time.values
                    if str(time_curframe) in dis_values:
                        #print(time_curframe)
                        x = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["x_coordinate"].values[0]
                        y = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["y_coordinate"].values[0]
                        if x != "loss" and y != "loss":
                            gaze_x = float(x) * (1092 / 1600)
                            gaze_y = float(y) * (614 / 900)
                            cv2.circle(still, (int(gaze_x), int(gaze_y)), 30, color_4, 2)
                if "Salvucci method" in ck_value:
                    fixation_detection(gaze_data = "piece_data.csv", sort_fix = "piece_fdata.csv", fix_intervals = "./fixation_intervals.txt", method = "salvucci", threshold=sav_thres)
                    sav_data = pd.read_csv('sv_rec.csv', sep='\t')
                    sav_data = sav_data.round({'fix_time': 3})
                    sav_data = sav_data.astype({'fix_time': 'str'})
                    os.remove("./fixation_intervals.txt")
                    sav_values = sav_data.fix_time.values
                    if str(time_curframe) in sav_values:
                        #print(time_curframe)
                        x = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["x_coordinate"].values[0]
                        y = gaze_data[gaze_data['time_stamp'] == str(time_curframe)]["y_coordinate"].values[0]
                        if x != "loss" and y != "loss":
                            gaze_x = float(x) * (1092 / 1600)
                            gaze_y = float(y) * (614 / 900)
                            cv2.circle(still, (int(gaze_x), int(gaze_y)), 35, color_5, 2)
            time_curframe += 0.001
            time_curframe = round(time_curframe, 3)
        out.write(still)
        cv2.imwrite("./frames/%d.png" % i, still)
    #list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
    out.release()
    vid.release()
    options = []
    for i in range(start*1000, end*1000+1):
        options.append(i/1000)
    clip = moviepy.VideoFileClip("period.mp4")
    p_name = "period"+str(time.time())+".mp4"
    clip.write_videofile(p_name)
    return [{'label': i, 'value': i} for i in sorted(options)], static_image_route + p_name

#Showing the image based on specific timestamp from the timewindow
@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image_dropdown', 'value'),
     dash.dependencies.Input('data', 'data')])
def update_image_src(value, data):
    gaze_data = pd.read_csv('../out_sample.txt', sep='\t')
    gaze_data = gaze_data.round({'time_stamp': 3})
    gaze_data = gaze_data.astype({'time_stamp': 'str'})
    if value != 'Select the image':
        img_num = math.ceil(((float(value) - 56.5) * 1000) / 40)
        return static_image_route + str(img_num)+'.png'

#Remove all files from a given folder
def clear_dir(dir):
    folder = dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#Server setting which enables local image serving
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    print(image_path)
    image_name = '{}.png'.format(image_path)
    print(image_name)
    return flask.send_from_directory(image_directory, image_name)

#Server setting which enables local video serving
@app.server.route('{}<video_path>.mp4'.format(static_image_route))
def serve_static(video_path):
    print(video_path)
    video_name = '{}.mp4'.format(video_path)
    print(video_name)
    return flask.send_from_directory(video_directory, video_name)

#Run the dashboard on local server
app.run_server(debug=True, use_reloader=False)




