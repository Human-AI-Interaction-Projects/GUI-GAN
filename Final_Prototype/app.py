# to run this web application on MAC terminal
# $ export FLASK_APP=app.py
# $ export FLASK_ENV=development                (only during debug)
# $ flask run
# after which go to the link displayed in the terminal window
#
# to run this web application on Windows terminal
# $ set FLASK_APP=app.py
# $ set FLASK_ENV=development                   (only during debug)
# $ flask run
# after which go to the link displayed in the terminal window
#
# code sources using partly in this project
# https://www.fullstackpython.com/blog/responsive-bar-charts-bokeh-flask-python-3.html

import igan_server
import igan_data
from time import sleep
import random

from flask import Flask, render_template, request, make_response, Response

from bokeh.models import PointDrawTool, ColumnDataSource, BoxSelectTool
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.colors import RGB

import numpy as np
import os

import io
import base64

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

LOG_FILE = 'tensorflow_logger.txt'
LOG_PATH = 'server_data'

# set a flask instance
app = Flask(__name__)
app.secret_key = 'some secret key'


# init handlers
load_handler = igan_server.LoadDataFormHandler("load_data_button", "input_file")
generator_handler = igan_server.GenerateDataFormHandler("generate_data_button")
imputation_handler = igan_server.ImputeDataFormHandler("impute_data_button")
switch_orig_handler = igan_server.SwitchHandler("submit_button", "original", "change_to_orig", 0)
switch_gen_handler = igan_server.SwitchHandler("submit_button", "synthesized", "change_to_gen", 0)
switch_to_prev_handler = igan_server.SwitchHandler("rotate_button", "<<<", "prev", True)
switch_to_next_handler = igan_server.SwitchHandler("rotate_button", ">>>", "next", True)
json_handler = igan_server.JSONHandler()


# define colors:
mid_black = RGB(50, 50, 50)
light_green = RGB(213, 232, 212)
light_purple = (225 / 255, 213 / 255, 231 / 255)

# init handler manager
manager = igan_server.HandlerManager([load_handler,
                                      generator_handler,
                                      imputation_handler,
                                      switch_orig_handler,
                                      switch_gen_handler,
                                      switch_to_prev_handler,
                                      switch_to_next_handler,
                                      json_handler])

# init data dictionary
data = {'orig_x': np.zeros((1, 1)), 'orig_y': np.zeros((1, 1)), 'orig_class': ['0'],
        'gen_x': np.zeros((1, 1)), 'gen_y': np.zeros((1, 1)), 'gen_class': ['0'],
        "ref_x": [], "ref_y": [],
        "start": 0, "end": 0,
        "current_orig": 0,
        "current_gen": 0,
        "display": "orig"}
# init dictionary for UI elements
ui = {"original_select": "select-button",
      "synthesized_select": "unselect-button",
      "orig_nvlt": "0", "orig_div": "0",
      "gen_nvlt": "0", "gen_div": "0", "gen_loss": "0", "gen_RMSE": "0", "gen_miscl": "0",
      "ref_x": [],
      "ref_y": [],
      "start": "0",
      "end": "0",
      "current_orig": "0",
      "current_gen": "0",
      "logs": []}

WIDTH = 1200
HEIGHT = 380
MARGIN = (0, 0, 0, 120)


@app.route('/', methods=['GET', 'POST', 'DELETE'])
def main_window():

    if request.method == 'POST':
        # prepare data pack for messages
        data_pack = {'data_dict': data}
        # get updates from the manager
        updates = manager.handle(request, data_pack)
        # manage updates
        if "orig_data_vals" in updates:
            data['orig_y'] = updates["orig_data_vals"]
            ui["orig_nvlt"] = str(round(igan_data.data_novelty(data['orig_y'])[3], 5))
            ui["orig_div"] = str(round(igan_data.data_diversity(data['orig_y'])[1], 5))
        if "orig_data_timestamps" in updates:
            data['orig_x'] = updates["orig_data_timestamps"]
        if "orig_data_classes" in updates:
            data['orig_class'] = updates["orig_data_classes"]
        if "gen_data_vals" in updates:
            data['gen_y'] = updates["gen_data_vals"]
            ui["gen_nvlt"] = str(round(igan_data.data_novelty(data['gen_y'])[3], 5))
            ui["gen_div"] = str(round(igan_data.data_diversity(data['gen_y'])[1], 5))
            ui["gen_RMSE"] = str(round(igan_data.feat_RMSE(data["orig_y"], data["gen_y"]), 5))
            ui["gen_miscl"] = "50"
        if "loss" in updates:
            ui["gen_loss"] = str(round(updates["loss"], 5))
        if "gen_data_timestamps" in updates:
            data['gen_x'] = updates["gen_data_timestamps"]
        if "gen_data_classes" in updates:
            data['gen_class'] = updates["gen_data_classes"]
        if "change_to_orig" in updates:
            data["current_orig"] = updates["change_to_orig"]
            data["display"] = "orig"
            ui["current_orig"] = str(updates["change_to_orig"])
            ui["current_orig_class"] = str(data["orig_class"][data["current_orig"]])
            ui["original_select"] = "select-button"
            ui["synthesized_select"] = "unselect-button"
        if "change_to_gen" in updates:
            data["current_gen"] = updates["change_to_gen"]
            data["display"] = "gen"
            ui["current_gen"] = str(updates["change_to_gen"])
            ui["current_gen_class"] = str(data["gen_class"][data["current_gen"]])
            ui["original_select"] = "unselect-button"
            ui["synthesized_select"] = "select-button"
        if "updated_sample" in updates:
            data["gen_y"][data["current_gen"], :] = updates["updated_sample"]
        if "next" in updates:
            if data["display"] == "orig":
                if not data["current_orig"] + 1 < data["orig_y"].shape[0]:
                    data["current_orig"] = 0
                else:
                    data["current_orig"] += 1
                ui["current_orig"] = str(data["current_orig"])
                ui["current_orig_class"] = str(data["orig_class"][data["current_orig"]])
            else:
                if not data["current_gen"] + 1 < data["gen_y"].shape[0]:
                    data["current_gen"] = 0
                else:
                    data["current_gen"] += 1
                ui["current_gen"] = str(data["current_gen"])
                ui["current_gen_class"] = str(data["gen_class"][data["current_gen"]])
        if "prev" in updates:
            if data["display"] == "orig":
                if data["current_orig"] - 1 < 0:
                    data["current_orig"] = data["orig_y"].shape[0] - 1
                else:
                    data["current_orig"] -= 1
                ui["current_orig"] = str(data["current_orig"])
                ui["current_orig_class"] = str(data["orig_class"][data["current_orig"]])
            else:
                if data["current_gen"] - 1 < 0:
                    data["current_gen"] = data["gen_y"].shape[0] - 1
                else:
                    data["current_gen"] -= 1
                ui["current_gen"] = str(data["current_gen"])
                ui["current_gen_class"] = str(data["gen_class"][data["current_gen"]])

        if "ref_points_x" in updates:
            data["ref_x"] = updates["ref_points_x"]
            ui["ref_x"] = [str(el) for el in data["ref_x"]]
        if "ref_points_y" in updates:
            data["ref_y"] = updates["ref_points_y"]
            ui["ref_y"] = [str(el) for el in data["ref_y"]]
        if "start" in updates:
            data["start"] = updates["start"]
            ui["start"] = str(updates["start"])
        if "end" in updates:
            data["end"] = updates["end"]
            ui["end"] = str(updates["end"])

    print("start", data["start"])
    print("end", data["end"])

    plot = create_chart()
    script, div = components(plot)



    return render_template("home.html",
                           the_div=div,
                           the_script=script,
                           UI=ui)


@app.route('/generated_data.csv', methods=['GET', 'POST', 'DELETE'])
def download_window():
    # if some data was generated
    if len(data["gen_y"]) != 0:
        # prepare data to save
        csv = ''
        for i in range(len(data["gen_y"])):
            csv += str(data["gen_y"][i])
        # prepare response to save
        response = make_response(csv)
        cd = 'attachment; filename=generated_data.csv'
        response.headers['Content-Disposition'] = cd
        response.mimetype='text/csv'

        return response

    # if there is no generated data yet
    else:
        return render_template("download.html")


@app.route("/log_stream", methods=["GET", "POST"])
def stream():
    """returns logging information"""
    if request.method == "GET":
        content = request.data
    else:
        content = None
    return Response(flask_logger(content), mimetype="text/plain", content_type="text/event-stream")


# allows to log message
def flask_logger(cell_content):
    # get path to the file where all the messages are stored
    log_path = os.path.join(LOG_PATH, LOG_FILE)

    # if cell_content is not None and '\n' in cell_content:
    #     cell_content = cell_content.split('\n')
    # else:
    #     cell_content = []

    while True:
        # read and yield new messages
        with open(log_path, "r") as f:
            # read all log messages and
            log_msg = f.read()
        # check if there is anything to print
        if log_msg != '':
            # split messages into strings
            log_msg = log_msg.split('\n')
            print("log_msg", log_msg)
            # if the last string is empty, delete it
            if log_msg[-1] == "": log_msg.pop()
            for i in range(len(log_msg)):
                # if the message has not been printed yet
                if log_msg[i] not in ui["logs"]:
                    # print it and save for the future
                    yield str(log_msg[i] + '<br/>')
                    ui["logs"].append(log_msg[i])
                # if the message has already been printed, do nothing
                else:
                    pass
                    # yield "No MSG"
        sleep(1)


def create_chart():
    # divide dict into parts
    orig_data = {"orig_x": data["orig_x"][data["current_orig"]],
                 "orig_y": data["orig_y"][data["current_orig"]]}
    gen_data = {"gen_x": data["gen_x"][data["current_gen"]],
                "gen_y": data["gen_y"][data["current_gen"]]}

    # init necessary tools
    tools = ["pan,wheel_zoom,box_zoom,reset"]

    # create column data sources for plot
    if ui["original_select"] == "select-button":
        # create a corresponding data source objects
        source = ColumnDataSource(data=orig_data)
    else:
        # create a corresponding data source objects
        source = ColumnDataSource(data=gen_data)
        added_points_source = ColumnDataSource(data={'gen_x': [], 'gen_y': []})

    # create a plot
    if ui["original_select"] == "select-button":
        p = figure(title="original data", x_axis_label='x', y_axis_label='y', tools=tools, sizing_mode="stretch_both")
        # add a line renderer with legend and line thickness
        p.line(x='orig_x', y='orig_y', source=source, legend_label="Temp.", line_width=2, color="crimson")
        # add a circle renderer with a size, color, and alpha
        p.circle(x='orig_x', y='orig_y', source=source, size=3, alpha=0.5, color="darkred")
    else:
        p = figure(title="genreated data", x_axis_label='x', y_axis_label='y', tools=tools, sizing_mode="stretch_both")
        # add a line renderer with legend and line thickness
        p.line(x='gen_x', y='gen_y', source=source, legend_label="Temp.", line_width=2, color="crimson")
        # add a circle renderer with a size, color, and alpha
        p.circle(x='gen_x', y='gen_y', source=source, size=3, alpha=0.5, color="darkred")

    # if the genreated data is currently considered, add box select and point draw tools
    if ui["synthesized_select"] == "select-button":
        # create a rendering tool for additional points
        r1 = p.circle(x='gen_x', y='gen_y', size=3, color="purple", source=added_points_source)
        draw_tool = PointDrawTool(renderers=[r1])
        p.add_tools(draw_tool)
        p.add_tools(BoxSelectTool(dimensions="width"))

        # set up a function that will send an asynchronous post request
        selection_callback = CustomJS(args=dict(source=source), code="""
        // copy selected indices
        var inds = cb_obj.indices;
        // copy data
        var d = source.data;
        // update text field
        var output = document.getElementById('points_select');
        output.value = d['gen_x'][inds[0]].toString() + "-" + d['gen_x'][inds[inds.length-1]].toString();
        // send data
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://127.0.0.1:5000/", true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({
            start: d['gen_x'][inds[0]],
            end: d['gen_x'][inds[inds.length-1]]
        }));
        console.log(xhr.response);
        """)

        add_point_callback = CustomJS(args={}, code="""
        // copy data
        var d = cb_obj.data;
        // update status string
        var output = document.getElementById('points_ref');
        var text = "";
        for (var i = 0; i < d['gen_y'].length; i++) {
            text += "(" + d['gen_x'][i].toFixed(0).toString() + ", " + d['gen_y'][i].toFixed(0).toString() + "), ";
        }
        output.value = text;
        // send data
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "http://127.0.0.1:5000/", true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify(d));
        console.log(xhr.response);
        console.log("???");
        """)

        source.selected.js_on_change('indices', selection_callback)
        added_points_source.js_on_change("data", add_point_callback)

    # specify background color
    p.background_fill_color = light_green
    p.border_fill_color = light_green
    # specify grid line colors
    p.xgrid.grid_line_color = mid_black
    p.xgrid.grid_line_width = 1
    p.ygrid.grid_line_color = mid_black
    p.ygrid.grid_line_width = 1
    # specify axis size and colors
    p.xaxis.axis_line_color = mid_black
    p.yaxis.axis_line_color = mid_black
    # delete legend
    p.legend.visible = False

    return p


@app.route('/orig_clsdist_plot.png')
def orig_clsdist_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["orig_y"]) >= 2:
        x,y = igan_data.class_dist(data["orig_class"])
        axis.pie(x, labels=y, autopct='%1.1f%%', shadow=True, startangle=90)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/orig_dsdiv_plot.png')
def orig_dsdiv_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["orig_y"]) >= 2:
        t, _ = igan_data.data_diversity(data["orig_y"])
        axis.hist(t, color='red', bins=40, label='Diversity')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/orig_dsdist_plot.png')
def orig_dsdist_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["orig_y"]) >= 2:
        m = igan_data.data_dist(data["orig_y"])
        axis.hist(m, color='green', bins=20)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/orig_dsnvlt_plot.png')
def orig_dsnvlt_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["orig_y"]) >= 2:
        _, flattened_f, _, _ = igan_data.data_novelty(data["orig_y"])
        axis.hist(flattened_f, color='blue', bins=20,label='Novelty')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/gen_clsdist_plot.png')
def gen_clsdist_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["gen_y"]) >= 2:
        x,y = igan_data.class_dist(data["gen_class"])
        axis.pie(x, labels=y, autopct='%1.1f%%', shadow=True, startangle=90)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/gen_dsdiv_plot.png')
def gen_dsdiv_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["gen_y"]) >= 2:
        t, _ = igan_data.data_diversity(data["gen_y"])
        axis.hist(t, color='red', bins=40, label='Diversity')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/gen_dsdist_plot.png')
def gen_dsdist_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["gen_y"]) >= 2:
        m = igan_data.data_dist(data["gen_y"])
        axis.hist(m, color='green', bins=20)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/gen_dsnvlt_plot.png')
def gen_dsnvlt_plot_png():
    # set up the figure
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    # axis.set_facecolor(light_purple)
    fig.patch.set_facecolor(light_purple)
    # if there is some data, add it to the plot
    if len(data["gen_y"]) >= 2:
        _, flattened_f, _, _ = igan_data.data_novelty(data["gen_y"])
        axis.hist(flattened_f, color='blue', bins=20,label='Novelty')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
