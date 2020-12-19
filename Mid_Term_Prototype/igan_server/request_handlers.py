import scipy.io
import numpy as np
import igan_data
import os

UPLOADS_DIR = 'server_data'
FILE_NAME = "my_data_zipped.zip"

# some constants we will need
ALLOWED_EXTENSIONS = ['mat', 'csv', 'zip']


# dummy function to emulate data generation
def dummy_generator(values, timestamps):
    return values, timestamps


# a function to check whether the file format is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# converts a dictionary with a single entry "val" to 2 arrays: time stamp and value
def ecg_mat_to_np_converter(inp):
    values = inp["val"].flatten()
    timestamps = np.arange(len(values))
    return values, timestamps


# class that manages of all handlers
class HandlerManager(object):
    def __init__(self, handlers):
        self.handlers = handlers

    def handle(self, form, data_pack):
        updates = {}
        for handler in self.handlers:
            updates.update(handler.handle(form, data_pack))

        return updates


# a base class for all interactions with forms
class FormHandler(object):
    # an interface that all interactions should support
    def handle(self, request, data_pack):
        updates = []
        return updates


# class to manage all data coming through json lists
class JSONHandler(FormHandler):
    def handle(self, request, data_pack):
        # handle only in case incoming request is a JSON
        if request.is_json:
            incoming = request.json
            # if it was a "selection" query
            if "start" in incoming:
                # just push it back to the app
                return incoming
            # if it was a new reference point
            elif "gen_x" in incoming:
                updates = {"ref_points_x": incoming['gen_x'],
                           "ref_points_y": incoming['gen_y']}
                return updates
            # if it was neither
            else:
                return {}
        else:
            return {}


# class to handle switch buttons
class SwitchHandler(FormHandler):
    # button_name - name of the button used for form submission
    def __init__(self, button_name, button_value, update_name, update_value):
        # save the name of the form this handler is connected to
        self.button_name = button_name
        self.button_value = button_value
        self.update_name = update_name
        self.update_value = update_value

    # switch to original
    def handle(self, request, data_pack):
        # handle only in case load button is pressed and there is a button
        if self.button_name in request.form and request.form[self.button_name] == self.button_value:
            updates = {self.update_name: self.update_value}
            return updates
        else:
            return {}


# class to handle data imputation
class ImputeDataFormHandler(FormHandler):
    # button_name - name of the button used for form submission
    def __init__(self, button_name):
        # save the name of the form this handler is connected to
        self.button_name = button_name

    # generate data using GAN
    def handle(self, request, data_pack):
        # handle only in case load button is pressed and there is a button
        if self.button_name in request.form:
            # if necessary data is loaded into the pack
            if 'data_dict' in data_pack:
                # prepare input data
                inp_data = np.copy(data_pack['data_dict']['gen_y'][data_pack['data_dict']["current_gen"]])
                # check if start index or end index are out-of-bounds
                start_indx = int(data_pack['data_dict']["start"])
                end_indx = int(data_pack['data_dict']["end"])
                if start_indx < 0: start_indx = 0
                if end_indx > inp_data.shape[0]: end_indx = inp_data.shape[0]
                print("start_indx = ", start_indx)
                print("end_indx = ", end_indx)
                # replace all values deleted by the user with NaNs
                for i in range(start_indx, end_indx):
                    inp_data[i] = np.NaN
                # replace missing data with specified reference points
                ref_x = data_pack['data_dict']["ref_x"]
                ref_y = data_pack['data_dict']["ref_y"]
                print("ref_x = ", ref_x)
                print("ref_y = ", ref_y)
                for j in range(len(ref_x)):
                    ref_x[j], ref_y[j] = int(ref_x[j]), float(ref_y[j])
                    inp_data[ref_x[j]] = data_pack['data_dict']["ref_y"][j]

                print("inp_data shape ", inp_data.shape)
                print("inp_data= ", inp_data)

                imputed_data = igan_data.impute_data(orig_data=data_pack['data_dict']['orig_y'],
                                                       data_type='.mat',
                                                       inp_data=inp_data,
                                                       miss_rate=0.3,
                                                       batch_size=128,
                                                       hint_rate=0.9,
                                                       alpha=100,
                                                       iterations=10000)

                print("imputed_data shape ", imputed_data.shape)
                print("imputed_data ", imputed_data)
                imputed_data = imputed_data[:, 0]
                print("after imputed_data shape ", imputed_data.shape)
                print("after imputed_data ", imputed_data)
                updates = {"updated_sample": imputed_data}
                return updates
            else:
                return {}
        else:
            return {}


# class to handle data generation
class GenerateDataFormHandler(FormHandler):
    # button_name - name of the button used for form submission
    def __init__(self, button_name):
        # save the name of the form this handler is connected to
        self.button_name = button_name

    # generate data using GAN
    def handle(self, request, data_pack):
        # handle only in case load button is pressed and there is a button
        if self.button_name in request.form:
            # if necessary data is loaded into the pack
            if 'data_dict' in data_pack:
                syn_data = igan_data.gen_data_GAN(data = data_pack['data_dict']['orig_y'],
                                                  data_type= '.mat',
                                                  num_seq=5,
                                                  model_chkpoint=10,
                                                  num_epochs=200,
                                                  out_dir="models/")
                # create timestamps accordingly
                timestamps = np.zeros_like(syn_data)
                for i in range(syn_data.shape[0]):
                    timestamps[i, :] = np.arange(syn_data.shape[1])
                updates = {"gen_data_vals": syn_data,
                           "gen_data_timestamps": timestamps,
                           "change_to_gen": 0}
                return updates
            else:
                return {}
        else:
            return {}


# class to handle data loading
class LoadDataFormHandler(FormHandler):
    # button_name - name of the button used for form submission
    def __init__(self, button_name, input_field_name):
        # save the name of the form this handler is connected to
        self.button_name = button_name
        # save the name of the form this handler is connected to
        self.input_field_name = input_field_name

    # this function loads data in the system and updates the chart
    def handle(self, request, data_pack):
        # handle only in case load button is pressed and there is a button
        if self.button_name in request.form:
            # we don't have to check this - it just confirms that there is a field for input file
            if self.input_field_name not in request.files:
                return {}
            else:
                # get file from the request object
                file = request.files[self.input_field_name]
                # if no file is submitted, do nothing
                if file.filename == '':
                    return {}
                # if file extension is allowed
                elif file and allowed_file(file.filename):
                    # if this file has *.mat extension
                    if file.filename.rsplit('.', 1)[1].lower() == "mat":
                        original_data = scipy.io.loadmat(file.stream)
                        updates = {"orig_data_vals": (ecg_mat_to_np_converter(original_data))[0],
                                   "orig_data_timestamps": (ecg_mat_to_np_converter(original_data))[1],
                                   "change_to_orig": True}
                        return updates
                    # if this file has *.zip extension
                    elif file.filename.rsplit('.', 1)[1].lower() == "zip":
                        # save data file to open it later
                        file.save(os.path.join(UPLOADS_DIR, FILE_NAME))
                        # read data from data file
                        values = igan_data.load_training_data(os.path.join(UPLOADS_DIR, FILE_NAME), '.mat')
                        # delete data
                        os.remove(os.path.join(UPLOADS_DIR, FILE_NAME))
                        # create timestamps accordingly
                        timestamps = np.zeros_like(values)
                        for i in range(values.shape[0]):
                            timestamps[i, :] = np.arange(values.shape[1])
                        updates = {"orig_data_vals": values,
                                   "orig_data_timestamps": timestamps,
                                   "change_to_orig": 0}
                        return updates
                    else:
                        return {}
                else:
                    return {}
        else:
            return {}
