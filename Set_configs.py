import os
import os.path
import xlsxwriter


def find_recent_two_files(direction):
    # This function is for find two of the most recently files
    # Set direction
    dirs = os.path.expanduser(direction)

    # Get files list
    files = [os.path.join(dirs, f) for f in os.listdir(dirs)]

    # Initialize store variable
    max_time1 = 0.1
    max_time2 = 0.0
    max_state1 = -1
    max_state2 = -1

    # Using cycle to find the most two recent files
    for i in range(len(files)):
        cur_time = os.path.getmtime(files[i])
        if cur_time > max_time2:
            if cur_time > max_time1:
                max_state2 = max_state1
                max_state1 = i
                max_time2 = max_time1
                max_time1 = cur_time
            else:
                max_state2 = i
                max_time2 = cur_time

    return max_state1, max_state2


def set_configs(config,Framework_name_in, for_train = True, Auto = True):
    # This function is design for set dirction

    # Choose the framework name before '.json'
    try:
        Framework_name = Framework_name_in[0:Framework_name_in.find('.')]
    except:
        Framework_name = Framework_name_in

    # Try if the model save files is exist
    try:
        os.listdir(config['Train_option']['model_path'])
    except:
        os.makedirs(config['Train_option']['model_path'])

    # Auto set files name and dir
    if Auto and for_train:
        config['Train_option']['model_save_name'] = Framework_name + '/' + 'Saved_Model'

    # Make model saved files
    model_save_name = config['Train_option']['model_save_name']
    model_dir = config['Train_option']['model_path'] + '/' + model_save_name[0:str(model_save_name).find('/')]

    # For test the model_exist must be true
    if for_train:
        # Try if the model save files is exist
        try:
            os.listdir(model_dir)
        except:
            os.makedirs(model_dir)
            model_exsit = False
        else:
            model_exsit = True
    else:
        model_exsit = True

    # Find the most recent model
    if model_exsit:
        # Find pre_trained model
        max_1 , max_2 = find_recent_two_files(model_dir)

        # Check if the model exist
        if max_1 == -1 and max_2 == -1:
            config['Auto_setting']['pre_trained_model'] = None
        else:
            # List model direction
            model_list = os.listdir(model_dir)

            # Check the mostly file is the checkpoint or model, and find the num in the name of the model files
            num_index = model_list[max_1][model_list[max_1].find('.') - 1]
            model_load_name = model_list[max_1][0:model_list[max_1].find('.')]
            if not num_index.isdigit():
                num_index = model_list[max_2][model_list[max_2].find('.') - 1]
                model_load_name = model_list[max_1][0:model_list[max_2].find('.')]

            # Set load model
            config['Auto_setting']['pre_trained_model'] = Framework_name + '/' + model_load_name

            # Set already trained times
            if for_train:
                config['Train_option']['already_train_times'] = int(num_index) + 1
                if not config['Train_option']['pre_trained']:
                    config['Train_option']['already_train_times'] = 0

    # Protect if not for train
    if not for_train:
        model_save_name = Framework_name + '/' + 'Saved_Model'

    # Join record path
    record_dir = config['Train_option']['record_path'] + '/' + model_save_name[0:str(model_save_name).find('/')]

    # Set record save directions
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)

    # While train create the record files
    if for_train:
        # Create excel for every train range
        for i in range(config['Train_option']['range_times']):
            record_file_name = config['Train_option']['record_path'] + '/' + config['Train_option']['model_save_name'] + '-' + str(config['Train_option']['already_train_times'] + i) + '.xlsx'
            if not os.path.isfile(record_file_name):
                xlsxwriter.Workbook(record_file_name)

    # If for test set record files name
    if not for_train:
        config['Auto_setting']['record_name'] = config['Train_option']['record_path'] + '/' + config['Auto_setting']['pre_trained_model'] + '.xlsx'
        record_file_name = config['Auto_setting']['record_name']

        # If excel doesn't exist, create file
        if not os.path.isfile(record_file_name):
            xlsxwriter.Workbook(record_file_name)

    return config




