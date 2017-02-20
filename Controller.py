import json
from nn_train import nn_train
from nn_test import nn_test
from Set_configs import set_configs

# Set whether train or test
train = True
test = True
auto = True

# Set Configs
Config_name = 'Configs.json'
Framework_name = 'VGG-C.json'

# Load configs
with open(Config_name) as json_data:
    config = json.load(json_data)

# Load framework
with open('framework/' + Framework_name) as json_framework:
    framework_file = json.load(json_framework)

# Train
if train:
    # Observe if train finish
    train_state = False

    if auto:
        # Rebuild configs
        config = set_configs(config,Framework_name_in=Framework_name,for_train=True)
        config['Train_option']['model_load_name'] = config['Auto_setting']['pre_trained_model']

    # Load train option
    train_option = config['Train_option']

    # Apply train
    train_state = nn_train(train_option=train_option,framework_file=framework_file)

    # Save recently model direction
    if train_state and config['Train_option']['save_model']:
        if config['Train_option']['pre_trained']:
            recent_num = config['Train_option']['already_train_times'] + config['Train_option']['range_times'] - 1
        else:
            recent_num = config['Train_option']['range_times'] - 1
        recent_name = config['Train_option']['model_save_name'] + '-' + str(recent_num)
        config['Auto_setting']['recent_model_name'] = recent_name
        config['Auto_setting']['keep_prob'] = config['Train_option']['keep_prob']
        with open('Configs.json','w') as json_data:
            json.dump(config, json_data, sort_keys=True, indent=4, separators=(',', ':'))

# Test
if test:
    # Load recently model
    if auto:
        config = set_configs(config, Framework_name_in=Framework_name, for_train=False)
        config['Test_option']['model_load_name'] = config['Auto_setting']['pre_trained_model']
        config['Test_option']['keep_prob'] = config['Auto_setting']['keep_prob']
        config['Test_option']['record_name'] = config['Train_option']['record_path'] + '/' + config['Auto_setting']['pre_trained_model'] + '.xlsx'
    else:
        config['Test_option']['record_name'] = config['Train_option']['record_path'] + '/' + config['Test_option']['model_load_name'] + '.xlsx'

    # Load test option
    test_option = config['Test_option']

    # Apply test
    nn_test(test_option=test_option, framework_file=framework_file)
