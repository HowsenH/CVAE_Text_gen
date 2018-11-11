import os
import shutil
import torch
import datetime

def load_model_by_name(model, global_step):
    file_path = os.path.join('checkpoints',
                             "CVAE",
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', "CVAE" + str(datetime.datetime.now()).replace(" ", "_"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))


def prepare_writer(model_name, overwrite_existing=False):
    log_dir = os.path.join('logs', model_name)
    save_dir = os.path.join('checkpoints', model_name)
    if overwrite_existing:
        delete_existing(log_dir)
        delete_existing(save_dir)
    writer = None
    return writer

def delete_existing(path):
    if os.path.exists(path):
        print("Deleting existing path: {}".format(path))
        shutil.rmtree(path)
