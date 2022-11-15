# -*- coding: utf-8 -*-
from network import Network
from tmed_patientloader import get_as_dataloader
import wandb
import yaml
import argparse

CONFIG_FILENAME = 'D:/Projects/TMED2/src/config.yaml'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog = 'Launcher for network training',
            description = 'Run training/test with setting from config YAML file',
            epilog = 'Example use: -y config.yaml')

    parser.add_argument("-y", "--yaml", help="Path to the .yaml file")
    args = parser.parse_args()
    
    if args.yaml:
        config_fn = args.yaml
    else:
        config_fn = CONFIG_FILENAME
    
    with open(config_fn) as f:
        config = yaml.safe_load(f)
    
    if config['use_wandb'] and config['mode'] != 'test':
        run = wandb.init(project="tmed2", entity="guangnan", config=config)
    
    if config['mode']=="train":
        dataloader_tr = []
        batch_size = [8, 8, 8, 4, 2, 1]
        max_frames = [4, 8, 16, 32, 64, 128]
        min_frames = [1, 4, 8, 16, 32, 64]
        for i in range(len(batch_size)):
            dataloader_tr.append(get_as_dataloader(config, batch_size[i], 
                                                   'train', 'train', 
                                                   min_frames[i], max_frames[i]))
        #dataloader_tr.append(get_as_dataloader(config, batch_size, split, mode, min_frames=0, max_frames=128))
    testloader = get_as_dataloader(config, batch_size=1, 
                                   split='val', mode='test', min_frames=0, max_frames=128)
    dataloader_te = [testloader]
    
    # initialize the model and any loaded weights with Network
    net = Network(config)
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_te)
    net.test_comprehensive(testloader, mode="test")
        
    if config['use_wandb']:
        wandb.finish()