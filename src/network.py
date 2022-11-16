# wrapper module for compcars training

import os
import yaml

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from tqdm import tqdm
import wandb

from get_model import GatedAttentionClassifier, count_module_params
from tmed_patientloader import get_label_weights

RESOLUTION = 112

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, config):
        """Initialize configuration."""
        super().__init__()
        #torch.autograd.set_detect_anomaly(True)
        
        self.config = config
        self.epsilon = 1e-8
        if config.get('batch_accum_iters'):
            self.batch_accum_iters = config['batch_accum_iters']
        else:
            self.batch_accum_iters = 1

        self.num_classes = config['num_classes']
        self.label_weights = torch.tensor(get_label_weights(config['label_scheme_name']))
        
        self.model = GatedAttentionClassifier('WideResnet-28-2',
                                              RESOLUTION,
                                              config['embedding_dim'], 
                                              config['dense_hidden_dim'],
                                              self.num_classes,
                                              config['use_attention'],
                                              config['encoder_pretrained'],
                                              config['use_gated'])
        count_module_params(self.model, 'classifier')
        if config['use_cuda']:
            if torch.cuda.is_available():
                self.use_cuda = True
                print('Using GPU acceleration: GPU{}'.format(torch.cuda.current_device()))
            else:
                self.use_cuda = False
                print('Warning: CUDA is requested but not available. Using CPU')
        else:
            self.use_cuda = False
            print('Using CPU')
            
        if self.use_cuda:
            self.model = self.model.cuda()
            self.label_weights = self.label_weights.cuda()
            
        self.optimizer, self.scheduler = self.configure_optimizers()

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory where we save states such as trained model.
        if self.config['mode'] == 'test':
            # find the location of the weights and use the same directory as the log directory
            if self.config['model_load_dir'] is None:
                raise AttributeError('For test-only mode, please specify the model state_dict folder')
            self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
        else: # training
            if self.config['use_wandb']:
                # use the wandb folder name as the new directory
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
            else:
                # use the base folder as the save folder
                self.log_dir = self.config['log_dir']
        # if self.config['use_wandb']:
        #     if self.config['mode'] == 'test':
        #         # use the previous directory as the log directory
        #         if self.config['model_load_dir'] is None:
        #             raise AttributeError('For test-only mode, please specify the model state_dict folder')
        #         self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
        #     else:
        #         # after this switch to the new log directory
        #         self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        # else:
        #     self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
            
        # We can save a copy of the training configurations for future reference
        if self.config['mode'] != 'test':
            yaml_filename = os.path.join(self.log_dir, "config.yaml")
            with open(yaml_filename, 'w') as file:
                yaml.dump(self.config, file)
                
        # File for loading a pretrained model before training starts, if applicable
        self.model_load_dir = self.config['model_load_dir']
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        self.optimizer.load_state_dict(load_res["optimizer"])
            
    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.config['lr'])
        vb = (self.config['mode'] != 'test')
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=self.config['exp_gamma'], verbose=vb)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=self.config['num_epochs'], verbose=vb)
        return optimizer, scheduler
    
    def _get_network_output(self, data):
        imgs = data[0] # len-N list of Bx3xHxW
        targets = data[1] # B
        #view = data[2]
        mask = data[3] # len-N list of B
        # Transfer data from CPU to GPU.
        if self.use_cuda:
            imgs = [i.cuda() for i in imgs]
            targets = targets.cuda()
            mask = [m.cuda() for m in mask]
        
        # get the logit output (NxBxC(1))
        logits, attns = self.model(imgs)
        mask = torch.stack(mask) # NxB
        return imgs, targets, logits, attns, mask
                
    # logits input to be NxBxC and NxBx1 attn, Bx1 targets
    def _get_loss(self, logits, targets, attns=None):
        if attns is not None:
            norm = torch.softmax(attns, dim=0) # NxBx1 (normalized across N dimension)
            # aggregating logits based on attention weighting
            weighted_logits = norm * logits # NxBxC
            agg_logits = torch.sum(weighted_logits, dim=0) # BxC
        else:
            agg_logits = torch.mean(logits, dim=0)
        loss = F.cross_entropy(agg_logits, targets, weight=self.label_weights)
        return loss
        
    # obtain summary statistics of
    # argmax, max_percentage, entropy for each function
    # expects logits input to be BxC
    def _get_prediction_stats(self, logits):
        prob = F.softmax(logits, dim=1)
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        return argm, max_percentage, entropy
    
    # similar to get_prediction_stats but uses attention-based aggregation
    # (or averaging) on NxBxC logits
    def _get_prediction_stats_agg(self, logits, attns=None):
        if attns is not None:
            norm = torch.softmax(attns, dim=0) # NxBx1 (normalized across N dimension)
            # aggregating logits based on attention weighting
            weighted_logits = norm * logits # NxBxC
            agg_logits = torch.sum(weighted_logits, dim=0) # BxC
        else:
            agg_logits = torch.mean(logits, dim=0)
        return self._get_prediction_stats(agg_logits)
        
        
    def _epoch(self, mode, loaders):
        losses = []
        gt = []
        pred = []
        # nc = self.num_classes
        # conf = np.zeros((nc, nc))
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
            
        for loader_idx in np.random.permutation(len(loaders)):
            loader = loaders[loader_idx]
            for i, data in enumerate(tqdm(loader)):
                imgs, targets, logits, attns, mask = self._get_network_output(data)
                
                loss = self._get_loss(logits, targets, attns)
                
                losses.append(loss)
                
                argm, _, _ = self._get_prediction_stats_agg(logits, attns)
                gt.append(targets)
                pred.append(argm)
                
                if mode == 'train':
                    # calculate the gradient with respect to the input
                    loss.backward()
                    # use the gradient accumulation trick to increase batch size
                    # see https://discuss.pytorch.org/t/how-to-increase-the-batch-size-but-keep-the-gpu-memory/26275
                    if (i+1) % self.batch_accum_iters == 0 or (i+1) == len(loader):
                        # Update the parameters according to the gradient.
                        self.optimizer.step()
                        # Zero the parameter gradients in the optimizer
                        self.optimizer.zero_grad() 

        loss_avg = torch.mean(torch.stack(losses)).item()
        gt = torch.cat(gt)
        pred = torch.cat(pred)
        acc = sum(gt == pred)/len(gt)
        f1 = f1_score(gt.cpu(), pred.cpu(), average='macro')
        
        return loss_avg, acc, f1
    
    def train(self, loaders_tr, loaders_va):
        """Training pipeline."""
        print("Training model via {}x batch accumulation".format(self.batch_accum_iters))
        
        # load the model from an external load file, if applicable
        if self.model_load_dir is not None:
            checkpoint = os.path.join(self.model_load_dir, "checkpoint.pth")
            self._restore(checkpoint)
        
        best_va_acc = 0.0 # Record the best validation metrics.
        for epoch in range(self.config['num_epochs']):
            loss, acc, f1 = self._epoch('train', loaders_tr)
            print(
                "Epoch: %3d, tr L/acc/f1: %.5f/%.3f/%.3f"
                % (epoch, loss, acc, f1)
            )
            # for validation: if torch.no_grad isn't called and loss.backward() also
            # isn't used, the GPU will keep accumulating the gradient which eventually 
            # cause an OOM error.
            # thus the torch.no_grad() before evaluation is mandatory.
            # if there's a more elegant way around this, let me know!
            with torch.no_grad():
                val_loss, val_acc, val_f1 = self._epoch('val', loaders_va)
            
            # Save model every epoch.
            self._save(self.checkpts_file)
            if self.config['use_wandb']:
                wandb.log({"tr_loss":loss, "val_loss":val_loss,
                           "tr_acc":acc, "tr_f1":f1,
                           "val_acc":val_acc, "val_f1":val_f1})

            # Early stopping strategy.
            if val_acc > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = val_acc
                best_va_f1 = val_f1
                self._save(self.bestmodel_file)
            
            print(
                "Epoch: %3d, val L/acc/f1: %.5f/%.3f/%.3f, top acc/f1: %.3f/%.3f"
                % (epoch, val_loss, val_acc, val_f1, best_va_acc, best_va_f1)
            )
            
            # modify the learning rate
            self.scheduler.step()
    
    @torch.no_grad()
    def test_comprehensive(self, loader, mode="test"):
        """Logs the network outputs in dataloader
        computes per-car preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()
        
        d = {}
        collector_keys = ['path','study_id','gt','pred','conf','ent','attn']
        for c in collector_keys:
            d[c] = []
        
        for data in tqdm(loader):
            study_id = data[4]
            path = data[5]
                
            # collect the model prediction info, one line per image
            imgs, gt, logits, attns, mask = self._get_network_output(data)
            for n in range(logits.size(0)):
                if mask[n] == 0:
                    continue
                pred, conf, ent = self._get_prediction_stats(logits[n])
                d['path'].append(path[n][0])
                d['study_id'].append(study_id[0])
                d['gt'].append(gt.cpu().numpy()[0])
                d['pred'].append(pred.cpu().numpy()[0])
                d['conf'].append(conf.cpu().numpy()[0])
                d['ent'].append(ent.cpu().numpy()[0])
                if attns is not None:
                    d['attn'].append(attns.cpu().numpy()[n].squeeze())
                else:
                    d['attn'].append(1/logits.size(0))
        
        # save the dataframe
        df = pd.DataFrame(data=d)
        test_results_file = os.path.join(self.log_dir, mode+".csv")
        df.to_csv(test_results_file)
        