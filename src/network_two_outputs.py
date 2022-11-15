'''
WARNING: THIS FILE IS INCOMPLETE. PROOF OF CONCEPT THAT ISNT TESTED

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

from get_model import GatedAttentionClassifier, AveragingClassifier
from losses import view_mask_loss

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, config, classind_to_model, taxonomy, batch_limit):
        """Initialize configuration."""
        super().__init__()
        #torch.autograd.set_detect_anomaly(True)
        
        self.config = config
        self.epsilon = 1e-8
        self.batch_size = config['batch_size']
        if config['batch_size'] > batch_limit:
            self.batch_accum_iters = config['batch_size'] // batch_limit
        else:
            self.batch_accum_iters = 1

        self.num_classes = config['num_classes']
        self.view_lambda = config['view_loss_weight']
        
        self.model = GatedAttentionClassifier(config['embedding_dim'], 
                                              config['dense_hidden_dim'],
                                              self.num_classes,
                                              config['encoder_pretrained'],
                                              config['use_gated'])
        
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

    
    # TODO is NxB or BxN better I really don't know arghhhhhh
    def _get_network_output(self, data):
        imgs = data[0] # len-N list of Bx3xHxW
        targets_AS = data[1] # B
        targets_view = data[2] # len-N list of B
        mask = data[3] # len-N list of B
        # Transfer data from CPU to GPU.
        if self.use_cuda:
            imgs = [i.cuda() for i in imgs]
            targets_AS = targets_AS.cuda()
            targets_view = [tv.cuda() for tv in targets_view]
            mask = [m.cuda() for m in mask]
        
        # get the logit output (NxBxC(1))
        logits_AS, logits_view, attns = self.model(imgs)
        mask = torch.stack(mask) # NxB
        return imgs, targets_AS, logits_AS, attns, targets_view, logits_view, mask
                
    # logits input to be NxBxC and NxBx1 attn, Bx1 targets
    def _get_loss(self, logits, targets, attns):
        norm = torch.softmax(attns, dim=0) # NxBx1 (normalized across N dimension)
        # aggregating logits based on attention weighting
        weighted_logits = norm * logits # NxBxC
        agg_logits = torch.sum(weighted_logits, dim=0) # BxC
        loss = F.cross_entropy(agg_logits, targets)
        return loss
    
    #view mask loss with NxBxC, N
    def _get_view_loss(self, logits, targets, mask):
        N, B, C = logits.size()
        logits_unrolled = logits.contiguous.view(N*B, C)
        return loss
        
    # obtain summary statistics of
    # argmax, max_percentage, entropy for each function
    # expects logits input to be BxC
    def _get_prediction_stats(self, logits):
        prob = F.softmax(logits, dim=1)
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        return argm, max_percentage, entropy
        
    def _epoch(self, mode, loader):
        losses_AS, losses_view = [], []
        gt_AS, gt_view = [], []
        pred_AS, pred_view = [], []
        # nc = self.num_classes
        # conf = np.zeros((nc, nc))
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        for i, data in enumerate(tqdm(loader)):
            imgs, y_AS, logits_AS, attn, y_view, logits_view, mask = self._get_network_output(data)
            
            loss_AS = self._get_loss(logits_AS, y_AS, attn, mask)
            loss_view = self._get_view_loss(logits_view, y_view, mask)
            loss = loss_AS + self.lambda_view * loss_view
            
            losses_AS.append(loss_AS)
            losses_view.append(loss_view)
            
            argm, _, _ = self._get_prediction_stats(logits_AS)
            gt_AS.append(y_AS)
            pred_AS.append(argm)

            argm, _, _ = self._get_view_prediction_stats(logits_view, mask)
            
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

        loss_avg_AS = torch.mean(torch.stack(losses_AS)).item()
        gt_AS = torch.cat(gt_AS)
        pred_AS = torch.cat(pred_AS)
        acc_AS = sum(gt_AS == pred_AS)/len(gt_AS)
        f1_AS = f1_score(gt_AS.cpu(), pred_AS.cpu(), average='macro')
        
        loss_avg_view = torch.mean(torch.stack(losses_view)).item()
        acc_view = sum(gt_view == pred_view)/len(gt_view)
        
        return loss_avg_AS, loss_avg_view, acc_AS, f1_AS, acc_view
    
    def train(self, loader_tr, loader_va):
        """Training pipeline."""
        print("Training model with total batch_size {} via {}x batch accumulation"
              .format(self.batch_size, self.batch_accum_iters))
        
        # load the model from an external load file, if applicable
        if self.model_load_dir != None:
            checkpoint = os.path.join(self.model_load_dir, "checkpoint.pth")
            self._restore(checkpoint)
        
        best_va_acc = 0.0 # Record the best validation metrics.
        for epoch in range(self.config['num_epochs']):
            loss_AS, loss_view, acc_AS, f1_AS, acc_view = self._epoch('train', loader_tr)
            print(
                "Epoch: %3d, tr L/acc/f1: %.5f/%.3f/%.3f (AS), L/acc: %.5f/%.3f (view)"
                % (epoch, loss_AS, acc_AS, f1_AS, loss_view, acc_view)
            )
            # for validation: if torch.no_grad isn't called and loss.backward() also
            # isn't used, the GPU will keep accumulating the gradient which eventually 
            # cause an OOM error.
            # thus the torch.no_grad() before evaluation is mandatory.
            # if there's a more elegant way around this, let me know!
            with torch.no_grad():
                vl_AS, vl_view, v_acc_AS, v_f1_AS, v_acc_view = self._epoch('val', loader_va)
            
            # Save model every epoch.
            self._save(self.checkpts_file)
            if self.config['use_wandb']:
                wandb.log({"tr_loss_AS":loss_AS, "val_loss_AS":vl_AS,
                           "tr_loss_view":loss_view, "val_loss_view": vl_view,
                           "tr_acc_AS":acc_AS, "tr_f1_AS":f1_AS,
                           "val_acc_AS":v_acc_AS, "val_f1_AS":v_f1_AS,
                           "tr_acc_view":acc_view, "val_acc_view":v_acc_view})

            # Early stopping strategy.
            if v_acc_AS > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = v_acc_AS
                self._save(self.bestmodel_file)
            
            print(
                "Epoch: %3d, val L/acc/f1: %.5f/%.3f/%.3f (AS), L/acc: %.5f/%.3f (view)"
                % (epoch, vl_AS, v_acc_AS, v_f1_AS, vl_view, v_acc_view)
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
        
        collector = {}
        collector_keys = ['path','study_id','y_view','p_view','conf_view','ent_view',
                          'y_AS', 'p_AS', 'conf_AS', 'ent_AS', 'attn']
        
        for data in tqdm(loader):
            s = {}
            s['study_id'] = data[4]
            s['path'] = data[5]
                
            # collect the model prediction info
            imgs, s['y_AS'], logits_AS, s['attn'], s['y_view'], logits_view, s['mask'] = self._get_network_output(data)
            s['p_AS'], s['conf_AS'], s['ent_AS'] = self._get_prediction_stats(logits_AS)
            s['p_view'], s['conf_view'], s['ent_view'] = self._get_prediction_stats(logits_view)
            # y_v_arr.append(y_view.cpu().numpy()[0])
            # pred_v_arr.append(argm.cpu().numpy()[0])
            # max_v_arr.append(max_p.cpu().numpy()[0])
            # att_AS_arr.append(attn.cpu().numpy()[0])
                
        # compile the information into a dictionary
        d = {'path':collector['path'], 'study_id':collector['study_id']}
        
        # save the dataframe
        df = pd.DataFrame(data=d)
        test_results_file = os.path.join(self.log_dir, mode+".csv")
        df.to_csv(test_results_file)