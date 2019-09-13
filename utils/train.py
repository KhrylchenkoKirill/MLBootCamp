import numpy as np
import pandas as pd

import torch
import copy
import time

from utils.evaluate import calculate_miou, get_iou, calculate_minmax_loss, get_df_miou

from collections import defaultdict

class Trainer:
    
    def __init__(self, model, criterion, optimizer=None, val_criterion=None):
        self.model = model
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), 1e-3)
        self.train_criterion = criterion
        self.val_criterion = criterion if val_criterion is None else val_criterion
        
    def train_step(self, dataloader, cache=False):
        self.model.train()
        self.optimizer.zero_grad()
        
        if cache:
            preds = []
        total_loss = 0.
        for obj, target in dataloader:
            pred = self.model(obj)
            if cache:
                preds.append(pred.detach().numpy())
            loss = self.train_criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item() * len(obj)
        total_loss /= len(dataloader.dataset)
        
        if cache:
            return -total_loss, np.vstack(preds)
        else:
            return -total_loss
        
    def eval_step(self, dataloader, cache=False):
        self.model.eval()
        
        if cache:
            preds = []
        total_loss = 0.
        for obj, target in dataloader:
            with torch.no_grad():
                pred = self.model(obj)
                loss = self.val_criterion(pred, target)
            if cache:
                preds.append(pred.numpy())
            total_loss += loss.item() * len(obj)
        total_loss /= len(dataloader.dataset)
        
        if cache:
            return -total_loss, np.vstack(preds)
        else:
            return -total_loss
        
    def train(
        self, train_dataloader, n_epochs, patience, n_restarts=0, gamma=0.9, val_dataloader=None, verbose=False
    ):
        best_loss = self.eval_step(val_dataloader if val_dataloader is not None else train_dataloader)
        best_weights = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        
        restarts = 0
        losses = defaultdict(list)
        if verbose:
            print('INIT: val_loss: {:.5f}'.format(best_loss))
        n_epoch = 1
        gap = 0
        while n_epoch < n_epochs and restarts <= n_restarts:
            train_loss = self.train_step(train_dataloader)
            if val_dataloader is not None:
                val_loss = self.eval_step(val_dataloader)
            else:
                val_loss = train_loss
            if verbose:
                print('epoch: {:>4}, train_loss: {:.5f}, val_loss: {:.5f}' \
                      .format(n_epoch, train_loss, val_loss), end='')
            if val_loss > best_loss:
                gap = 0
                best_loss = val_loss
                best_epoch = n_epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                if verbose:
                    print(', NEW BEST')
            else:
                gap += 1
                if gap == patience:
                    self.model.load_state_dict(best_weights)
                    gap = 0
                    restarts += 1
                    if restarts <= n_restarts:
                        if verbose:
                            print(', RESTART')
                        for opt in self.optimizer.param_groups:
                            opt['lr'] *= gamma
                else:
                    if verbose:
                        print('')
                    
            losses['train'].append(train_loss)
            losses['val'].append(val_loss)
            n_epoch += 1
        
        if verbose:
            print('\nBEST: epoch: {:>4}, val_loss: {:.5f}' \
                      .format(best_epoch, best_loss))
            
        return best_loss, best_epoch, losses

get_df = lambda df, items: df.loc[df['item'].isin(set(items))].copy().reset_index(drop=True)

class Validator:
    
    def __init__(self, data, targets, n_splits=10, seed=15):
        self.data = data.copy()
        self.targets = targets
        self.r = np.random.RandomState(seed)
        items = self.r.permutation(data['item'].unique()).tolist()
        self.items = items
        self.n_items = len(items)
        self.n_splits = n_splits
        fold_size = self.n_items // self.n_splits

        folds = [items[i * fold_size : (i + 1) * fold_size] for i in range(n_splits)]
        folds += [items[n_splits * fold_size : ]]

        self.splits = [(sum(folds[:i], []) + sum(folds[i + 1:], []), folds[i]) for i in range(n_splits)]
        
    def train_fold(self, fold, make_dataloader, make_model, make_criterion, make_val_criterion, train_params, cache=False):
        tr_idx, val_idx = self.splits[fold]
        tr_dl = make_dataloader(get_df(self.data, tr_idx), self.targets)
        val_dl = make_dataloader(get_df(self.data, val_idx), self.targets)
        model = make_model(tr_dl.dataset)
        criterion, val_criterion = make_criterion(model), make_val_criterion(model)
        trainer = Trainer(model, criterion, val_criterion=val_criterion)
        best_loss, best_epoch, losses = trainer.train(
            train_dataloader=tr_dl, val_dataloader=val_dl, **train_params
        )
        if cache:
            _, val_preds = trainer.eval_step(val_dl, cache=True)
            return best_loss, best_epoch, trainer, val_preds
        else:
            return best_loss, best_epoch, trainer
    
    def train(self, make_dataloader, make_model, make_criterion, make_val_criterion, train_params, cache=False, verbose=False):
        models = []
        losses = []
        if cache:
            oof_preds = []
        for split in range(self.n_splits):
            if verbose:
                start = time.time()
            if cache:
                loss, epoch, trainer, fold_preds = self.train_fold(
                    split, make_dataloader, make_model, make_criterion, make_val_criterion, train_params, cache=cache
                )
                oof_preds.append(fold_preds)
            else:
                loss, epoch, trainer = self.train_fold(
                    split, make_dataloader, make_model, make_criterion, make_val_criterion, train_params
                )
            if verbose:
                print(
                    'fold: {:>2}, loss: {:.4f}, epoch: {:>4}, time: {:.3f}s' \
                      .format(split + 1, loss, epoch, time.time() - start)
                )
            models.append(trainer.model)
            losses.append(loss)
        if verbose:
            print('mean: {:.4f}, std: {:.4f}'.format(np.mean(losses), np.std(losses)))
        if cache:
            preds = pd.DataFrame()
            preds['item'] = self.items
            preds.set_index('item', inplace=True)
            val_ids = [el[1] for el in self.splits]
            for col in ['Xmin', 'Ymin', 'Xbias', 'Ybias']:
                preds[col] = 0.
            for i in range(self.n_splits):
                preds.loc[val_ids[i], ['Xmin', 'Ymin', 'Xbias', 'Ybias']] = oof_preds[i]
            return models, losses, preds.reset_index().sort_values('item').reset_index(drop=True)
        else:
            return models, losses
    
    
class HoldoutValidator:
    
    def __init__(self, data, targets, n_splits=10, seed=15, inner_seed=13, ):
        self.r = np.random.RandomState(seed)
        items = self.r.permutation(data['item'].unique()).tolist()
        n_items = len(items)
        holdout_size = n_items // (n_splits + 1)
        self.holdout_items = items[-holdout_size:]
        self.holdout_df = get_df(data, self.holdout_items)
        self.holdout_minmax_loss = calculate_minmax_loss(self.holdout_df, targets)
        self.crossval_items = items[:-holdout_size]
        self.val = Validator(get_df(data, self.crossval_items), targets, n_splits=n_splits, seed=inner_seed)
        
        #self.val.splits.append(
        #    tuple([self.val.splits[0][0] + self.val.splits[0][1]] * 2)
        #)
        #self.val.n_splits += 1
        
    def train(self, make_dataloader, make_model, make_criterion, make_val_criterion, train_params, cache=False, verbose=False):
        holdout_dl = make_dataloader(self.holdout_df, self.val.targets)
        
        models = []
        losses = []
        holdout_losses = []
        if cache:
            oof_preds = []
            holdout_preds = []
        for split in range(self.val.n_splits):
            if verbose:
                start = time.time()
            if cache:
                loss, epoch, trainer, fold_preds = self.val.train_fold(
                    split, make_dataloader, make_model, make_criterion, make_val_criterion, train_params, cache=cache
                )
                holdout_loss, holdout_pred = trainer.eval_step(holdout_dl, cache)
                oof_preds.append(fold_preds)
                holdout_preds.append(holdout_pred)
            else:
                loss, epoch, trainer = self.val.train_fold(
                    split, make_dataloader, make_model, make_criterion, make_val_criterion, train_params
                )
                holdout_loss = trainer.eval_step(holdout_dl)
            if verbose:
                print(
                    'fold: {:>2}, loss: {:.4f}, epoch: {:>4}, holdout: {:.4f}, time: {:.3f}s' \
                    .format(split + 1, loss, epoch, holdout_loss, time.time() - start)
                )
            models.append(trainer.model)
            losses.append(loss)
            holdout_losses.append(holdout_loss)
        if verbose:
            print('CV. mean: {:.4f}, std: {:.4f}'.format(np.mean(losses), np.std(losses)))
            print('Holdout. mean: {:.4f}, std: {:.4f}, minmax: {:.4f}' \
                  .format(np.mean(holdout_losses), np.std(holdout_losses), self.holdout_minmax_loss))
        if cache:
            preds = pd.DataFrame()
            preds['item'] = sorted(self.val.items)
            preds.set_index('item', inplace=True)
            val_ids = [el[1] for el in self.val.splits]
            for col in ['Xmin', 'Ymin', 'Xbias', 'Ybias']:
                preds[col] = 0.
            for i in range(self.val.n_splits):
                preds.loc[sorted(val_ids[i]), ['Xmin', 'Ymin', 'Xbias', 'Ybias']] = oof_preds[i]
            val_preds = preds.reset_index().sort_values('item').reset_index(drop=True)
            
            preds = pd.DataFrame()
            preds['item'] = sorted(self.holdout_items)
            for col in ['Xmin', 'Ymin', 'Xbias', 'Ybias']:
                preds[col] = 0.
            preds[['Xmin', 'Ymin', 'Xbias', 'Ybias']] = pd.DataFrame(np.mean(holdout_preds, axis=0))
            ho_preds = preds.sort_values('item').reset_index(drop=True)
            return models, losses, holdout_losses, val_preds, ho_preds
        else:
            return models, losses, holdout_losses
