import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from cores.models import GraphGlue
from data import (
    load_few_shot_multi_graph_data,
    load_few_shot_single_graph_data,
    load_few_shot_link_graph_data,
    LinkDataLoader
)
from downstream.adapter import GraphGlueAdapter
from downstream.tasks import train_step, eval_step
from utils.checkpoints import (
    load_checkpoint,
    EarlyStopping
)
from utils.logger import create_logger


class AdaptTrainer:
    TASK_CONFIGS = {
        'node_cls': {'label_attr': 'y'},
        'graph_cls': {'label_attr': 'y'},
        'link_cls': {'label_attr': 'edge_label'},
    }
    def __init__(self, configs, logger=None):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger if logger is not None else create_logger(configs.log_path)

        self.start_epoch = 0

        self.task_type = configs.task_type

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)
        os.makedirs("./results", exist_ok=True)

    def train(self):
        loaders, num_classes, num_features = self.get_loaders(self.configs)
        train_loaders = loaders[0]
        val_loaders = loaders[1]
        test_loaders = loaders[2]

        # Train loop
        total_metric = []
        total_macro_f1 = []
        total_test_loss = []
        total_task_loss = []
        total_holo_loss = []
        total_curv_loss = []
        with open(f"./results/{self.configs.data_name}.txt", "a") as f:
            f.write(f"============={self.configs.k_shot}-Shot {self.configs.task_type}=================\n")
            f.write(f"Pretraining Model: {self.configs.pretrained_checkpoint}\n")
        f.close()
        for trial in range(self.configs.num_trials):
            pretrained_model = GraphGlue(self.configs)
            load_checkpoint(self.configs.pretrained_checkpoint, pretrained_model, map_location='cuda')
            model = GraphGlueAdapter(self.configs, num_features, pretrained_model,
                                     self.configs.task_type, num_classes).to(self.device)
            optimizer = Adam(
                model.parameters(),
                lr=self.configs.lr_task,
                weight_decay=self.configs.task_weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.configs.task_epochs,
                eta_min=self.configs.lr_task * 0.01
            )
            early_stopping = EarlyStopping(
                patience=self.configs.patience,
                mode='max',
                delta=0.001,
                checkpoint_dir=self.configs.checkpoint_dir,
                verbose=True
            )
            model.train()
            for epoch in range(self.start_epoch, self.configs.task_epochs):
                epoch_start_time = time.time()
                train_loss, _, train_metric = self._train_epoch(train_loaders[trial], model, optimizer, trial)
                scheduler.step()
                epoch_time = time.time() - epoch_start_time

                self.logger.info(
                    f'Epoch {epoch:03d}/{self.configs.task_epochs} | '
                    f'Train Loss: {train_loss:.6f} | '
                    f'Train {self.configs.metric.upper()}: {train_metric * 100:.2f}% | '
                    f'Time: {epoch_time:.2f}s | '
                    f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                )

                # Evaluation
                if (epoch + 1) % self.configs.eval_interval == 0:
                    val_loss, _, val_metric, _, _ = eval_step(val_loaders[trial], model, self.device,
                                           **AdaptTrainer.TASK_CONFIGS[self.task_type],
                                                  metric=self.configs.metric)
                    self.logger.info(f'Epoch {epoch:03d} | Val {self.configs.metric.upper()}: {val_metric * 100:.2f}%')

                    if early_stopping.step(
                            metric=val_metric,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            config=self.configs
                    ):
                        break

            # Final save
            final_path = os.path.join(self.configs.checkpoint_dir, f'downstream_final_{trial}.pth')
            torch.save({'state_dict': model.state_dict()}, final_path)
            self.logger.info(f"Trial {trial} | Training finished. Final model saved to {final_path}")

            self.logger.info(f"===========Loading best checkpoint from {self.configs.checkpoint_dir}/model_best.pth===========")
            load_checkpoint(f"{self.configs.checkpoint_dir}/model_best.pth", model)
            model.eval()
            test_loss, task_loss, test_metric, holo_loss, curv_loss, test_macro_f1 = eval_step(
                test_loaders[trial],
                model,
                self.device,
                **AdaptTrainer.TASK_CONFIGS[self.task_type],
                metric=self.configs.metric,
                return_macro_f1=True,
            )
            self.logger.info("=====================================================")
            info = f'Trial {trial:02d} | Test {self.configs.metric.upper()}: {test_metric * 100:.2f}%' \
                             f'| Test Loss: {test_loss:.6f} ' \
                             f'| Test Task Loss: {task_loss:.6f}' \
                             f'| Test Holonomy Loss: {holo_loss:.6f} ' \
                             f'| Test Curvature Loss: {curv_loss:.6f} '
            self.logger.info(info)
            self.logger.info("=====================================================")
            total_metric.append(test_metric)
            total_macro_f1.append(test_macro_f1)
            total_test_loss.append(test_loss)
            total_task_loss.append(task_loss)
            total_holo_loss.append(holo_loss)
            total_curv_loss.append(curv_loss)
            with open(f"./results/{self.configs.data_name}.txt", "a") as f:
                f.write(info + "\n")
            f.close()
        info = f'Final Test {self.configs.metric.upper()}: ' \
                f'{np.mean(total_metric) * 100:.2f} \u00B1 {np.std(total_metric) * 100:.2f} % \n' \
               f'Final Test Macro F1: {np.mean(total_macro_f1) * 100:.2f} \u00B1 {np.std(total_macro_f1) * 100:.2f} % \n' \
                f'Final Test Loss: {np.mean(total_test_loss):.6f} \u00B1 {np.std(total_test_loss):.6f} \n' \
               f'Final Test Task Loss: {np.mean(total_task_loss):.6f} \u00B1 {np.std(total_task_loss):.6f} \n' \
               f'Final Test Holonomy Loss: {np.mean(total_holo_loss):.6f} \u00B1 {np.std(total_holo_loss):.6f} \n' \
               f'Final Test Curvature Loss: {np.mean(total_curv_loss):.6f} \u00B1 {np.std(total_curv_loss):.6f}'
        self.logger.info(info)
        with open(f"./results/{self.configs.data_name}.txt", "a") as f:
            f.write(info + "\n")
            f.write("======================================================================\n")
        f.close()

    def _train_epoch(self, train_loader, model, optimizer, trial):
        loss, task_loss, acc, _, _ = train_step(train_loader, optimizer, model, self.device,
                               **AdaptTrainer.TASK_CONFIGS[self.task_type],
                               metric=self.configs.metric)
        return loss, task_loss, acc

    def get_loaders(self, configs):
        train_loaders = []
        val_loaders = []
        test_loaders = []
        if configs.task_type == "node_cls":
            dataset, train_mask, val_mask, test_mask = load_few_shot_single_graph_data(configs, configs.data_name,
                                                                                      configs.k_shot,
                                                                                      configs.num_trials,
                                                                                      configs.num_val)
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            for t in range(configs.num_trials):
                train_loaders.append(DataLoader(dataset[train_mask[:, t]],
                                                batch_size=configs.batch_size,
                                                shuffle=True,
                                                exclude_keys=["edge_attr"]))
                val_loaders.append(DataLoader(dataset[val_mask[:, t]],
                                              batch_size=configs.batch_size,
                                              shuffle=False,
                                              exclude_keys=["edge_attr"]))
                test_loaders.append(DataLoader(dataset[test_mask[:, t]],
                                               batch_size=configs.batch_size,
                                               shuffle=False,
                                               exclude_keys=["edge_attr"]))
        elif configs.task_type == "graph_cls":
            dataset, train_mask, val_mask, test_mask = load_few_shot_multi_graph_data(configs, configs.data_name,
                                                           configs.k_shot, configs.num_trials,
                                                           configs.num_val)
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            for t in range(configs.num_trials):
                train_loaders.append(DataLoader(dataset[train_mask[:, t]],
                                                batch_size=configs.batch_size,
                                                shuffle=True))
                val_loaders.append(DataLoader(dataset[val_mask[:, t]],
                                              batch_size=configs.batch_size,
                                              shuffle=False))
                test_loaders.append(DataLoader(dataset[test_mask[:, t]],
                                               batch_size=configs.batch_size,
                                               shuffle=False))

        elif configs.task_type == "link_cls":
            data, train_sets, val_sets, test_sets = load_few_shot_link_graph_data(configs, configs.data_name,
                                                                 configs.k_shot, configs.num_trials,
                                                                 configs.num_val)
            num_classes = configs.num_way_link
            num_features = data.x.shape[-1]
            for t in range(configs.num_trials):
                train_loaders.append(LinkDataLoader(train_sets[t],
                                                    batch_size=configs.batch_size,
                                                    shuffle=True))
                val_loaders.append(LinkDataLoader(val_sets[t],
                                                  batch_size=configs.batch_size,
                                                  shuffle=False))
                test_loaders.append(LinkDataLoader(test_sets[t],
                                                   batch_size=configs.batch_size,
                                                   shuffle=False))
        else:
            raise NotImplementedError
        return (train_loaders, val_loaders, test_loaders), num_classes, num_features