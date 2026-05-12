import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torch_geometric.data import Batch
from cores.models import GraphGlue
from data import (
    load_pretrain_single_graph_data,
    load_pretrain_multi_graph_data,
    Node2GraphDataset)
from utils import (
    search_triangles,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
    create_logger,
    format_time)
import os
import time
import gc
import warnings

warnings.filterwarnings("ignore")


class Pretrainer:
    def __init__(self, configs, logger=None):
        self.final_model_path = None
        self.configs = configs
        self.pretrain_single_graph_data = configs.pretrain_single_graph_data
        self.pretrain_multi_graph_data = configs.pretrain_multi_graph_data
        self.dataset_dict = {k: v for v, k in
                             enumerate(self.pretrain_single_graph_data + self.pretrain_multi_graph_data)}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphGlue(configs).to(self.device)
        self.logger = create_logger(configs.log_path) if logger is None else logger
        self.start_epoch = 0
        self.start_time = None
        self.epoch_times = []

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.lr_pretrain,
            weight_decay=self.configs.pretrain_weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.configs.pretrain_epochs,
            eta_min=self.configs.lr_pretrain * 0.01
        )

        if self.configs.resume_checkpoint:
            latest_check_path = get_latest_checkpoint(self.configs.checkpoint_dir)
            if latest_check_path:
                self.start_epoch = load_checkpoint(latest_check_path, self.model, optimizer, scheduler)
                self.logger.info(f"Resumed from main checkpoint at epoch {self.start_epoch}")
            else:
                self.start_epoch = 0
        else:
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.configs.pretrain_epochs):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(optimizer, epoch)

            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch:03d}/{self.configs.pretrain_epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Time: {epoch_time:.2f}s | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            if (epoch + 1) % self.configs.save_interval == 0 or (epoch + 1) == self.configs.pretrain_epochs:
                checkpoint_path = os.path.join(
                    self.configs.checkpoint_dir,
                    f'pretrain_epoch_{epoch + 1}.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=checkpoint_path
                )

                # Optional
                cleanup_old_checkpoints(self.configs.checkpoint_dir, keep_last=20)

            if (epoch + 1) == self.configs.pretrain_epochs:
                final_model_path = os.path.join(
                    self.configs.checkpoint_dir,
                    'pretrain_final_model.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=final_model_path
                )
                self.logger.info(f'Saved final model: {final_model_path}')
                self.final_model_path = final_model_path

    def _train_epoch(self, optimizer, epoch):
        if self.start_time is None:
            self.start_time = time.time()
        start_epoch_time = time.time()

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        # ===== Mix training for locality =====
        self.logger.info("---------------Mix training for locality----------------")
        loader = self._get_mix_loader()
        loader_start_time = time.time()
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            z, z_tan = self.model(data)
            local_loss = self.model.local_struct_loss(z, z_tan)

            if epoch >= self.configs.warmup_epochs and epoch >= 1:
                proto_loss = self.model.prototype_loss(z, data.data_name_map)
                local_loss += proto_loss

            local_loss.backward()
            optimizer.step()
            self.model.update_prototype(z.detach(), z_tan.detach(), data.data_name_map)

            total_loss += local_loss.item()
            total_batches += 1

            if (batch_idx + 1) % self.configs.log_interval == 0:
                self._log_progress(
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                    dataset_len=len(loader),
                    loss=local_loss.item(),
                    start_loader_time=loader_start_time,
                    batches_done=batch_idx + 1
                )

            del data, z, z_tan
            torch.cuda.empty_cache()
        # gc.collect()

        # ===== Mix training for global distribution =====
        self.logger.info("---------------Mix training for global distribution----------------")
        loader_start_time = time.time()
        for batch_idx, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            z, z_tan = self.model(data)
            with torch.no_grad():
                knn_edge_index, _ = self.model.knn_graph(z, self.configs.knn,
                                                         is_cross=True,
                                                         data_name_map=data.data_name_map,
                                                         is_to_undirected=True)
                triple_paths, _, _ = search_triangles(knn_edge_index,
                                                      self.configs.num_path_samples_global,
                                                      self.configs.path_sample_times_global,
                                                      return_relabel_mapping=True)
            geo_loss = 0.
            for t in range(self.configs.path_sample_times_global):
                geo_loss += self.model.manifold_gluing_loss(z_tan, triple_paths[t])
            geo_loss /= self.configs.path_sample_times_global
            geo_loss.backward()
            optimizer.step()

            total_loss += geo_loss.item()
            total_batches += 1

            if (batch_idx + 1) % self.configs.log_interval == 0:
                self._log_progress(
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                    dataset_len=len(loader),
                    loss=geo_loss.item(),
                    start_loader_time=loader_start_time,
                    batches_done=batch_idx + 1
                )

        del data, z, z_tan
        torch.cuda.empty_cache()

        # ===== Refine manifold structure from locality =====
        self.logger.info("--------------Refine manifold structure from locality---------------")
        for data_name in self.pretrain_single_graph_data:
            self.logger.info(f"===============Refining {data_name} =======================")
            data = load_pretrain_single_graph_data(self.configs, data_name)
            dataset = Node2GraphDataset(data, self.configs.k_hops,
                                        self.configs.num_neighbors,
                                        self.dataset_dict[data_name])
            with torch.no_grad():
                triple_paths, mappings, _ = search_triangles(data.edge_index,
                                                             self.configs.num_path_samples_local,
                                                             self.configs.path_sample_times_local,
                                                             return_relabel_mapping=True)
            loader_start_time = time.time()
            for t in range(self.configs.path_sample_times_local):
                input_node_idx = mappings[t][torch.unique(triple_paths[t])]
                graph = Batch.from_data_list([dataset[i] for i in input_node_idx.cpu().tolist()]).to(self.device)
                optimizer.zero_grad()
                z, z_tan = self.model(graph)
                geo_loss = self.model.manifold_gluing_loss(z_tan, triple_paths[t])
                geo_loss.backward()
                optimizer.step()

                total_loss += geo_loss.item()
                total_batches += 1

                if (t + 1) % self.configs.log_interval == 0:
                    self._log_progress(
                        epoch=epoch,
                        batch_idx=t + 1,
                        dataset_len=self.configs.path_sample_times_local,
                        loss=geo_loss.item(),
                        start_loader_time=loader_start_time,
                        batches_done=t + 1
                    )
                del z, z_tan, graph
                torch.cuda.empty_cache()

            del data, dataset, triple_paths, mappings
            torch.cuda.empty_cache()
            gc.collect()

        for data_name in self.pretrain_multi_graph_data:
            self.logger.info(f"===============Refining {data_name} =======================")
            dataset = load_pretrain_multi_graph_data(self.configs, data_name, self.dataset_dict[data_name])
            loader = DataLoader(dataset, batch_size=self.configs.batch_size, shuffle=True,
                                num_workers=self.configs.num_workers)
            loader_start_time = time.time()
            for batch_idx, data in enumerate(loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                z, z_tan = self.model(data)
                knn_edge_index, _ = self.model.knn_graph(z, self.configs.knn, is_to_undirected=True)
                triple_paths, _, _ = search_triangles(knn_edge_index,
                                                      self.configs.num_path_samples_global,
                                                      self.configs.path_sample_times_global,
                                                      return_relabel_mapping=True)
                geo_loss = 0.
                for t in range(self.configs.path_sample_times_global):
                    geo_loss += self.model.manifold_gluing_loss(z_tan, triple_paths[t])
                geo_loss /= self.configs.path_sample_times_global
                geo_loss.backward()
                optimizer.step()

                total_loss += geo_loss.item()
                total_batches += 1

                if (batch_idx + 1) % self.configs.log_interval == 0:
                    self._log_progress(
                        epoch=epoch,
                        batch_idx=batch_idx + 1,
                        dataset_len=len(loader),
                        loss=geo_loss.item(),
                        start_loader_time=loader_start_time,
                        batches_done=batch_idx + 1
                    )

                del data, z, z_tan
            del loader, dataset
            # gc.collect()

        # Log
        self._log_epoch_summary(epoch, start_epoch_time)
        self._update_epoch_time(epoch, start_epoch_time)

        return total_loss / max(1, total_batches)

    def _log_progress(self, epoch, batch_idx, dataset_len, loss, start_loader_time,
                      batches_done):
        current_time = time.time()

        batches_remaining = dataset_len - batches_done
        recent_avg_batch_time = (current_time - start_loader_time) / batches_done
        loader_remaining_time = recent_avg_batch_time * batches_remaining

        if len(self.epoch_times) == 0:
            elapsed_total = current_time - self.start_time
            avg_epoch_time = elapsed_total / (epoch + 1)
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))

        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        self.logger.info(
            f'Epoch {epoch} | Batch {batch_idx}/{dataset_len} | '
            f'Loss: {loss:.6f} | '
            f'Loader ETA: {format_time(loader_remaining_time)} | '
            f'Total ETA: {format_time(total_remaining_time)}'
        )

    def _log_epoch_summary(self, epoch, start_epoch_time):
        if len(self.epoch_times) == 0:
            avg_epoch_time = time.time() - self.start_time
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))
        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        epoch_duration = time.time() - start_epoch_time

        self.logger.info(
            f'Epoch {epoch} completed in {format_time(epoch_duration)}. '
            f'Estimated remaining training time: {format_time(total_remaining_time)} '
            f'({remaining_epochs} epochs left)'
        )

    def _update_epoch_time(self, epoch, start_epoch_time):
        epoch_duration = time.time() - start_epoch_time
        self.epoch_times.append(epoch_duration)

    def _get_mix_loader(self):
        datasets = []

        for data_name in self.pretrain_single_graph_data:
            data = load_pretrain_single_graph_data(self.configs, data_name)
            datasets.append(Node2GraphDataset(data,
                                              self.configs.k_hops,
                                              self.configs.num_neighbors,
                                              self.dataset_dict[data_name])
                            )

        for data_name in self.pretrain_multi_graph_data:
            datasets.append(
                load_pretrain_multi_graph_data(self.configs,
                                               data_name,
                                               self.dataset_dict[data_name])
            )

        weights = []
        for d in datasets:
            n = len(d)
            weights.extend([1.0 / n] * n)
        weights = torch.tensor(weights)

        datasets = ConcatDataset(datasets)
        sampler = WeightedRandomSampler(weights, num_samples=len(datasets), replacement=True)
        loader = DataLoader(datasets, batch_size=self.configs.batch_size, sampler=sampler,
                            num_workers=self.configs.num_workers, persistent_workers=False)
        return loader