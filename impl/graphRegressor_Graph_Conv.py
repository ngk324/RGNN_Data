import torch
import torch.nn.functional as F
import os
import datetime
import time
from sklearn.metrics import mean_squared_error

predict_fn = lambda output: output.detach().cpu()  # Return the output as-is

def pause():
    input("Paused. Press Any Key to Continue...")

def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), "w+")
    test_log = open(os.path.join(log_dir, (test_name + "_test")), "w+")
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), "w+")

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + "\n")
        f.write("#epoch \t split \t loss \t mse \t avg_epoch_time \t avg_epoch_cost \n")

    return train_log, test_log, valid_log


class modelImplementation_GraphRegressor(torch.nn.Module):
    def __init__(self, model, lr, criterion, device="cpu"):
        super(modelImplementation_GraphRegressor, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.loss = torch.nn.MSELoss()
        self.L1loss = torch.nn.L1Loss()
        self.device = device
        self.out_fun = torch.nn.Identity()
        
        self.set_optimizer()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def set_optimizer(self, weight_decay=1e-4):
        train_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # self.optimizer = torch.optim.AdamW(
        #     train_params, lr=self.lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        # )
        self.optimizer = torch.optim.RMSprop(
            train_params, lr=self.lr, weight_decay=weight_decay, #betas=(0.9, 0.999)
        )

    def train_test_model(
        self,
        split_id,
        loader_train,
        loader_test,
        loader_valid,
        n_epochs,
        test_epoch,
        aggregator=None,
        test_name="",
        log_path=".",
    ):

        train_log, test_log, valid_log = prepare_log_files(
            test_name + "--split-" + str(split_id), log_path
        )

        train_loss, n_samples = 0.0, 0
        best_val_mse, best_val_loss, best_val_acc = 100.0, 100.0, 0.0
        epoch_time_sum = 0
        for epoch in range(n_epochs):
            self.model.train()
            epoch_start_time = time.time()
            for batch in loader_train:
                data = batch.to(self.device)
                self.optimizer.zero_grad()                
                out, _ = self.model(data, hidden_layer_aggregator=aggregator)
                
                # mse_threshold = 0.2
                loss = self.loss(out, data.y)
                # if loss.item() < mse_threshold:
                #     loss = self.L1loss(out, data.y)             
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scheduler.step(loss.item())
                self.optimizer.step()
                
                train_loss += loss.item() * len(out)
                n_samples += len(out)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time
            if epoch % test_epoch == 0:
                print("epoch: ", epoch, "; loss: ", train_loss / n_samples)

                (mse_train_set, loss_train_set, acc_train_set) = self.eval_model(loader_train, aggregator)
                (mse_test_set, loss_test_set, acc_test_set) = self.eval_model(loader_test, aggregator)
                (mse_valid_set, loss_valid_set, acc_valid_set) = self.eval_model(loader_valid, aggregator)

                print(
                    f"split {split_id}:\n"
                    f"\ttrain: mse={mse_train_set:.4f}, loss={loss_train_set:.4f}, acc={acc_train_set:.4f}\n",
                    f"\ttest: mse={mse_test_set:.4f}, loss={loss_test_set:.4f}, acc={acc_test_set:.4f}\n",
                    f"\tvalidation: mse={mse_valid_set:.4f}, loss={loss_valid_set:.4f}, acc={acc_valid_set:.4f}"
                )
                print("------")
                # pause()
                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_train_set,
                        mse_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples,
                    )
                )

                train_log.flush()

                test_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_test_set,
                        mse_test_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples,
                    )
                )

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_valid_set,
                        mse_valid_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples,
                    )
                )

                valid_log.flush()

                if mse_valid_set < best_val_mse or loss_valid_set < best_val_loss or acc_valid_set > best_val_acc:
                    best_val_mse = mse_valid_set
                    best_val_loss = loss_valid_set
                    best_val_acc = acc_valid_set
                    self.save_model(
                        test_name=test_name,
                        path=log_path,
                        epoch=epoch,
                        split=split_id,
                        loss=self.criterion,
                    )
                train_loss, n_samples = 0, 0
                epoch_time_sum = 0
                
            if epoch % 80 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"Grad of {name}: {param.grad.norm()}")
                print(torch.abs(out - data.y))
                # pause()
            # pause()

    def save_model(
        self, test_name="test", path="./", epoch=None, split=None, loss="None"
    ):
        if epoch is not None and epoch is not None:
            test_name = (
                test_name + "--split_" + str(split) + "--epoch_" + str(epoch) + ".tar"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "split": split,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join(path, test_name),
            )
        else:
            torch.save(self, self.model.state_dict(), os.path.join(path, test_name))

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return self.model, self.optimizer, epoch, loss

    def eval_model(self, loader, aggregator):
        self.model.eval()
        n_samples = 0
        loss = 0.0
        acc = 0.0
        mse = 0.0
        n_batches = 0
        for batch in loader:
            data = batch.to(self.device)
            
            value_out, validity_out = self.model(data, aggregator)
            
            pred = predict_fn(value_out)
            
            validity_target = (data.y > 1e-4).float()
            validity_pred = (validity_out > 0).float()

            validity_acc = torch.sum(1-torch.abs(validity_target - validity_pred)) / validity_pred.numel()
            acc += validity_acc
            
            n_samples += len(value_out)
            n_batches += 1

            mse_ = self.loss(data.y.cpu(), pred)
            mse += mse_
            loss += mse_

        return mse / n_samples, loss / n_samples, 100 * acc / n_batches
