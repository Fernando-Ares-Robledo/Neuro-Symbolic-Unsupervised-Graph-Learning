#trainer.py
import time
import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
import optuna
from neural import *
class EarlyStop:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_metric: float = -float('inf')
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        if metric > self.best_metric + self.delta:
            self.best_metric = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loaders: tuple,                       
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        early_stopper: EarlyStop,
        beta_KL: float,                       
        gamma_CE: float,                      
        lambdas: tuple[float, float],         
        eta_fuse: float,                     
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.train_loader, self.val_loader = loaders
        self.opt = optimizer
        self.sched = scheduler
        self.stopper = early_stopper

        self.beta  = beta_KL
        self.gamma = gamma_CE
        self.lam_rec, self.lam_lat = lambdas
        self.eta     = eta_fuse

        self.device = device
        self.best_mcc = -float('inf')
        self.best_thr = 0.0

        self.bce_logits = torch.nn.BCEWithLogitsLoss()


    def fit(self, epochs: int, trial: optuna.Trial = None) -> float:
        start = time.time()
        for epoch in range(1, epochs + 1):
        
            mem_usage, t_elapsed = record_memory_usage(
                label="train_epoch",
                record_every=1,
                iteration=epoch
            )
            if mem_usage is not None:
                logging.info(f"[Epoch {epoch}] Memory: {mem_usage:.2f} GB at t={t_elapsed:.1f}s")
            # —————————————————————————————————————————————————

            self.model.train()
            total_loss = 0.0
            for A_list, X_list, y_sym_list, _, _ in self.train_loader:
                self.opt.zero_grad()
                batch_loss = 0.0
                for A, X, y_sym in zip(A_list, X_list, y_sym_list.tolist()):
                    A = A.to(self.device)
                    X = X.to(self.device)
                    y_sym_tensor = torch.tensor([y_sym], dtype=torch.float32, device=self.device)

                    A_hat, mu, logvar, p_hat_logit = self.model(A, X)

                    rec_loss = F.mse_loss(A_hat, A, reduction='mean')
                    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    ce = self.bce_logits(p_hat_logit.unsqueeze(0), y_sym_tensor)

                    loss = self.lam_rec * rec_loss + self.beta * kl + self.gamma * ce
                    loss.backward()
                    batch_loss += loss.item()

                self.opt.step()
                total_loss += batch_loss

            scores, latents, p_hats, s_syms, y_reals = self.evaluate(self.val_loader)
            s_neural = [
                self.lam_rec * s + self.lam_lat * l + (1 - self.lam_rec - self.lam_lat) * p
                for s, l, p in zip(scores, latents, p_hats)
            ]
            s_final = [
                self.eta * sn + (1 - self.eta) * ss
                for sn, ss in zip(s_neural, s_syms)
            ]
            mcc, thr = self._best_mcc_thr(s_final, y_reals)

            if self.sched is not None:
                self.sched.step(mcc)

            if trial is not None:
                trial.report(mcc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if mcc > self.best_mcc:
                self.best_mcc, self.best_thr = mcc, thr
            if self.stopper(mcc):
                break

            logging.info(f"Epoch {epoch:03d} | loss={total_loss:.3f} "
                         f"val_MCC={mcc:.4f} thr={thr:.4f}")

        elapsed = time.time() - start
        logging.info(f"Training completed in {elapsed:.1f}s best_MCC={self.best_mcc:.4f}")
        return self.best_mcc

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        scores, latents, p_hats, y_syms, y_reals = [], [], [], [], []
        for A_list, X_list, y_sym_tensor, y_real_tensor, _ in loader:
            y_syms_batch = y_sym_tensor.tolist()
            y_reals_batch = y_real_tensor.tolist()
            for A, X, y_sym, y_real in zip(A_list, X_list, y_syms_batch, y_reals_batch):
                A = A.to(self.device)
                X = X.to(self.device)

                A_hat, mu, logvar, p_hat_logit = self.model(A, X)

                err = reconstruction_error(A.cpu(), A_hat.cpu())
                scores.append(err)

                lat = latent_distance(mu.cpu(), logvar.cpu())
                latents.append(lat)

                prob = torch.sigmoid(p_hat_logit).item()
                p_hats.append(prob)

                y_syms.append(y_sym)
                y_reals.append(y_real)

        return scores, latents, p_hats, y_syms, y_reals


    @staticmethod
    def _best_mcc_thr(pred_scores, true_labels):
        best_mcc, best_thr = -float('inf'), 0.0
        for thr in sorted(set(pred_scores)):
            preds = [1 if s >= thr else 0 for s in pred_scores]
            m = matthews_corrcoef(true_labels, preds)
            if m > best_mcc:
                best_mcc, best_thr = m, thr
        return best_mcc, best_thr