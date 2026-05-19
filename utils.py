import os
import socket
import warnings
from datetime import datetime, timezone, timedelta
import numpy as np
import torch

try:
    import faiss
except ImportError:
    print("Faiss missing, some functionality will not work")

warnings.filterwarnings('ignore', category=UserWarning, message='.*TF32.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*tensorfloat-32.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.backends.cuda')


def mean_squared_error(x, y):
    return float(((x - y) ** 2).sum(1).mean())

def pairwise_distances(a, b):
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T

def compute_batch_distances(a, b):
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return (
        anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))
    )


def assign_batch_multiple(x, zqs):
    bs, d = x.shape
    bs2, K, d2 = zqs.shape
    assert bs == bs2 and d == d2

    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    zqs_norm = (zqs ** 2).sum(dim=2)
    xz = torch.bmm(x.unsqueeze(1), zqs.transpose(1, 2)).squeeze(1)
    L2distances = x_norm + zqs_norm - 2 * xz
    
    idx = torch.argmin(L2distances, dim=1)
    quantized = zqs[torch.arange(bs, device=zqs.device), idx]
    return idx, quantized


def assign_to_codebook(x, c, bs=16384):
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        dis = pairwise_distances(x, c)
        return dis.argmin(1)

    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c**2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i : i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j : j + bs] - 2 * x[i : i + bs] @ c[j : j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin = dmini
                imin = imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask] = dmini[mask]
                imin[mask] = imini[mask] + j
        res[i : i + bs] = imin
    return res


def find_free_port(start_port=50000, max_attempts=100):
    for i in range(max_attempts):
        port = start_port + (hash(os.getpid()) % 1000) + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    for _ in range(max_attempts):
        port = np.random.randint(50000, 65000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port after {} attempts".format(max_attempts * 2))


def fix_random_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            elif hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True


def get_model_state_dict(model):
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    is_ddp = isinstance(model, DDP)
    actual_model = model.module if is_ddp else model
    
    model_to_save = actual_model
    if hasattr(actual_model, '_orig_mod'):
        model_to_save = actual_model._orig_mod
    
    return {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}


def save_checkpoint(model, checkpoint_path, epoch, scheduler=None, loss=None, val_loss=None):
    state_dict = get_model_state_dict(model)
    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
    }
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler
    if loss is not None:
        checkpoint['loss'] = loss
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, checkpoint_path, device, rank=0):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, False
    
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', {})
    
    if 'codebook0.weight' not in state_dict:
        if rank == 0:
            print(f"ERROR: Missing critical key 'codebook0.weight' in checkpoint")
        raise KeyError("Missing critical key 'codebook0.weight' in checkpoint")
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"Warning: {len(missing_keys)} missing keys in checkpoint")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
    except Exception as e:
        if rank == 0:
            print(f"ERROR loading checkpoint: {e}")
        raise
    
    return checkpoint, True


class Scheduler:
    def __init__(self, lr0, filename, lr_patience=10):
        self.lr = lr0
        self.filename = filename
        self.verbose = True
        self.loss_values = []
        self.last_lr_update = 0
        self.lr_patience = lr_patience

    def quiet(self):
        self.verbose = False
        self.filename = None

    def append_loss(self, loss, model, epoch=None):
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        self.loss_values.append(loss)
        loss_values = np.array(self.loss_values, dtype=float)
        if epoch is None:
            epoch = len(loss_values)

        best_loss_value = loss_values.min()
        if self.filename and loss_values[-1] == best_loss_value:
            print("Best validation loss so far, storing", self.filename)
            state_dict = get_model_state_dict(model)
            torch.save({
                'state_dict': state_dict,
                'epoch': epoch,
                'loss': loss,
                'best_loss': best_loss_value,
            }, self.filename)

        if epoch > self.last_lr_update + 50 and np.all(
            loss_values[-50:] > best_loss_value
        ):
            if self.verbose:
                print("Val loss did not improve for 50 epochs, stopping")
            self.last_lr_update = epoch
            self.lr = 0
        elif epoch > self.last_lr_update + self.lr_patience and np.all(
            loss_values[-self.lr_patience:] > best_loss_value
        ):
            if self.verbose:
                print(f"Val loss did not improve for {self.lr_patience} epochs, reduce LR")
            self.last_lr_update = epoch
            self.lr /= 10
            if self.lr < 5e-6:
                if self.verbose:
                    print("LR too small, stopping")
                self.lr = 0

    def should_stop(self):
        return self.lr == 0
    
    def get_min_loss(self):
        if len(self.loss_values) == 0:
            return float('inf')
        return min(self.loss_values)


class TrainingLogger:
    def _get_shanghai_time(self):
        return datetime.now(timezone(timedelta(hours=8)))
    
    def __init__(self, log_path, resume=False):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        
        log_exists = os.path.exists(log_path)
        if resume and log_exists:
            self.log_file = open(log_path, 'a')
            self.log_file.write(f"\nResuming training at {self._get_shanghai_time().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n")
        else:
            self.log_file = open(log_path, 'w')
            self.log_file.write(f"Training log started at {self._get_shanghai_time().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n")
    
    def log_args(self, args):
        args_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
        self.log_file.write("\nTraining Arguments:\n")
        self.log_file.write("-" * 80 + "\n")
        for key, value in sorted(args_dict.items()):
            self.log_file.write(f"  {key}: {value}\n")
        self.log_file.write("-" * 80 + "\n\n")
        self.log_file.flush()
    
    def log_epoch(self, epoch, val_mse, lr=None, min_val_mse=None):
        log_line = f"Epoch {epoch:4d}: val MSE = {val_mse:.6e}"
        if lr is not None:
            log_line += f", lr = {lr:.2e}"
        if min_val_mse is not None:
            log_line += f", min val MSE = {min_val_mse:.6e}"
        self.log_file.write(log_line + "\n")
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.write(f"\nTraining completed at {self._get_shanghai_time().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 80 + "\n")
            self.log_file.close()


def setup_logger(model_path, resume=False, args=None):
    log_path = model_path.replace('.pt', '.log') if model_path.endswith('.pt') else model_path + '.log'
    logger = TrainingLogger(log_path, resume=resume)
    if args:
        logger.log_args(args)
    return logger

