import argparse
import os
import time

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
from torch import optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import faiss
import datasets
import model_rqmoe
from utils import (
    mean_squared_error,
    fix_random_seed,
    find_free_port,
    Scheduler,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
)
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks


def train_one_epoch(model, xt, idx_batches, optimizer, cur_epoch, verbose=True, rank=0, world_size=1):
    is_ddp = isinstance(model, DDP)
    actual_model = model.module if is_ddp else model
    device = next(actual_model.parameters()).device
    d = xt.shape[1]
    sum_loss = 0
    t0 = time.time()
    
    for i, idx_batch in enumerate(idx_batches):
        model.zero_grad()
        batch = xt[idx_batch]
        batch = torch.from_numpy(batch).to(device) / actual_model.db_scale

        codes, xhat, losses = model(batch)

        eps = 1e-6
        nor_loss = []
        original_energy = (batch**2).sum() + eps
        nor_loss.append(torch.log1p(losses[0] / original_energy))
        for layer in range(1, len(losses)):
            ratio = losses[layer] / (losses[layer-1].detach() + eps)
            nor_loss.append(torch.log1p(ratio))
        loss = torch.stack(nor_loss).sum()
        
        loss.backward()
        optimizer.step()

        loss = loss.item()
        sum_loss += loss
        if verbose and rank == 0:
            print(
                f"[{time.time() - t0:.2f} s] train {i} / {len(idx_batches)} "
                f"loss={loss:g}",
                end="\r",
                flush=True,
            )
    return sum_loss / len(idx_batches)


def compute_MSE(model, x, bs, rank=0):
    is_ddp = isinstance(model, DDP)
    actual_model = model.module if is_ddp else model
    device = next(actual_model.parameters()).device
    actual_model.eval()
    with torch.no_grad():
        t0 = time.time()
        sum_errs = 0
        n = 0
        for i0 in range(0, len(x), bs):
            batch = torch.from_numpy(x[i0 : i0 + bs]).to(device) / actual_model.db_scale
            codes, xhat = actual_model.encode(batch)
            sum_errs += ((xhat - batch) ** 2).sum().item() * actual_model.db_scale**2
            n += len(batch)
            if rank == 0:
                print(
                    f"[{time.time() - t0:.2f} s] inference {n} / {len(x)} "
                    f"MSE={sum_errs / n:g}",
                    end="\r",
                    flush=True,
                )
    actual_model.train()
    return sum_errs / len(x)


def train(args, xt, xval, model, rank=0, world_size=1, logger=None):
    seed = 1234
    fix_random_seed(seed)

    is_ddp = isinstance(model, DDP)
    actual_model = model.module if is_ddp else model

    bs = args.batch_size
    bs_per_gpu = bs // world_size if world_size > 1 else bs
    t0 = time.time()

    epoch0 = 0
    checkpoint_data = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        epoch0 = checkpoint_data.get('epoch', 0) + 1
        is_resuming = True
        if rank == 0:
            print(f"Resuming from epoch {epoch0} (model parameters already loaded)")
        
        if 'scheduler' in checkpoint_data:
            scheduler = checkpoint_data['scheduler']
            if rank != 0:
                scheduler.quiet()
        else:
            scheduler = Scheduler(args.lr, args.model, lr_patience=args.lr_patience)
            if rank != 0:
                scheduler.quiet()
    else:
        if rank == 0:
            MSE_val = compute_MSE(actual_model, xval, bs=32768, rank=rank)
            print(f"Before optimization: val MSE={MSE_val:g}")
        else:
            MSE_val = 0.0

        scheduler = Scheduler(args.lr, args.model, lr_patience=args.lr_patience)
        if rank != 0:
            scheduler.quiet()

    for epoch in range(epoch0, args.max_epochs):
        lr = scheduler.lr
        if rank == 0:
            print(f"[{time.time() - t0:.2f} s] epoch {epoch} {lr=:g}")

        rs = np.random.RandomState(epoch + seed)
        perm = rs.permutation(len(xt))
        
        if world_size > 1:
            perm = perm[: len(perm) // bs * bs]
            idx_batches = [perm[i0 + bs_per_gpu * rank :][:bs_per_gpu] 
                          for i0 in range(0, len(perm), bs)]
        else:
            idx_batches = [perm[i0 : i0 + bs] for i0 in range(0, len(xt), bs)]

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        mean_loss = train_one_epoch(model, xt, idx_batches, optimizer, epoch, 
                                   verbose=(rank == 0), rank=rank, world_size=world_size)

        if world_size > 1:
            device = next(actual_model.parameters()).device
            mean_loss_tensor = torch.tensor(mean_loss, device=device)
            torch.distributed.all_reduce(mean_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            mean_loss = mean_loss_tensor.item() / world_size

        if rank == 0:
            MSE_train = 0
            MSE_val = compute_MSE(actual_model, xval, bs=32768, rank=rank)
            print(f"End of epoch {epoch} train loss={mean_loss:g} train MSE={MSE_train:g} val MSE={MSE_val:g}")
            scheduler.append_loss(MSE_val, actual_model, epoch=epoch)
            
            if logger:
                min_val_mse = scheduler.get_min_loss()
                logger.log_epoch(epoch, MSE_val, lr=lr, min_val_mse=min_val_mse)
            
            if args.checkpoint:
                save_checkpoint(actual_model, args.checkpoint, epoch, scheduler, mean_loss, MSE_val)
        else:
            MSE_val = 0.0

        if scheduler.should_stop():
            break

    if rank == 0:
        print("Training done")
        if logger:
            logger.close()


def train_job(rank, port, args, xt, xval, model, rq_centroids, db_scale):
    world_size = len(args.gpus)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    
    max_retries = 5
    timeout = torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None
    
    for retry in range(max_retries):
        try:
            if timeout:
                init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timeout)
            else:
                init_process_group(backend="nccl", rank=rank, world_size=world_size)
            break
        except Exception as e:
            if "EADDRINUSE" in str(e) or "address already in use" in str(e).lower():
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    if rank == 0:
                        print(f"Port {port} is in use, waiting {wait_time}s before retry {retry + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Rank {rank}: Port {port} is still in use after {max_retries} attempts. "
                                     f"Please wait for other training processes to finish or manually specify a different port.")
            else:
                raise
    
    print(f"Start train_job rank={rank} on GPU {args.gpus[rank]}")

    gpu_id = args.gpus[rank]
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    model.to(device)
    model.db_scale = db_scale
    
    checkpoint, checkpoint_loaded = load_checkpoint(model, args.checkpoint, device, rank=rank)
    
    if not checkpoint_loaded:
        if rank == 0:
            print("Initializing model with centroids (no checkpoint found)")
        initialize_model(model, rq_centroids, db_scale)
    else:
        if rank == 0:
            print("Skipping centroid initialization (using checkpoint parameters)")
    
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)
    model.module.db_scale = db_scale

    logger = setup_logger(args.model, resume=checkpoint_loaded, args=args) if rank == 0 else None

    try:
        train(args, xt, xval, model, rank=rank, world_size=world_size, logger=logger)
    except Exception as e:
        if rank == 0:
            print(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        try:
            destroy_process_group()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Error during process group cleanup: {e}")
    
    if rank == 0:
        print(f"Stop train_job rank={rank}")


def initialize_model(model, rq_centroids, db_scale):
    with torch.no_grad():
        model.codebook0.weight.copy_(torch.from_numpy(rq_centroids[0]) / db_scale)
        for i, step in enumerate(model.steps):
            step.codebook.weight.copy_(
                torch.from_numpy(rq_centroids[i + 1]) / db_scale
            )

def train_rq(args, xt, xval):
    nbit = int(np.log2(args.K))
    print(f"training RQ {args.M}x{nbit}, beam_size={args.rq_beam_size}")
    t0 = time.time()
    rq = faiss.ResidualQuantizer(xt.shape[1], args.M, nbit)
    rq.max_beam_size = args.rq_beam_size
    rq.train(xt)
    print(f"[{time.time() - t0:.2f} s] training done")
    MSE = mean_squared_error(rq.decode(rq.compute_codes(xt)), xt)
    MSE_val = mean_squared_error(rq.decode(rq.compute_codes(xval)), xval)
    print(f"train set {MSE=:g} validation MSE={MSE_val:g}")
    rq_centroids = np.array(get_additive_quantizer_codebooks(rq))
    print(f"RQ centroids size {rq_centroids.shape}")
    return rq_centroids

def main():
    parser = argparse.ArgumentParser()

    def param(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("model parameters")
    param("--d", type=int, help="Input dimension")
    param("--K", type=int, default=256, help="Codebook size")
    param("--M", type=int, default=4, help="Number of quantization steps")
    param("--N", type=int, default=4, help="Number of experts in each RQMoE step")
    param("--L", type=int, default=2, help="Number of layers in each expert (each layer: d->H->d with residual)")
    param("--H", type=int, default=256, help="Hidden dimension for RQMoE")
    param("--dropout", type=float, default=0.1, help="Dropout rate for MLP layers")
    param("--rq_beam_size", type=int, default=1, help="beam size for the initial residual quantizer")

    group = parser.add_argument_group("training parameters")
    param("--lr", type=float, default=1e-3, help="Learning rate")
    param("--lr_patience", type=int, default=10, help="Number of epochs without improvement before reducing learning rate")
    param("--batch_size", type=int, default=4096, help="Batch size")
    param("--max_epochs", type=int, default=1000, help="Maximum number of epochs")

    param("--nt", type=int, default=500_000, help="nb training vectors to use")
    param("--nval", type=int, default=10_000, help="additional validation vectors")

    group = parser.add_argument_group("files")
    param("--model", default="checkpoint/rq_model.pt", help="Model checkpoint path (best model)")
    param("--checkpoint", default="", help="Checkpoint file to load/store during training (for resume)")
    param("--dataset", default="bigann1M", choices=datasets.available_names, help="Dataset name")
    param("--training_data", default="", help="flat npy array with training vectors")
    param("--centroids", default="", help="npy file with initial centroids (if any)")
    param("--db_scale", default=-1, type=float, help="force database scaling. If not set, the maximum is determined automatically from the training set.",)

    group = parser.add_argument_group("computation")
    param("--device", default="cuda:0", help="Device to use (for single GPU)")
    param("--gpus", type=str, default="", help="Comma-separated GPU IDs for DDP (e.g., '0,1,2,3'). If empty, use --device")

    args = parser.parse_args()

    if args.gpus:
        args.gpus = [int(gpu.strip()) for gpu in args.gpus.split(',') if gpu.strip()]
        ngpu = len(args.gpus)
    else:
        if args.device.startswith('cuda:'):
            gpu_id = int(args.device.split(':')[1])
            args.gpus = [gpu_id]
        else:
            args.gpus = [0]
        ngpu = 1

    print("args:", args)
    fix_random_seed(1234)

    if args.training_data:
        print("Loading training data from", args.training_data)
        xt = np.load(args.training_data, mmap_mode="r")
        if len(xt) >= args.nt + args.nval:
            print(f"   Size {xt.shape} -> restrict to {args.nt} + {args.nval}")
            xt = np.array(xt[: args.nt + args.nval])
        else:
            raise RuntimeError("not enough training data")
        args.d = xt.shape[1]
    else:
        print(f"Loading dataset {args.dataset}")
        ds = datasets.dataset_from_name(args.dataset)
        print(f"   {ds}")
        xt = ds.get_train(maxtrain=args.nt + args.nval)
        args.d = ds.d
    
    xt, xval = xt[: -args.nval], xt[-args.nval :]

    print(f"Training set: {xt.shape}, validation: {xval.shape}")

    if args.db_scale > 0:
        db_scale = args.db_scale
    else:
        db_scale = xt.max()
    print(f"Setting scaling factor to {db_scale}")

    if args.centroids and os.path.exists(args.centroids):
        print("Loading initial centroids from", args.centroids)
        rq_centroids = np.load(args.centroids)
    else:
        print("No centroids provided, training RQ")
        rq_centroids = train_rq(args, xt, xval)
        if args.centroids:
            os.makedirs(os.path.dirname(args.centroids) if os.path.dirname(args.centroids) else '.', exist_ok=True)
            np.save(args.centroids, rq_centroids)
            print("Stored RQ centroids to", args.centroids)
    
    if ngpu == 1:
        print("Running single GPU training")
        model = model_rqmoe.RQMoE(
            d=args.d,
            K=args.K,
            M=args.M,
            N=args.N,
            L=args.L,
            H=args.H,
            dropout=args.dropout,
        )
        device = f"cuda:{args.gpus[0]}"
        model.to(device)
        model.db_scale = db_scale
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"nb trainable parameters {num_params}")
        
        checkpoint, checkpoint_loaded = load_checkpoint(model, args.checkpoint, device, rank=0)
        
        if not checkpoint_loaded:
            print("Initializing model with centroids (no checkpoint found)")
            initialize_model(model, rq_centroids, db_scale)
        else:
            print("Skipping centroid initialization (using checkpoint parameters)")

        logger = setup_logger(args.model, resume=checkpoint_loaded, args=args)

        train(args, xt, xval, model, rank=0, world_size=1, logger=logger)
        print("Training completed!")
    else:
        print(f"Running DDP training on {ngpu} GPUs: {args.gpus}")
        assert torch.cuda.device_count() >= ngpu, f"Need at least {ngpu} GPUs, but only {torch.cuda.device_count()} available"
        
        model = model_rqmoe.RQMoE(
            d=args.d,
            K=args.K,
            M=args.M,
            N=args.N,
            L=args.L,
            H=args.H,
            dropout=args.dropout,
        )
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"nb trainable parameters {num_params}")
        
        if "MASTER_PORT" in os.environ:
            port = int(os.environ["MASTER_PORT"])
            print(f"Using manually specified port {port} for DDP communication")
        else:
            port = find_free_port()
            print(f"Using auto-selected port {port} for DDP communication")
        
        mp.spawn(
            train_job,
            args=(port, args, xt, xval, model, rq_centroids, db_scale),
            nprocs=ngpu,
        )
    print("Training completed!")


if __name__ == "__main__":
    main()

