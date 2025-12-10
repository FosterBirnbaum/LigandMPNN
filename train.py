import argparse
import os.path
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import torch.distributed as dist
import torch.multiprocessing as mp
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(rank, world_size, args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.utils.checkpoint as cp
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    import multiprocessing as mp
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor 
    from data_utils import worker_init_fn, get_pdbs, loader_pdb, PDB_dataset, StructureDataset, StructureSampler, CustomDistributedSampler, _to_dev
    from model_utils import loss_smoothed, nlcpl, loss_nll, get_std_opt, ProteinMPNN, checkpoint_multiple_outputs
    os.environ["RANK"] = str(rank)
    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder, exist_ok=True)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    meta_path = args.path_for_meta_data
    subset_chains = args.subset_chains
    params = {
        "PDBDIR"     : f"{data_path}",
        "METADIR"    : f"{meta_path}",
        "HOMO"       : 0.70, #min seq.id. to detect homo chains,
        "SUBSET"     : subset_chains,
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 0}

   
    if args.debug:
        args.num_examples_per_epoch = 500
        args.max_protein_length = 2000
        args.batch_size = 2000

    print('loaded args')
    print(args)
    print(params)
    with open(os.path.join(args.path_for_data_splits, 'train.json'), 'r') as f:
        train_list = json.load(f) 
    with open(os.path.join(args.path_for_data_splits, 'valid.json'), 'r') as f:
        valid_list = json.load(f)  

    if args.debug:
        train_list = train_list[:1000]
        valid_list = valid_list[:1000]
        train_list = ['3tdq', 
                        '3rs7',
                        '213l',
                        '5t6j',
                        '6zl3',
                        '6u5d',
                        '7loa',
                        '5f79',
                        '4qnc',
                        '7loe',
                        '6u5g',
                        '2yq6',
                        '4low',
                        '6kyg',
                        '6vd8',
                        '218l',
                        '2vy1',
                        '4jhh',
                        '4eqk',
                        '3vpv',
                        '5jgz',
                        '4j46',
                        '5ta2',
                        '4yul'
                        ]
        
        valid_list = ['3tdq', 
                        '3rs7',
                        '213l',
                        '5t6j',
                        '6zl3',
                        '6u5d',
                        '7loa',
                        '5f79',
                        '4qnc',
                        '7loe',
                        '6u5g',
                        '2yq6',
                        '4low',
                        '6kyg',
                        '6vd8',
                        '218l',
                        '2vy1',
                        '4jhh',
                        '4eqk',
                        '3vpv',
                        '5jgz',
                        '4j46',
                        '5ta2',
                        '4yul'
                        ]

    train_set = PDB_dataset(train_list, loader_pdb, params)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(valid_list, loader_pdb, params)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    print('loaded loaders')

    if args.debug:
        for i, _ in enumerate(train_loader):
            pass
        for i_valid, _ in enumerate(valid_loader):
            pass
        print('debug loader len: ', i, i_valid)

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=args.k_neighbors,
        device=device,
        atom_context_num=args.atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=args.ligand_mpnn_use_side_chain_context,
        output_dim=args.output_dim,
        node_self_sub=args.node_self_sub,
        clone=args.clone,
        use_potts=args.use_potts,
        updated_alist=args.updated_alist,
        potts_context=args.potts_context
    )

    if PATH:
        checkpoint = torch.load(PATH, map_location=device, weights_only=False)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        # state_dict = {}
        # for k, v in checkpoint['model_state_dict'].items():
        #     state_dict[k[7:]] = v
        model.load_state_dict(checkpoint['model_state_dict'])
        print('LOADED CP!')
    else:
        total_step = 0
        epoch = 0
    model.to(device=device)

    # Set up DDP
    
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank])

    # for npa, p in model.named_parameters():
    #     print(rank, npa, p.dtype, p.device)
    print('setup model')


    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # for state in optimizer.optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(rank)
    print('setup optimizer')
    kwargs = {}
    kwargs['num_workers'] = args.num_workers
    kwargs['multiprocessing_context'] = 'spawn'


    pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
    pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch)

    esm, alphabet, batch_converter, esm_embed_layer, esm_embed_dim = None, None, None, None, None
    one_hot = True

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)

    # loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)

    train_batch_sampler = StructureSampler(dataset_train, world_size, rank, batch_size=args.batch_size, device='cpu', cutoff_for_score=args.cutoff_for_score, ligand_mpnn_use_atom_context=args.ligand_mpnn_use_atom_context, atom_context_num=args.atom_context_num, model_type=args.model_type, ligand_mpnn_use_side_chain_context=args.ligand_mpnn_use_side_chain_context, parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy, transmembrane_buried=args.transmembrane_buried, transmembrane_interface=args.transmembrane_interface, global_transmembrane_label=args.global_transmembrane_label, subset_chains=args.subset_chains, updated_alist=args.updated_alist)
    # ddp_train_sampler = CustomDistributedSampler(train_batch_sampler, num_replicas=world_size, rank=rank)

    loader_train = DataLoader(dataset_train, batch_sampler=train_batch_sampler, collate_fn=train_batch_sampler.package, pin_memory=True, **kwargs)
    # loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    valid_batch_sampler = StructureSampler(dataset_valid, world_size, rank, batch_size=args.batch_size, device='cpu', cutoff_for_score=args.cutoff_for_score, ligand_mpnn_use_atom_context=args.ligand_mpnn_use_atom_context, atom_context_num=args.atom_context_num, model_type=args.model_type, ligand_mpnn_use_side_chain_context=args.ligand_mpnn_use_side_chain_context, parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy, transmembrane_buried=args.transmembrane_buried, transmembrane_interface=args.transmembrane_interface, global_transmembrane_label=args.global_transmembrane_label, subset_chains=args.subset_chains, updated_alist=args.updated_alist)
    # ddp_val_sampler = CustomDistributedSampler(valid_batch_sampler, num_replicas=world_size, rank=rank)
    loader_valid = DataLoader(dataset_valid, batch_sampler=valid_batch_sampler, collate_fn=valid_batch_sampler.package, pin_memory=True, **kwargs)

    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        nlcpl_train_sum = 0.
        adj_denom = 0
        train_acc = 0.
        best_val_loss = np.inf
    
        train_batch_sampler._set_epoch(e)
        # ddp_train_sampler._set_epoch(e)
        if rank == 0:
            progress = tqdm(total=len(loader_train))
        # try:
        for i_train_batch, feature_dict in enumerate(loader_train):
            if feature_dict is None:
                print(rank, "SKIP!")
                continue
            _to_dev(feature_dict, device, rank)
            S_true = feature_dict["S"]
            mask_for_loss = feature_dict["mask"] * feature_dict["chain_mask"]
            mask = feature_dict["mask"]

            optimizer.zero_grad()

            if False and args.mixed_precision:
                with torch.cuda.amp.autocast():

                    # set the number of checkpoint segments
                    # segments = 2

                    # get the modules in the model. These modules should be in the order
                    # the model should be executed
                    # modules = [module for k, module in model._modules.items()]
                    # functions = [model.encode, model.decode]
                    # segments = 2

                    # encoded = cp.checkpoint(lambda fdict: model.encode(fdict), feature_dict)
                    # log_probs, etab, E_idx = cp.checkpoint(lambda enc: model.decode(enc), encoded)

                    # # Helper to flatten multiple tensors into a single tensor
                    # def flatten_outputs(outputs):
                    #     return torch.cat([t.contiguous().view(-1) for t in outputs])

                    # # Helper to reconstruct tensors from flat vector
                    # def reconstruct(flat_tensor, templates):
                    #     sizes = [t.numel() for t in templates]
                    #     outputs = []
                    #     offset = 0
                    #     for size, t in zip(sizes, templates):
                    #         out = flat_tensor[offset:offset + size].view_as(t)
                    #         outputs.append(out)
                    #         offset += size
                    #     return tuple(outputs)

                    # # Encode + flatten for checkpointing
                    # def encode_and_flatten(inputs):
                    #     outputs = model.encode(inputs)
                    #     return flatten_outputs(outputs)

                    # flat_encoded = checkpoint(encode_and_flatten, feature_dict)

                    # # Reconstruct encoded tensors using non-grad version
                    # with torch.no_grad():
                    #     enc_template = model.encode(feature_dict)
                    # reconstructed_enc = reconstruct(flat_encoded, enc_template)

                    # # Decode + flatten
                    # def decode_and_flatten(*encoded_tensors):
                    #     outputs = model.decode(encoded_tensors)
                    #     return flatten_outputs(outputs)

                    # flat_decoded = checkpoint(lambda *x: decode_and_flatten(*x), *reconstructed_enc)

                    # # Reconstruct decode outputs
                    # with torch.no_grad():
                    #     dec_template = model.decode(enc_template)
                    # log_probs, etab, E_idx = reconstruct(flat_decoded, dec_template)

                    # encoded = checkpoint_multiple_outputs(model.encode, feature_dict)
                    # log_probs, etab, E_idx = checkpoint_multiple_outputs(model.decode, encoded)
                    # now call the checkpoint API and get the output
                    # log_probs, etab, E_idx = checkpoint_sequential(functions, segments, feature_dict, use_reentrant=False)
                    try:
                        log_probs, etab, E_idx = model(feature_dict)
                    
                        loss_calc, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss)
                    except:
                        print(feature_dict['filepaths'], S_true.shape, mask_for_loss.shape)
                    if loss_calc is None:
                        continue
                    
                    if args.etab_loss:
                        nlcpl_loss, n_edges = nlcpl(etab, E_idx, S_true, mask)
                        if n_edges != -2:
                            loss_av_smoothed += nlcpl_loss
                            nlcpl_train_sum += nlcpl_loss.cpu().item()
                        else:
                            adj_denom += 1
                            continue
        
                scaler.scale(loss_av_smoothed).backward()
                    
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs, etab, E_idx = model(feature_dict)
                loss_calc, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss)
                if loss_calc is None:
                    continue
                if args.etab_loss:
                    nlcpl_loss, n_edges = nlcpl(etab, E_idx, S_true, mask)
                    if n_edges != -2:
                        loss_av_smoothed += nlcpl_loss
                        nlcpl_train_sum += nlcpl_loss.cpu().item()
                    else:
                        adj_denom += 1
                        continue
                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
        # allocated = torch.cuda.memory_allocated(device)
        # reserved = torch.cuda.memory_reserved(device)
        # max_allocated = torch.cuda.max_memory_allocated(device)
        # print(f"[Batch {i_train_batch}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max Allocated: {max_allocated:.2f} MB")
        # torch.cuda.reset_peak_memory_stats(device)
        
            loss, loss_av, true_false = loss_nll(S_true, log_probs, mask_for_loss)
            # if args.etab_loss:
            #     nlcpl_loss, _ = nlcpl(etab, E_idx, S_true, mask)
            #     nlcpl_train_sum += nlcpl_loss.cpu().item()
        
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            total_step += 1

            if rank == 0 and i_train_batch % 10 == 0:
                str_loss = np.format_float_positional(np.float32(train_sum / train_weights), unique=False, precision=3)
                str_acc = np.format_float_positional(np.float32(train_acc / train_weights), unique=False, precision=3)
                progress.update(10)
                progress.refresh()
                progress.set_description_str(f'avg loss {str_loss} | acc {str_acc}')

        # except Exception as err:
        #     # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #     #     log_probs2, etab, E_idx = model(feature_dict)
        #     #     print('log probs 0: ', log_probs2.sum(), torch.isnan(log_probs2).any())
        #     # # for k, v in feature_dict.items():
        #     # #     if torch.is_tensor(v):
        #     # #         print(k, v.shape)
        #     print(feature_dict['filepaths'])
        #     log_probs, etab, E_idx = model(feature_dict)
        #     print('log probs 1: ', log_probs.sum(), torch.isnan(log_probs).any())
        #     # str_rank = str(rank)
        #     # checkpoint_filename_last = base_folder+f'model_weights/epoch_error_rank{str_rank}.pt'
        #     # torch.save({
        #     #         'epoch': e+1,
        #     #         'step': total_step,
        #     #         'num_edges' : args.num_neighbors,
        #     #         'noise_level': args.backbone_noise,
        #     #         'noise_type': args.noise_type,
        #     #         'model_state_dict': model.state_dict(),
        #     #         'optimizer_state_dict': optimizer.optimizer.state_dict(),
        #     #         }, checkpoint_filename_last)
            
        #     # model = model.eval()
        #     # with torch.no_grad():
        #     #     log_probs, etab, E_idx = model(feature_dict)
        #     #     try:
        #     #         loss, loss_av, true_false = loss_nll(S_true, log_probs, mask_for_loss)
        #     #     except Exception as err2:
        #     #         print(err2)
        #     # print('nll loss:')
        #     # print(loss.sum())
        #     # print(loss_av)
        #     # print(torch.isnan(loss).any())
        #     # print('log probs 2: ', log_probs.sum(), torch.isnan(log_probs).any())
        #     # try:
        #     #     loss_calc, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss)
        #     #     print("SUCCESS!")
        #     # except Exception as err3:
        #     #     S_onehot = torch.nn.functional.one_hot(S_true.to(dtype=torch.long), 21).float()

        #     #     # Label smoothing
        #     #     weight=0.1
        #     #     S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        #     #     S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

        #     #     loss_calc = -(S_onehot * log_probs).sum(-1)
        #     #     loss_av_smoothed = torch.sum(loss * mask) / 2000.0 #fixed 
        #     #     print(err3)
        #     #     raise err3
        #     # print('t2: ', loss_calc.sum(), torch.isnan(loss_calc).any(), loss_av_smoothed)
        #     # print('lc shape: ', loss_calc.shape)
        #     # model = model.train()
        #     # log_probs, etab, E_idx = model(feature_dict)
        #     # print('log probs 3: ', log_probs.sum(), torch.isnan(log_probs).any())
        #     # loss_calc, loss_av_smoothed = loss_smoothed(S_true, log_probs, mask_for_loss)
            
        #     raise err
        
        if rank == 0:
            progress.close()
            progress = tqdm(total=len(loader_valid))
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            nlcpl_validation_sum = 0.
            adj_denom_val = 0
            validation_acc = 0.
            for i_val_batch, feature_dict in enumerate(loader_valid):
                if feature_dict is None:
                    continue
                _to_dev(feature_dict, device, rank)
                S_true = feature_dict["S"]
                mask_for_loss = feature_dict["mask"] * feature_dict["chain_mask"]
                mask = feature_dict["mask"]
                log_probs, etab, E_idx = model(feature_dict)
                loss, loss_av, true_false = loss_nll(S_true, log_probs, mask_for_loss)
                if args.etab_loss:
                    nlcpl_loss, n_edges = nlcpl(etab, E_idx, S_true, mask)
                    if n_edges != -1:
                        nlcpl_validation_sum += nlcpl_loss.cpu().item()
                    else:
                        adj_denom_val += 1
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                if rank == 0 and i_val_batch % 10 == 0:
                    str_loss = np.format_float_positional(np.float32(validation_sum / validation_weights), unique=False, precision=3)
                    str_acc = np.format_float_positional(np.float32(validation_acc / validation_weights), unique=False, precision=3)
                    progress.update(10)
                    progress.refresh()
                    progress.set_description_str(f'avg val loss {str_loss} | val acc {str_acc}')
        if rank == 0:
            progress.close()
        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
        comb_loss = copy.deepcopy(validation_loss)

        if args.etab_loss:
            train_nlcpl = nlcpl_train_sum / (i_train_batch + 1 - adj_denom)
            validation_nlcpl = nlcpl_validation_sum / (i_val_batch + 1 - adj_denom_val)
            comb_loss += validation_nlcpl
            train_nlcpl = np.format_float_positional(np.float32(train_nlcpl), unique=False, precision=3)   
            validation_nlcpl = np.format_float_positional(np.float32(validation_nlcpl), unique=False, precision=3)   
            

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        with open(logfile, 'a') as f:
            f.write(f'rank: {rank}, epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            if args.etab_loss:
                f.write(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}\n')
        print(f'rank: {rank}, epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
        if args.etab_loss:
            print(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}')

        if rank == 0:
            if comb_loss < best_val_loss:
                checkpoint_filename_last = base_folder+'model_weights/epoch_best.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'noise_type': args.noise_type,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)
                best_val_loss = comb_loss
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'noise_type': args.noise_type,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'noise_type': args.noise_type, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)
                
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_meta_data", type=str, default="my_path/pdb_2021aug02", help="path for loading meta data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=8000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--noise_type", type=str, default='atomic', help="type of noise to add during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--replicate", type=int, default=1, help='replicate to use in setting seed for noise')
    argparser.add_argument("--feat_type", type=str, default="protein_mpnn", help="type of featurizer to use")
    argparser.add_argument("--etab_loss", type=int, default=0, help="whether to use Potts model loss")
    argparser.add_argument("--output_dim", type=int, default=400, help="Potts model output dimension")
    argparser.add_argument("--node_self_sub", type=str, default=None, help="whether to use sequence probabilities as Potts model self energies")
    argparser.add_argument("--clone", type=bool, default=True, help="whether to clone node tensors if using for Potts model self energies")
    argparser.add_argument("--seq_encoding", type=str, default="one_hot", help="Sequence encoding to use")
    argparser.add_argument("--num_workers", type=int, default=12, help="number of workers to use for data loading")
    argparser.add_argument("--k_neighbors", type=int, default=32, help="number of nearest neighbors to use in graph")
    argparser.add_argument("--atom_context_num", type=int, default=16, help="number of atom neighbors to use in graph")
    argparser.add_argument("--ligand_mpnn_use_side_chain_context", type=int, default=0, help="flag to use side chain atoms as ligand context for the fixed residues")
    argparser.add_argument('--ligand_mpnn_use_atom_context', type=int, default=1, help='1 - use atom context, 0 - do not use atom context')
    argparser.add_argument("--cutoff_for_score", type=float, default=8.0, help="cutoff for score")
    argparser.add_argument("--use_potts", type=int, default=0, help="whether to use potts model")
    argparser.add_argument("--model_type", type=str, default='ligand_mpnn', help='type of model to train')
    argparser.add_argument("--path_for_data_splits", type=str, default='/orcd/home/002/fosterb/LigandMPNN/training/', help='path for train/val/test splits')
    argparser.add_argument("--parse_atoms_with_zero_occupancy", type=int, default=0, help='to parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy')
    argparser.add_argument("--transmembrane_buried", type=str, default='', help='provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25')
    argparser.add_argument("--transmembrane_interface", type=str, default='', help='provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25')
    argparser.add_argument("--global_transmembrane_label", type=int, default=0, help='provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble')
    argparser.add_argument("--subset_chains", type=int, default=0, help='whether to subset chains to biological assembly')
    argparser.add_argument("--updated_alist", type=int, default=1, help='whether to use updated other atom list (including X) (1 for use, 0 for not use)')
    argparser.add_argument("--potts_context", type=str, default='', help='how to incorporate ligand context into Potts model')
    args = argparser.parse_args()
    print('etab loss: ', args.etab_loss)
    print('num devices: ', torch.cuda.device_count())
    args.use_potts = bool(args.use_potts)
    args.subset_chains = bool(args.subset_chains)
    args.etab_loss = bool(args.etab_loss)
    args.updated_alist = bool(args.updated_alist)
    print('etab loss: ', args.etab_loss)
    print('starting')    

    world_size = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(world_size)

    # mp.spawn(main, args=(world_size, args,), nprocs=world_size, join=True)

    main(0, 1, args)   
