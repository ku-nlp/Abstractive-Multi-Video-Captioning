import os
from timeit import default_timer as timer

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

from models import model_utils
from datasets import dataset_utils
from run.utils import count_model_params, train_epoch, evaluate, EarlyStopping, collate_fn_multi
from run.make_vocab_vatex_preprocessed import make_vocab

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


@hydra.main(version_base=None, config_path="./conf/lrec-coling2024", config_name="end2end-2videos")
def train(cfg: DictConfig):
    """
    Main function for training the model.
    """
    torch.manual_seed(0)

    project = cfg.run.project
    name = cfg.run.name
    model_ckpt_dir = cfg.run.model_ckpt_dir
    model_ckpt_name = cfg.run.model_ckpt_name
    num_videos = cfg.model.num_videos
    num_epoch = cfg.train.num_epoch
    batch_size = cfg.train.batch_size
    adam_lr = cfg.train.adam_lr
    num_videos = cfg.model.num_videos
    print("num_videos:", num_videos)

    wandb.init(
        mode="disabled",
        project=project,
        name=name
    )

    # vocab を読み込む
    cwd = os.getcwd()
    vocab_path = os.path.join(cwd, "data/VATEX/training.jsonl")
    vocab = make_vocab(vocab_path)
    print("Vocab size: {}".format(len(vocab)))

    # データセットを読み込む
    cfg_dataset = OmegaConf.to_container(cfg.dataset)
    cfg_dataset["num_videos"] = num_videos
    cfg_dataset["vocab"] = vocab
    dataset_name = cfg_dataset["name"]
    train_dataset = dataset_utils.get_dataset(dataset_name, "train", cfg_dataset)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, 100))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multi, num_workers=2, pin_memory=True)
    val_dataset = dataset_utils.get_dataset(dataset_name, "val", cfg_dataset)
    # val_dataset = torch.utils.data.Subset(val_dataset, range(0, 100))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_multi, num_workers=2, pin_memory=True)
    print(f'Dataset Size (Train): {len(train_dataset)}')
    print(f'Dataset Size (Val): {len(val_dataset)}')

    tgt_vocab_size = len(vocab)
    cfg_model = OmegaConf.to_container(cfg.model)
    cfg_model["tgt_vocab_size"] = tgt_vocab_size
    model = model_utils.get_model('transformer-multi', cfg_model)
    
    if cfg.train.finetune:
        model_ckpt_path = os.path.join(model_ckpt_dir, wandb.run.project, f"{model_ckpt_name}.pth")
        print(f"model path for fine-tuning: {model_ckpt_path}")
        model.load_state_dict(torch.load(model_ckpt_path))
    model = model.to(DEVICE)

    total_params, trainable_params = count_model_params(model)
    print(f"num of model parameters: {trainable_params} / {total_params} (trainable / total)")

    if cfg.train.finetune is False:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, betas=(0.9, 0.98), eps=1e-9)

    model_save_dir = f"./models/checkpoints/{project}"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{name}-{wandb.run.id}.pth")
    earlystopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)
    for epoch in range(1, num_epoch+1):
        start_time = timer()
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
        end_time = timer()
        val_loss = evaluate(model, val_dataloader, loss_fn)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": end_time - start_time
        })
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        earlystopping(val_loss, model)
        if earlystopping.early_stop:
            print("Early-stopped !")
            break


if __name__ == "__main__":
    train()
