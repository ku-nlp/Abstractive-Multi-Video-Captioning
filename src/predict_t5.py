import os
from timeit import default_timer as timer

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

from datasets import dataset_utils
from run.utils_t5 import count_model_params

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


@hydra.main(version_base=None, config_path="./conf/thesis", config_name="t5_2videos")
def predict(cfg: DictConfig):
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

    # データセットを読み込む
    cfg_dataset = OmegaConf.to_container(cfg.dataset)
    cfg_dataset["num_videos"] = num_videos
    dataset_name = cfg_dataset["name"]
    test_dataset = dataset_utils.get_dataset(dataset_name, "test", cfg_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f'Dataset Size (Test): {len(test_dataset)}')

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model_ckpt_path = os.path.join(model_ckpt_dir, cfg.run.project, f"{model_ckpt_name}")
    model = T5ForConditionalGeneration.from_pretrained(model_ckpt_path)
    model = model.to(DEVICE)

    total_params, trainable_params = count_model_params(model)
    print(f"num of model parameters: {trainable_params} / {total_params} (trainable / total)")

    # predict on test set
    output_dir = f"./models/predictions/{project}"
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"{model_ckpt_name}.txt")
    f = open(output_name, "w")
    with tqdm(test_dataloader) as pbar:
        pbar.set_description(f'[Test]')
        for i, (src, tgt) in enumerate(pbar):
            input_ids = tokenizer(src, return_tensors='pt', padding=True).input_ids.to(DEVICE)
            output_sequences = model.generate(input_ids=input_ids)
            preds = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            for pred in preds:
                f.write(pred + "\n")
    f.close()


if __name__ == "__main__":
    predict()
