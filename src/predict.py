import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from models import model_utils
from datasets import dataset_utils
from run.utils import translate, count_model_params, collate_fn_multi
from run.make_vocab_vatex_preprocessed import make_vocab

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


@hydra.main(version_base=None, config_path="./conf/lrec-coling2024", config_name="end2end_2videos")
def predict(cfg: DictConfig):
    """
    Main function for training the model.
    """
    torch.manual_seed(0)

    project = cfg.run.project
    model_ckpt_dir = cfg.run.model_ckpt_dir
    model_ckpt_name = cfg.run.model_ckpt_name
    num_videos = cfg.model.num_videos

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
    test_dataset = dataset_utils.get_dataset(dataset_name, "test", cfg_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_multi, num_workers=2, pin_memory=True)
    print(f'Dataset Size (Test): {len(test_dataset)}')

    # モデルを読み込む
    tgt_vocab_size = len(vocab)
    cfg_model = OmegaConf.to_container(cfg.model)
    cfg_model["tgt_vocab_size"] = tgt_vocab_size
    model = model_utils.get_model('transformer-multi', cfg_model)

    model_ckpt_path = os.path.join(model_ckpt_dir, cfg.run.project, f"{model_ckpt_name}.pth")
    model.load_state_dict(torch.load(model_ckpt_path))
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
            pred = translate(model, src, vocab)
            f.write(pred + "\n")
    f.close()


if __name__ == "__main__":
    predict()
