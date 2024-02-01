"""Example implementation to run Nyonic Models."""
import os
import functools
import torch
import argparse
from argparse import Namespace
from omegaconf import OmegaConf

from tokenizer import NyonicTokenizer
from model import GPTModel
from sampling import sample, generate_tokens


def parse_args() -> Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_conf", type=str, default="confs/nyonic_1.5B_v0.yaml")
    parser.add_argument("--max_tokens", default=200)
    parser.add_argument("strategy", default="vanilla")
    parser.add_argument("top_p", default=1.0)
    parser.add_argument("top_k", default=100)
    parser.add_argument("temperature", default=1.0)
    parser.add_argument('device', default="cuda:0")
    return parser.parse_args()


def generate():
    """Completion of given prompts."""
    args = parse_args()
    assert os.path.exists(args.model_conf)
    cfg = OmegaConf.load(args.model_conf)
    tokenizer = NyonicTokenizer(
        location=cfg.tokenizer,
    ).load()
    device_list = set(
        ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    )
    if args.device and args.device in device_list:
        device = torch.device(args.device)
        print(f"Using device '{args.device}'")
    else:
        print("Device not specified")
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    print(f"Loading model from {args.model_conf}")
    nyonic = GPTModel(cfg.model_args).to(torch.bfloat16)
    ckpt = torch.load(cfg.model_path, device)["module"]
    ckpt = {
        key[6:] if key.startswith("model.") else key: value
        for key, value in ckpt.items()
    }
    nyonic.load_state_dict(ckpt)
    nyonic.to(device)
    nyonic.eval()
    print("Model loaded")

    prompt = "This is test, please write a lovely poem "
    print(f"Using prompt {prompt}")
    context_enc = torch.tensor([tokenizer.encode(prompt)]).to(device)
    gen_length = min(args.gen_conf.max_seq_len - context_enc.shape[1], args.gen_conf.max_gen_len)
    sampler = functools.partial(
        sample,
        **cfg.gen_conf.sampling,
    )
    content = torch.cat(
        [
            context_enc,
            torch.tensor(
                [[tokenizer.pad_id] * gen_length],
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )
    completion = generate_tokens(
        model=nyonic,
        tokens=content,
        min_prompt_len=context_enc.shape[1],
        pad_id=tokenizer.pad_id,
        sampler=sampler,
    )[0].tolist()[context_enc.shape[1]:]
    completion = tokenizer.decode(completion)
    return completion


if __name__ == "__main__":

    print("Completion generated:\n")
    print(generate())
