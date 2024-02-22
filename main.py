"""Example implementation to run Nyonic Models."""
import os
import functools
import torch
import argparse
from argparse import Namespace
from omegaconf import OmegaConf

from nyonic.tokenizer import NyonicTokenizer
from nyonic.model import GPTModel
from nyonic.sampling import sample, generate_tokens


def parse_args() -> Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_conf", type=str, default="confs/wonton-1.5B.yaml")
    parser.add_argument("--max_tokens", default=200)
    parser.add_argument("--strategy", default="vanilla")
    parser.add_argument("--top_p", default=1.0)
    parser.add_argument("--top_k", default=100)
    parser.add_argument("--temperature", default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def generate() -> str:
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
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    print(f"Loading model from {args.model_conf}")
    nyonic = GPTModel(cfg.model_args).to(torch.bfloat16)
    ckpt = (
        torch.load(cfg.model_path, device)["state_dict"]
        if "1.5B" in args.model_conf
        else torch.load(cfg.model_path, device)["module"]
    )
    omit_args = [
        "encoder_layer.self_attn.in_proj_weight",
        "encoder_layer.self_attn.in_proj_bias",
        "encoder_layer.self_attn.out_proj.weight",
        "encoder_layer.self_attn.out_proj.bias",
        "encoder_layer.linear1.weight",
        "encoder_layer.linear1.bias",
        "encoder_layer.linear2.weight",
        "encoder_layer.linear2.bias",
        "encoder_layer.norm1.weight",
        "encoder_layer.norm1.bias",
        "encoder_layer.norm2.weight",
        "encoder_layer.norm2.bias",
        "final_norm.weight",
        "final_norm.bias",
        "self_attn.rotary_embedding.inv_freq"
    ]
    arg_mappings = {
        "out_proj_weight": "out_proj.weight",
        "out_proj_bias": "out_proj.bias"
    }

    def map_arg(arg):
        for source, target in arg_mappings.items():
            if source in arg:
                return arg.replace(source, target)
        return arg

    ckpt = {
        key[6:] if key.startswith("model.") else key: value
        for key, value in ckpt.items()
    }
    ckpt = {map_arg(k): v for k, v in ckpt.items() if k not in omit_args}
    nyonic.load_state_dict(ckpt)
    nyonic.to(device)
    nyonic.eval()
    print("Model loaded")

    prompt = "This is test, please write a lovely poem "
    print(f"Using prompt {prompt}")
    context_enc = torch.tensor([tokenizer.encode(prompt)]).to(device)
    gen_length = min(
        cfg.gen_conf.max_seq_len - context_enc.shape[1], cfg.gen_conf.max_gen_len
    )
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
    )[0].tolist()[context_enc.shape[1] :]
    completion = tokenizer.decode(completion)
    return completion


if __name__ == "__main__":
    generated = generate()
    print(f"Completion generated:\n {generated}")
