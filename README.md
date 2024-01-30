# nyonic-public 
This repository contains minimal code to run models from Nyonic Model Factory which produces a set of large language models with a stable and fast data processing and training pipeline. 
## Installation
```
git clone https://github.com/nyonicai/nyonic-public.git
cd nyonic-public
pip install -r requirements.txt
```

## Download the model
Please fill in [this form](...) to request access to our model(s). The model file shoud be placed under the `models ` folder.



## Run the model
For a trivial test, simply run

```
python -m main
```

which use default inference settings from [Nyonic-1.5B-v0](conf/nyonic_1.5B_v0.yaml), you can specify a model you want use with `--model_conf` argument, or change any supported sampling parameters. A complete example:

```
python -m main --model_conf /path/to/model_conf.yaml --max_tokens 200 --strategy top_p --top_p 0.6 --temperature 0.8
```

