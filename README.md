Project for testing various LLMs on CUDA vs. CPU

## Requirements

```bash
sudo apt-get install nvtop
```

## Install


```bash
pyton3 -m venv .venv

source .venv/bin/activate
deactivate

pip install -r src/requirements.txt
```

## Run

```python
python3 src/semantic-analysis.py
python3 src/speach-recognition.py
```

## HuggingFace

1. [Datasets](https://huggingface.co/docs/datasets/index)
1.1 [Arrow](https://huggingface.co/docs/datasets/about_arrow)
1.1 [Cache](https://huggingface.co/docs/datasets/about_cache)
1.1 [Quick Start](https://huggingface.co/docs/datasets/quickstart)

- https://huggingface.co/docs/datasets/quickstart
- [Getting started with Datasets](https://huggingface.co/docs/datasets/en/quickstart)
- [Getting started with Transformers](https://huggingface.co/docs/transformers/quicktour)
