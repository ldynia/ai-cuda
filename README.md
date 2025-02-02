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

pip install -r requirements.txt
```

## Run

```python
python3 semantic-analysis.py
python3 speach-recognition.py
```

## Links

- [Getting started with Transformers](https://huggingface.co/docs/transformers/quicktour)
- [Getting started with Datasets](https://huggingface.co/docs/datasets/en/quickstart)