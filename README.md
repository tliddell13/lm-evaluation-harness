# Language Model Evaluation Harness with Word Order Permutations

This is an extended version of the original [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) that adds support for word order permutations and other text modifications during evaluation. This extension was developed as part of research investigating the sensitivity of Large Language Models to linguistic structures, particularly word order.

## 🔬 Research Extensions

This version allows for various permutations to be applied to datasets during evaluation to study model robustness:

### Available Permutation Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `shuffle` | Shuffles question words | `unigram`, `bigram`, `trigram` |
| `shuffleAnswer` | Shuffles answer choices | `unigram`, `bigram`, `trigram` |
| `remove_question` | Removes the question entirely | `True`/`False` |
| `posReplace` | Replaces parts of speech with synonyms | POS tag |
| `extra_answers` | Generates distracting answers | `True`/`False` |
| `named_entities` | Modify named entities | `remove_all`, `keep_only` |
| `cot` | Use chain of thought prompting for GSM8K | `True`/`False` |

### Implementation Details
- **Permutation functions**: Defined in `permutations.py`
- **Integration**: Applied during evaluation in `evaluator.py`
- **Examples**: Slurm job scripts in `SlurmEvals/` folder
- **Results**: Analysis and outputs in `permutation_results/` folder

---

## 📖 Original LM-Evaluation-Harness Documentation

### Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.
- Task versioning to ensure reproducibility when tasks are updated.

## Install

To install `lm-eval` from this extended repository:

```bash
git clone https://github.com/tliddell13/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models:

```bash
pip install -e ".[auto-gptq]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.

### Standard Evaluation

To evaluate a model on standard benchmarks:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0
```

### Evaluation with Permutations

To evaluate with word order shuffling:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --shuffle unigram \
    --device cuda:0
```

To evaluate with question removal:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks truthfulqa_mc \
    --remove_question True \
    --device cuda:0
```

### Commercial APIs

Our library also supports language models served via the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

## Research Applications

This extended version was used in research examining:
- LLM sensitivity to word order perturbations
- Benchmark robustness to input modifications  
- The role of superficial cues vs. linguistic understanding in model performance

Results showed that many models maintain surprisingly high performance even with shuffled word order, suggesting reliance on keyword associations rather than deep linguistic understanding.

## Advanced Usage

For models loaded with the HuggingFace `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library.

GPTQ quantized models can be loaded by specifying their file names in `,quantized=NAME` (or `,quantized=True` for default names) in the `model_args` argument:

```bash
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=model-name-or-path,quantized=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

## Implementing New Tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

We provide utilities for comparing results on a benchmark using only the data points not found in the model training set. For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device cuda:0
```

## Cite as

### Original Framework

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```

### Word Order Permutation Extensions

If you use the permutation capabilities, please also cite:

```
@mastersthesis{liddell2024experiments,
  author = {Tyler Liddell},
  title = {Experiments With Word Order In Large Language Models},
  school = {City University of London},
  year = {2024},
  type = {MSci Project}
}
```

## Repository Structure

```
├── lm_eval/                    # Core evaluation framework
├── permutations.py            # Word order permutation functions
├── evaluator.py               # Modified evaluator with permutation support  
├── SlurmEvals/                # Example Slurm job scripts
├── permutation_results/       # Results and analysis from permutation experiments
└── docs/                      # Documentation
```

## Contributing

We welcome contributions to both the core framework and the permutation extensions. Please see the original repository's contribution guidelines and ensure any new permutation methods are well-documented and tested.
