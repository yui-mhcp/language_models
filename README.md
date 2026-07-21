# :yum: Language Models

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

This repository provides an all-in-one API for `Large Language Models (LLM)` built on top of the `BaseModel` interface : a keras `TextGenerator`, `Retriever-Augmented Generation (RAG)`, function calling, multimodal (text + image) support and optional `TensorRT-LLM` accelerated inference.

**Important** : this project does **not** re-implement every existing LLM. Instead, it builds upon the highly optimized [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) library, which offers large-model support with the best performances compared to other libraries !

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   │   ├── *_arch.py                : concrete transformer models (bart, bert, gpt2, mbart, mistral, t5, whisper, xlm-roberta)
│   │   ├── text_transformer_arch.py : base blocks for text-based Transformers
│   │   └── transformer_arch.py      : generic Transformer blocks
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   └── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing (callbacks, losses, metrics, optimizers, ...)*
├── loggers                  : custom utilities for the `logging` module*
├── models                   : main directory for model classes
│   ├── core                     : the `BaseModel` foundation (mixins + saving / migration utils)*
│   ├── nlu                      : Natural-Language-Understanding (LLM) implementations
│   │   ├── conversations            : conversation / message-history management
│   │   │   ├── conv_item_selectors      : strategies to build the message history (naive, last-messages, RAG)
│   │   │   ├── conversation.py          : single conversation abstraction
│   │   │   ├── conversation_manager.py  : manages multiple conversations / chats
│   │   │   └── message.py               : message abstraction
│   │   ├── prompts                  : multilingual prompt formatting
│   │   │   ├── templates                : language-specific prompt templates (en, fr)
│   │   │   └── prompt_formatter.py      : builds task-specific prompts
│   │   ├── tools                    : tool / function-calling support
│   │   │   ├── llm_tool.py              : LLM-as-a-tool wrapper
│   │   │   ├── openweathermap_tool.py   : tool calling the OpenWeatherMap API
│   │   │   ├── rag_tool.py              : retrieval-augmented-generation tool
│   │   │   ├── tool_executor.py         : safe python execution for tool calling
│   │   │   └── tool.py                  : interfaces for tool definition
│   │   ├── workflows                : experimental multi-step workflow engine
│   │   │   └── nodes                    : typed workflow nodes (llm, condition, loop, iterator, web, ...)
│   │   ├── base_llm.py              : abstract base class for LLM models (`BaseLLM`)
│   │   ├── mistral3.py             : Mistral-3 multimodal (text + image) implementation
│   │   ├── multimodal.py          : multimodal mixin / helpers
│   │   ├── streaming.py           : token-streaming utilities
│   │   └── text_generator.py       : generative language model (`TextGenerator`)
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models        : pretrained / converted checkpoints (e.g. TensorRT-LLM engines)
├── tests                    : `pytest` suite mirroring the `utils` / `loggers` and model tree*
├── utils                    : shared data-processing / keras utilities*
├── example_llm.ipynb        : notebook illustrating the LLM tasks (Q&A, RAG, function calling) + TRT-LLM engine creation
├── example_workflows.ipynb  : notebook demonstrating the experimental workflow engine
├── CHANGELOG.md             : project-specific changelog
├── CITATION.cff             : citation metadata
├── CONTRIBUTING.md          : contribution guidelines
├── INSTALLATION.md          : shared GPU environment installation guide
├── LICENCE                  : project license file
├── pyproject.toml           : project metadata, dependencies and tooling config
├── README.md                : this file
└── __version__.py           : single-source project version
```

\* The `loggers`, `utils`, `custom_train_objects`, `models/core` and most of the `tests` tree are the shared foundation maintained in the [base_dl_project](https://github.com/yui-mhcp/base_dl_project) and [data_processing](https://github.com/yui-mhcp/data_processing) repositories — check them for more information on these modules / main classes.

## Available features

- **Text generation** (module `models.nlu`) :

| Feature           | Function / class    | Description |
| :---------------- | :------------------ | :---------- |
| Text generation   | `TextGenerator`     | Generic generative LLM with streaming / batched inference   |
| Question-Answering | `answer` / `infer` | Answer questions, optionally grounded on provided documents |
| RAG               | `infer(documents=...)` | Retrieval-Augmented Generation over a set of documents   |
| Function calling  | `infer(tools=...)`  | Let the model call tools / execute code to answer          |
| Conversations     | `models.nlu.conversations` | Multi-turn chat / message-history management        |
| Multimodal        | `Mistral3`          | Text + image inputs (multimodal generation)                |
| Workflows         | `models.nlu.workflows` | Experimental multi-step workflow engine                 |

The `example_llm.ipynb` and `example_workflows.ipynb` notebooks provide concrete demonstrations of these features.

## Installation and usage

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/language_models.git`
2. Go to the root of this repository : `cd language_models`
3. Install the package with a backend : `pip install -e .[tf]` (or `pip install -e .[torch]`)
4. Open the `example_llm.ipynb` notebook and follow the instructions !

Like the [base_dl_project](https://github.com/yui-mhcp/base_dl_project), this project runs real models, so it needs a **keras backend**. The text and `transformers` dependencies (required by every LLM) are installed by default — you only pick the backend extra matching your setup :

```bash
pip install -e .[tf]         # keras + tensorflow (+ torch for checkpoint conversion)
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# optional helpers :
pip install -e .[image]      # cv2 / pillow-based utilities for multimodal image inputs (utils/image)
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas)
pip install -e .[dev]        # test tooling (pytest & plugins)

# extras can be combined :
pip install -e .[tf,image]
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the notebooks.

**Important Note** : for **accelerated inference**, the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) support targets `tensorrt_llm < 1.3` on a `python 3.12` environment ; it is installed separately from the NVIDIA package index — see the [installation guide](INSTALLATION.md) (which also notes the `1.2.1` int8 caveat on Blackwell) and the `example_llm.ipynb` notebook for the engine-creation steps. ;)

### Quickstart

```python
from models.nlu import TextGenerator

# This example requires a compiled TRT-LLM engine (cf the example_llm notebook)
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model_dir  = model_name.split('/')[1].replace('-', '_')
path = os.path.expanduser('~/.cache/tensorrt_llm/{}/'.format(get_module_version('tensorrt_llm'))) + model_dir
name = 'trtllm-{}_'.format(get_module_version('tensorrt_llm')) + model_dir.lower()

# 1. Build a generative model (downloaded / converted from the `transformers` hub)
model = TextGenerator(
    'multi', name = name, tokenizer = model_name, path = path, kv_cache_free_gpu_memory = 2048, runtime = 'trt_llm'
)

# 2. Answer a simple question
res = model.answer('Hello World !', messages = [])
print(res['predicted'])
```

See the `example_llm.ipynb` notebook for the full walkthrough (Q&A, summarization, translation, entity extraction, RAG and function calling), and `example_workflows.ipynb` for the workflow engine. :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list :

- [x] Make the TO-DO list
- [x] Make an installation guide for `TensorRT-LLM`
- [x] Implement a wrapper around `tensorrt_llm.ModelRunner(Cpp)`
- [x] Support the streaming mode with a `streaming_callback` argument
- [x] Support batched inference
- [x] Refactor the `html` processing method to have a common return structure for all documents and web-search results
- [x] Support discussion handling (i.e., by saving and forwarding previous messages)
- [ ] Save/load conversations/chats in the `infer` method
- [x] Support function calling
- [x] Support workflows
- [x] Support multi-modality (e.g., text + image --> text)
- [x] Support chunking documents with overlap between chunks
- [x] Support grouping documents by sections/... for better chunks
- [x] Define custom prompts for standard `NLU` tasks, with an appropriate method :
    - [x] Question-Answering (Q&A)
    - [x] Machine Translation
    - [x] Summarization
    - [x] Text reformulation
    - [x] Entity extraction
    - [x] Retriever-Augmented Generator (RAG)
    - [x] Function calling

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
