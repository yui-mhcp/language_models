# :yum: Language Models

**NEW** This repository proposes an all-in-one class for `Large Language Models (LLM)` ! The objective is to progressively integrate new features to enhance this initial implementation of a `Retriever Augmented Generator (RAG)` system. This project is still experimental, and will largely evolve in future updates to integrate new features, such as `function calling`, web search, document tokenization, ...

**Important** : this project will **not** re-implement all the existing LLM. Instead, this is currently built upon the highly optimized [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) library that offers large model support with the best performances compared to other libraries !*

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

*The comparison has been made with the `Mistral-7B` architecture between the `transformers` library (that uses `pytorch`) (48 tokens/sec), `TensorRT-LLM` (56 tokens/sec), and the `tensorflow` implementation from this repository executed with `XLA` (52 tokens/sec). These performances were observed based on some practical experiments, and is **not** a rigorous benchmark ;)


## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers               : custom layer implementations
│   ├── transformers         : transformer architecture implementations
│   ├── common_blocks.py     : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py  : utilities for text and sequence generation
│   ├── hparams.py           : hyperparameter management
│   └── simple_models.py     : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing
├── loggers                  : logging utilities for tracking experiment progress
├── models                   : main directory for model classes
│   ├── interfaces           : directories for interface classes
│   ├── nlu                  : OCR implementations
│   │   ├── conversations        : general module handling conversation management
│   │   │   ├── base_chat.py         : base interface for messages, conversations and chats
│   │   │   └── message_selector.py  : interfaces for message selection to build the message history passed to the LLM
│   │   ├── prompts              : general module handling prompts formats
│   │   │   ├── base_prompt.py       : base interface to define multilingual prompts
│   │   │   └── task_prompts.py      : generic prompts formatted for different tasks
│   │   ├── tools                : general module for tools support
│   │   │   ├── openweathermap_tool.py : tool calling the OWM api
│   │   │   ├── tool_executor.py       : safe python script execution for tool calling
│   │   │   └── tool.py                : interfaces for tool definition
│   │   ├── base_language_model.py : abstract class for LM models
│   │   └── text_generator.py      : implementation of generative language models
│   └── weights_converter.py : utilities to convert weights between different models
├── tests                    : unit and integration tests for model validation
├── utils                    : utility functions for data processing and visualization
├── LICENCE                  : project license file
├── example_llm.ipynb        : notebook illustrating different language models tasks + TRT-LLM engine creation
├── README.md                : this file
└── requirements.txt         : required packages
```

## Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

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
- [ ] Support workflows
- [ ] Support multi-modality (e.g., text + image --> text)
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

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```