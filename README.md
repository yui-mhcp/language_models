# :yum: Language Models

**NEW** This repository proposes an all-in-one class for `Large Language Models (LLM)` ! The objective is to progressively integrate new features to enhance this initial implementation of a `Retriever Augmented Generator (RAG)` system. This project is still experimental, and will largely evolve in future updates to integrate new features, such as `function calling`, web search, document tokenization, ...

**Important** : this project will **not** re-implement all the existing LLM. Instead, this is currently built upon the highly optimized [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) library that offers large model support with the best performances compared to other libraries !*

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

*The comparison has been made with the `Mistral-7B` architecture between the `transformers` library (that uses `pytorch`) (48 tokens/sec), `TensorRT-LLM` (56 tokens/sec), and the `tensorflow` implementation from this repository executed with `XLA` (52 tokens/sec). These performances were observed based on some practical experiments, and is **not** a rigorous benchmark ;)


## Project structure

```bash
├── custom_architectures
├── custom_layers
├── custom_train_objects
├── loggers
├── models
│   ├── encoder
│   │   └── text_encoder.py     : text encoder used in the RAG pipeline
│   ├── nlu
│   │   ├── base_language_model.py  : base abstraction for language models
│   │   ├── prompts.py          : default prompts for different tasks / languages
│   │   └── text_generator.py   : text generator (large language models)
│   ├── utils
│   │   └── trt_llm_runner.py   : wrapper for the `tensorrt_llm.ModelRunner` class
├── pretrained_models
├── unitests
├── utils
├── speaker_verification.ipynb
└── information_retrieval.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

**Important Note** : this project is the keras 3 extension of the [NLP](https://github.com/yui-mhcp/nlp) project. When this project will be in a more stable stage, the older one will be removed in favor of this one. 

## Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Make an installation guide for `TensorRT-LLM`
- [x] Implement a wrapper around `tensorrt_llm.ModelRunner(Cpp)`
- [x] Support the streaming mode with a `streaming_callback` argument
- [x] Support batched inference
- [x] Refactor the `html` processing method to have a common return structure for all documents and web-search results
- [ ] Support discussion handling (i.e., by saving and forwarding previous messages)
- [ ] Support function calling
- [ ] Support multi-modality (e.g., text + image --> text)
- [ ] Support chunking documents with overlap between chunks
- [ ] Support grouping documents by sections/... for better chunks
- [x] Define custom prompts for standard `NLU` tasks, with an appropriate method :
    - [x] Question-Answering (Q&A)
    - [x] Machine Translation
    - [x] Summarization
    - [x] Text reformulation
    - [x] Entity extraction
    - [x] Retriever-Augmented Generator (RAG)
    - [ ] Function calling
    - [ ] Multi-modal inputs

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

Tutorials : 
- [LLama-index tutorial](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/) on information retrieval with dense vectors

Github project :
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) : the TensorRT-LLM library that proposes highly optimized LLM architectures in C++
- [LLama-index](https://github.com/run-llama/llama_index) : well known library for information retrieval
- [langchain](https://github.com/langchain-ai/langchain) : well known library for large language models