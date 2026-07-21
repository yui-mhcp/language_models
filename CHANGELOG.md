# :yum: CHANGELOG :yum:

This file tracks the major updates of the [language_models](https://github.com/yui-mhcp/language_models) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the LLM stack, built on top of the shared `BaseModel` foundation (see the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) and the `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .[tf]` (or `[torch]`) and pick the optional helpers you need (`image`, `datasets`, `dev`)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The minimum supported Python is now `3.12`, and the version is single-sourced in `__version__`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured (inherited from `base_dl_project`) : the old `models/interfaces/base_*_model.py` classes are gone. `BaseModel` now lives in `models/core/base_model.py` and is composed from small, reusable **mixins** in `models/core/mixins/` (`audio`, `image`, `text`, `classification`, `processing`, `checkpoint`, `training`)
- **[BREAKING CHANGE]** the LLM models now live in `models/nlu/` : `base_llm.py` (abstract `BaseLLM`), `text_generator.py` (the generative `TextGenerator`) and `mistral3.py` (the `Mistral3` multimodal text + image model)

### Minor updates

- **Conversation management** has been reworked into `models/nlu/conversations/` (`conversation`, `conversation_manager`, `message`) with pluggable `conv_item_selectors` (naive, last-messages and RAG-based history selection)
- **Prompt formatting** now lives in `models/nlu/prompts/` : a `prompt_formatter` builds task-specific prompts from multilingual `templates/` (`en`, `fr`)
- A new **experimental workflow engine** has been added in `models/nlu/workflows/`, composing typed `nodes/` (`llm`, `condition`, `loop`, `iterator`, `extractor`, `web`, ...) — demonstrated in the new `example_workflows.ipynb` notebook
- `TensorRT-LLM` accelerated inference has been updated to the `1.2.1` library (Python 3.12), with support for streaming and batched inference
- The `example_llm.ipynb` notebook has been reworked around the new `TextGenerator` interface (Q&A, summarization, translation, entity extraction, RAG and function calling)
