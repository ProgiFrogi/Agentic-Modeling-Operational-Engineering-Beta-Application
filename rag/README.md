# Usage
## Retrieving entries
To retrieve anything from storage you need `VectorStore` instance. It is able to both store and retrieve entries. In parameters you only need to pay attention to `overload_factor`. It controls how many entries will be retrieved, reranked and filtered. For example if `overload_factor=3` and you ask to retrieve 2 entries, it will actually get 6 from storage, rank them and return top 2 most similar to query by reranker.

You will need env variables (prefer absolute paths)
- EMBEDDING_MODEL
- RERANKER_MODEL
- MODELS_HUB (supposed to be "${MODELS_LOCATION}/hub")
- STORAGE_LOCATION (folder where storage is located, directly in this folder must be `chroma.sqlite3` and other chroma folders - otherwise they will be created)

Usage example:
```
storage = VectoreStore()
storage.search_chunks("Filtering of outliers in data", chunk_type=ChunkType.MARKDOWN_CELL, n_results=5)
```
`chunk_type` can be used to retrieve only markdown cells content (ChunkType.MARKDOWN_CELL) or only code (ChunkType.CODE_SNIPPET). If you do not specify it, any will be returned. `CODE_SNIPPET` is further subdivied in classes and functions code but it is not recommended for use. 

Return is a list of dictionaries. As it is formed you can look [here](https://github.com/ProgiFrogi/Agentic-Modeling-Operational-Engineering-Beta-Application/blob/fd496ec83eb03ffc5659ac96dec13fd67842727a/rag/storage.py#L161). Most important field is `content` which is exactly string that was stored in rag. Length of `content` is arbitrary so better cut it.

Db can be taken [here](https://drive.google.com/file/d/1a5XHmaxvlezXrELyiSiArYxX-63-k3jO/view?usp=sharing). Maximum size of `content` in it is 1000 characters.

## Generating database
To generate database or add new content to it, you can run [init](https://github.com/ProgiFrogi/Agentic-Modeling-Operational-Engineering-Beta-Application/blob/rag/rag/init.py). Main function is self explanatory. Pay attention that to run it you will need all env variables to be specified (except ones related to coder)
