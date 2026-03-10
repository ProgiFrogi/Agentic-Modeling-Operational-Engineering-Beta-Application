from openai import OpenAI

from rag.pipeline import KaggleRAGPipeline


# Example usage
def main():
    """Build an index from Kaggle and demonstrate a search."""
    code_describe = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    pipeline = KaggleRAGPipeline(code_describe, "qwen-q4_k", 1000, 200)

    try:
        print("Building index from Kaggle...")
        pipeline.build_index_from_kaggle(
            query="house",
            n_competitions=40,
            notebooks_per_comp=5,
        )
    except Exception as e:
        print(f"Kaggle indexing skipped: {e}")

    print("\nSearch demo:")
    results = pipeline.search(
        query="Feature selection examples",
        n_results=5,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['chunk_type']}] {r['source_title']} (score={r['similarity_score']:.3f}) - {r['content']}")


if __name__ == "__main__":
    main()
