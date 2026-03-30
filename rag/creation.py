from rag.pipeline import KaggleRAGPipeline


# Example usage
def main():
    """Build an index from Kaggle and demonstrate a search."""
    pipeline = KaggleRAGPipeline(1000, 200)

    try:
        print("Building index from Kaggle...")
        pipeline.build_index_from_kaggle(
            query="house",
            n_competitions=20,
            notebooks_per_comp=10,
        )
        pipeline.build_index_from_kaggle(
            query="prediction",
            n_competitions=20,
            notebooks_per_comp=10,
        )
        pipeline.build_index_from_kaggle(
            query="regression",
            n_competitions=30,
            notebooks_per_comp=10,
        )
    except Exception as e:
        print(f"Kaggle indexing skipped: {e}")

    # print("\nSearch demo:")
    # results = pipeline.search(
    #     query="ANOVA F-value feature selection",
    #     n_results=5,
    #     chunk_type=ChunkType.CODE_SNIPPET,
    # )
    # for i, r in enumerate(results, 1):
    #     print(f"{i}. [{r['chunk_type']}] {r['source_title']} (score={r['similarity_score']:.3f}) - {r['content']}")


if __name__ == "__main__":
    main()
