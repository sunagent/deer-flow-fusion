from codegraph_rag import CodeGraphRAG


def main() -> None:
    rag = CodeGraphRAG(".", auto_index=True)
    results = rag.search(
        "find config loading function",
        top_k=5,
        language="python",
    )
    for result in results:
        print(
            f"{result.score:.4f}",
            result.qualified_name or result.symbol_name,
            f"{result.file_path}:{result.line_number}",
        )


if __name__ == "__main__":
    main()
