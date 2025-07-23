def metadata(chunks, source="data\Raw\Quy-chế-ĐTĐH-3626.pdf"):
    return [{"content": chunk, "metadata": {"source": source}} for chunk in chunks]
