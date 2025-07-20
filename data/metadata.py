def metadata(chunks, source="data\Raw\ctdt_vnu.pdf"):
    return [{"content": chunk, "metadata": {"source": source}} for chunk in chunks]
