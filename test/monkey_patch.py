def patch_openai_embeddings_llm():
    from graphrag.llm.openai.openai_embeddings_llm import OpenAIEmbeddingsLLM
    import ollama

    async def _execute_llm(self, input, **kwargs):
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model=self.configuration.model, prompt=inp)
            embedding_list.append(embedding["embedding"])
        return embedding_list

    OpenAIEmbeddingsLLM._execute_llm = _execute_llm
