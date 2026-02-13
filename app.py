from src.data_loader import process_all_pdfs, split_documents
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever, LLM, rag_advanced, AdvancedRAGPipeline

# Example usage
if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline...\n")
    
    # Step 1: Load all PDFs from data directory
    print("📄 Loading PDFs...")
    all_documents = process_all_pdfs("./data")
    
    # Step 2: Split documents into chunks
    print("\n✂️  Splitting documents into chunks...")
    chunks = split_documents(all_documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Initialize embedding manager
    print("\n🧠 Initializing embedding manager...")
    embedding_manager = EmbeddingManager()
    
    # Step 4: Initialize vector store
    print("\n🗄️  Initializing Qdrant vector store...")
    vector_store = VectorStore(embedding_manager, collection_name="pdf_documents")
    
    # Step 5: Generate embeddings and add to vector store
    print("\n⚙️  Generating embeddings and adding to vector store...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)
    
    # Step 6: Initialize retriever and LLM
    print("\n🔍 Initializing retriever and LLM...")
    retriever = RAGRetriever(vector_store, embedding_manager)
    llm = LLM(model_name="gemma3:4b")
    
    # Step 7: Run sample queries
    print("\n" + "="*60)
    print("🤖 Running RAG Queries")
    print("="*60)
    
    queries = [
        "What did Aditya do at Neilsoft?",
        "What technologies is Aditya interested in?"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        result = rag_advanced(query, retriever, llm, top_k=3, min_score=0.1, return_context=True)
        print(f"💬 Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        if result['sources']:
            print(f"📚 Sources: {[s['source'] for s in result['sources']]}")
    
    # Step 8: Test advanced pipeline
    print("\n" + "="*60)
    print("🎯 Testing Advanced RAG Pipeline")
    print("="*60)
    
    adv_pipeline = AdvancedRAGPipeline(retriever, llm)
    advanced_result = adv_pipeline.query(
        "Summarize Aditya's professional background",
        top_k=3,
        min_score=0.1,
        summarize=True
    )
    print(f"\n📋 Full Answer with Citations:\n{advanced_result['answer']}")
    if advanced_result['summary']:
        print(f"\n📝 Summary: {advanced_result['summary']}")
