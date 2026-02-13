from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import time


class RAGRetriever:
    """Handles query-based retrieval from the Qdrant vector store"""
    
    def __init__(self, vector_store, embedding_manager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            # Use query_points instead of search
            response = self.vector_store.client.query_points(
                collection_name=self.vector_store.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold  # Qdrant can filter by score natively!
            )

            retrieved_docs = []
            # Note: query_points returns an object with a .points attribute
            for i, result in enumerate(response.points):
                payload = result.payload
                retrieved_docs.append({
                    'id': result.id,
                    'content': payload.get('page_content', ''),  # Matches the key from our previous 'upsert'
                    'metadata': payload.get('metadata', {}),
                    'similarity_score': result.score,
                    'rank': i + 1
                })

            return retrieved_docs         
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


class LLM:
    def __init__(self, model_name: str = "gemma3:4b"):
        """
        Initialize LLM
        
        Args:
            model_name: AI model name (gemma3:4b)
        """
        self.model_name = model_name
        
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0.6,
            num_predict=512
        )
        
        print(f"Initialized LLM with model: {self.model_name}")

    def invoke(self, input_data, **kwargs):
        return self.llm.invoke(input_data, **kwargs)
    
    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length
            
        Returns:
            Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""
        
        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"


def rag_simple(query: str, retriever: RAGRetriever, llm: LLM, top_k: int = 3) -> str:
    """Simple RAG pipeline: retrieve context + generate response"""
    # Retrieve the context
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found to answer the question."
    
    # Generate the answer
    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def rag_advanced(query: str, retriever: RAGRetriever, llm: LLM, top_k: int = 5, 
                min_score: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])
    
    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output


class AdvancedRAGPipeline:
    def __init__(self, retriever: RAGRetriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, 
             stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        """
        Execute a query through the advanced RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of top results to retrieve
            min_score: Minimum similarity score threshold
            stream: Whether to stream the response
            summarize: Whether to generate a summary
            
        Returns:
            Dictionary containing answer, sources, summary, and history
        """
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            
            # Generate answer
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }


# Example usage
if __name__ == "__main__":
    from embedding import EmbeddingManager
    from vectorstore import VectorStore
    from langchain_core.documents import Document
    
    # Initialize components
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(embedding_manager)
    llm = LLM(model_name="gemma3:4b")
    
    # Create sample documents and add to vector store
    sample_docs = [
        Document(page_content="Python is a popular programming language for AI", metadata={"source_file": "ai.pdf", "page": 1}),
        Document(page_content="LLMs are trained on large amounts of text data", metadata={"source_file": "llm.pdf", "page": 2}),
        Document(page_content="RAG systems combine retrieval and generation", metadata={"source_file": "rag.pdf", "page": 3}),
    ]
    
    texts = [doc.page_content for doc in sample_docs]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(sample_docs, embeddings)
    
    # Initialize retriever
    retriever = RAGRetriever(vector_store, embedding_manager)
    
    # Test simple RAG pipeline
    query = "What is RAG?"
    print(f"\n🔍 Query: {query}")
    result = rag_simple(query, retriever, llm, top_k=2)
    print(f"\n📝 Answer: {result}")
    
    # Test advanced RAG pipeline
    print(f"\n{'='*60}")
    adv_result = rag_advanced(query, retriever, llm, top_k=2, min_score=0.1, return_context=True)
    print(f"\n📝 Advanced Answer: {adv_result['answer']}")
    print(f"\n📊 Confidence: {adv_result['confidence']:.4f}")
    print(f"\n📚 Sources: {adv_result['sources']}")