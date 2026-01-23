"""
LLM generator for legal question answering
Handles prompt construction and response generation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.rag.reranker import RerankResult
from src.config import config


@dataclass
class GenerationResult:
    """Result from LLM generation"""
    answer: str
    sources: List[Dict[str, Any]]
    prompt_used: str
    model_used: str


# Legal domain prompts
SYSTEM_PROMPT = """You are JudicAIry, an expert legal research assistant specializing in U.S. Supreme Court jurisprudence. Your role is to provide accurate, well-cited answers based on Supreme Court opinions and legal precedents.

Guidelines:
1. Always cite specific cases when making legal claims
2. Distinguish between majority opinions, concurrences, and dissents
3. Explain legal concepts in clear, accessible language
4. Acknowledge when information is uncertain or when cases may have been overruled
5. Reference the specific sections of opinions (syllabus, holding, reasoning) when relevant
6. Be precise about legal terminology and constitutional provisions

You have access to excerpts from Supreme Court opinions. Base your answers primarily on these sources, but you may use your general legal knowledge to provide context."""

ANSWER_PROMPT = """Based on the following excerpts from U.S. Supreme Court opinions, please answer the user's question.

CONTEXT FROM COURT OPINIONS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, comprehensive answer based on the sources above
- Cite specific cases using their names and years
- If the sources don't fully address the question, say so
- Structure your answer with clear reasoning
- Include relevant quotes when helpful

ANSWER:"""

SUMMARIZE_PROMPT = """Summarize the following Supreme Court opinion excerpt. Focus on:
1. The key legal issue(s)
2. The Court's holding
3. The main reasoning
4. Any important precedents cited or established

OPINION EXCERPT:
{text}

SUMMARY:"""


class LegalGenerator:
    """
    LLM generator for legal Q&A
    
    Supports multiple backends:
    - HuggingFace Transformers (local)
    - OpenAI API
    - Ollama (local)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_openai: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        self.model_name = model_name or config.LLM_MODEL
        self.use_openai = use_openai
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._pipeline = None
        
        if use_openai and config.OPENAI_API_KEY:
            self._init_openai()
        else:
            self._init_local()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.model_name = "gpt-4-turbo-preview"
            print(f"Using OpenAI: {self.model_name}")
        except ImportError:
            print("OpenAI package not installed, falling back to local model")
            self._init_local()
    
    def _init_local(self):
        """Initialize local HuggingFace model"""
        try:
            from transformers import pipeline
            print(f"Loading local model: {self.model_name}")
            print("Note: This may take a while and require significant GPU memory")
            
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                torch_dtype="auto"
            )
            print("Local model loaded successfully")
        except Exception as e:
            print(f"Could not load local model: {e}")
            print("Generator will return placeholder responses")
            self._pipeline = None
    
    def generate(
        self,
        question: str,
        context_results: List[RerankResult],
        system_prompt: str = None
    ) -> GenerationResult:
        """
        Generate an answer based on retrieved context
        
        Args:
            question: User's question
            context_results: Reranked retrieval results
            system_prompt: Custom system prompt (optional)
            
        Returns:
            GenerationResult with answer and sources
        """
        # Build context string
        context = self._build_context(context_results)
        
        # Build prompt
        prompt = ANSWER_PROMPT.format(
            context=context,
            question=question
        )
        
        system = system_prompt or SYSTEM_PROMPT
        
        # Generate response
        if self._client:  # OpenAI
            answer = self._generate_openai(system, prompt)
        elif self._pipeline:  # Local model
            answer = self._generate_local(system, prompt)
        else:
            answer = self._generate_placeholder(question, context_results)
        
        # Build sources list
        sources = [
            {
                "case_name": r.metadata.get("case_name", "Unknown"),
                "citation": r.metadata.get("citation", ""),
                "section": r.metadata.get("section_type", "body"),
                "excerpt": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "score": r.combined_score
            }
            for r in context_results
        ]
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            prompt_used=prompt,
            model_used=self.model_name
        )
    
    def _build_context(self, results: List[RerankResult]) -> str:
        """Build context string from retrieval results"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            case_name = result.metadata.get("case_name", "Unknown Case")
            section = result.metadata.get("section_type", "")
            citation = result.metadata.get("citation", "")
            
            header = f"[Source {i}] {case_name}"
            if citation:
                header += f" ({citation})"
            if section:
                header += f" - {section.title()}"
            
            context_parts.append(f"{header}\n{result.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_openai(self, system: str, prompt: str) -> str:
        """Generate using OpenAI API"""
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
    
    def _generate_local(self, system: str, prompt: str) -> str:
        """Generate using local model"""
        full_prompt = f"{system}\n\n{prompt}"
        
        result = self._pipeline(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            return_full_text=False
        )
        
        return result[0]["generated_text"]
    
    def _generate_placeholder(
        self,
        question: str,
        results: List[RerankResult]
    ) -> str:
        """Generate a placeholder response when no LLM is available"""
        sources = [r.metadata.get("case_name", "Unknown") for r in results[:3]]
        
        return f"""Based on the retrieved Supreme Court opinions, here are the most relevant sources for your question:

**Question:** {question}

**Relevant Cases:**
{chr(10).join(f"- {s}" for s in sources)}

**Note:** This is a placeholder response. To get full AI-generated answers, please configure either:
1. An OpenAI API key (OPENAI_API_KEY environment variable)
2. A local LLM model with sufficient resources

The retrieved case excerpts have been indexed and are ready for semantic search. You can explore the sources directly through the search API."""
    
    def summarize(self, text: str) -> str:
        """Summarize a legal text"""
        prompt = SUMMARIZE_PROMPT.format(text=text)
        
        if self._client:
            return self._generate_openai(SYSTEM_PROMPT, prompt)
        elif self._pipeline:
            return self._generate_local(SYSTEM_PROMPT, prompt)
        else:
            return f"Summary not available (no LLM configured).\n\nOriginal text preview:\n{text[:500]}..."
