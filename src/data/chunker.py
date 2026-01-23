"""
Document chunking for legal texts
Implements semantic and hierarchical chunking strategies
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

from src.config import config
from src.data.loader import SCOTUSCase


@dataclass
class DocumentChunk:
    """Represents a chunk of a legal document"""
    chunk_id: str
    case_id: str
    case_name: str
    content: str
    chunk_index: int
    total_chunks: int
    
    # Section information
    section_type: str = "body"  # syllabus, background, holding, reasoning, dissent
    
    # Metadata for retrieval
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "chunk_id": self.chunk_id,
            "case_id": self.case_id,
            "case_name": self.case_name,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "section_type": self.section_type,
            "metadata": self.metadata
        }


class DocumentChunker:
    """Chunk legal documents for RAG indexing"""
    
    # Section patterns for legal documents
    SECTION_PATTERNS = {
        "syllabus": r"(?i)^syllabus|^held:|^summary",
        "background": r"(?i)^background|^facts|^procedural history",
        "holding": r"(?i)^holding|^judgment|^decision|^we hold",
        "reasoning": r"(?i)^reasoning|^analysis|^discussion",
        "dissent": r"(?i)^dissent|^dissenting opinion|justice .+ dissenting",
        "concurrence": r"(?i)^concur|^concurring opinion|justice .+ concurring"
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        tokenizer_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Use tiktoken for accurate token counting
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # LangChain splitter for recursive splitting
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
    
    def _token_length(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_case(self, case: SCOTUSCase) -> List[DocumentChunk]:
        """
        Chunk a single case into retrievable segments
        
        Strategy:
        1. Separate syllabus from opinion
        2. Identify sections within opinion
        3. Split each section with overlap
        """
        chunks = []
        chunk_index = 0
        
        # Process syllabus separately (it's a summary)
        if case.syllabus:
            syllabus_chunks = self._chunk_section(
                text=case.syllabus,
                case=case,
                section_type="syllabus",
                start_index=chunk_index
            )
            chunks.extend(syllabus_chunks)
            chunk_index += len(syllabus_chunks)
        
        # Process main opinion
        if case.opinion_text:
            # Try to identify sections
            sections = self._identify_sections(case.opinion_text)
            
            for section_type, section_text in sections:
                section_chunks = self._chunk_section(
                    text=section_text,
                    case=case,
                    section_type=section_type,
                    start_index=chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
            
        return chunks
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """
        Identify logical sections in the opinion text
        Returns list of (section_type, text) tuples
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        sections = []
        current_section = "body"
        current_text = []
        
        for para in paragraphs:
            # Check if this paragraph starts a new section
            new_section = None
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, para[:100]):
                    new_section = section_type
                    break
            
            if new_section and new_section != current_section:
                # Save current section
                if current_text:
                    sections.append((current_section, '\n\n'.join(current_text)))
                current_section = new_section
                current_text = [para]
            else:
                current_text.append(para)
        
        # Don't forget the last section
        if current_text:
            sections.append((current_section, '\n\n'.join(current_text)))
        
        # If no sections were identified, return entire text as body
        if not sections:
            sections = [("body", text)]
            
        return sections
    
    def _chunk_section(
        self,
        text: str,
        case: SCOTUSCase,
        section_type: str,
        start_index: int
    ) -> List[DocumentChunk]:
        """Chunk a section of text"""
        if not text.strip():
            return []
            
        # Use LangChain splitter
        text_chunks = self.splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                chunk_id=f"{case.case_id}_{section_type}_{start_index + i}",
                case_id=case.case_id,
                case_name=case.case_name,
                content=chunk_text,
                chunk_index=start_index + i,
                total_chunks=0,  # Will be updated later
                section_type=section_type,
                metadata={
                    "citation": case.citation,
                    "decision_date": case.decision_date,
                    "term": case.term,
                    "legal_provisions": case.legal_provisions,
                    "issue_areas": case.issue_areas,
                    "token_count": self._token_length(chunk_text)
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def chunk_cases(self, cases: List[SCOTUSCase]) -> List[DocumentChunk]:
        """Chunk multiple cases"""
        from tqdm import tqdm
        
        all_chunks = []
        for case in tqdm(cases, desc="Chunking cases"):
            chunks = self.chunk_case(case)
            all_chunks.extend(chunks)
            
        print(f"Created {len(all_chunks)} chunks from {len(cases)} cases")
        return all_chunks


def create_chunks_from_cases(cases: List[SCOTUSCase]) -> List[DocumentChunk]:
    """Convenience function to chunk cases"""
    chunker = DocumentChunker()
    return chunker.chunk_cases(cases)
