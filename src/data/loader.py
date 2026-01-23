"""
Data loader for SCOTUS datasets
Handles CaseSumm from HuggingFace and SCDB metadata
"""

import json
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List
from dataclasses import dataclass, field

from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

from src.config import config


@dataclass
class SCOTUSCase:
    """Represents a Supreme Court case"""
    case_id: str
    case_name: str
    docket_number: Optional[str] = None
    decision_date: Optional[str] = None
    term: Optional[str] = None
    
    # Opinion content
    opinion_text: str = ""
    syllabus: str = ""  # Official summary
    
    # Metadata
    citation: Optional[str] = None
    legal_provisions: List[str] = field(default_factory=list)
    issue_areas: List[str] = field(default_factory=list)
    decision_direction: Optional[str] = None  # liberal/conservative
    majority_votes: Optional[int] = None
    minority_votes: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "docket_number": self.docket_number,
            "decision_date": self.decision_date,
            "term": self.term,
            "opinion_text": self.opinion_text,
            "syllabus": self.syllabus,
            "citation": self.citation,
            "legal_provisions": self.legal_provisions,
            "issue_areas": self.issue_areas,
            "decision_direction": self.decision_direction,
            "majority_votes": self.majority_votes,
            "minority_votes": self.minority_votes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCOTUSCase":
        """Create from dictionary"""
        return cls(
            case_id=data.get("case_id", ""),
            case_name=data.get("case_name", ""),
            docket_number=data.get("docket_number"),
            decision_date=data.get("decision_date"),
            term=data.get("term"),
            opinion_text=data.get("opinion_text", ""),
            syllabus=data.get("syllabus", ""),
            citation=data.get("citation"),
            legal_provisions=data.get("legal_provisions", []),
            issue_areas=data.get("issue_areas", []),
            decision_direction=data.get("decision_direction"),
            majority_votes=data.get("majority_votes"),
            minority_votes=data.get("minority_votes"),
            metadata=data.get("metadata", {})
        )


class DataLoader:
    """Load and process SCOTUS datasets"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or config.RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_casesumm(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        streaming: bool = False
    ) -> Iterator[SCOTUSCase]:
        """
        Load CaseSumm dataset from HuggingFace
        
        Args:
            split: Dataset split (train, validation, test)
            limit: Limit number of cases to load
            streaming: Use streaming mode for large datasets
            
        Yields:
            SCOTUSCase objects
        """
        print(f"Loading CaseSumm dataset (split={split})...")
        
        try:
            dataset = load_dataset(
                config.CASESUMM_DATASET,
                split=split,
                streaming=streaming,
                cache_dir=str(self.cache_dir),
                token=config.HUGGINGFACE_TOKEN
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting to load without authentication...")
            dataset = load_dataset(
                config.CASESUMM_DATASET,
                split=split,
                streaming=streaming,
                cache_dir=str(self.cache_dir)
            )
        
        count = 0
        for item in tqdm(dataset, desc="Processing cases"):
            if limit and count >= limit:
                break
                
            case = self._parse_casesumm_item(item)
            if case:
                yield case
                count += 1
                
        print(f"Loaded {count} cases from CaseSumm")
    
    def _parse_casesumm_item(self, item: Dict[str, Any]) -> Optional[SCOTUSCase]:
        """Parse a CaseSumm dataset item into SCOTUSCase"""
        try:
            # Extract case ID from available fields
            case_id = item.get("id") or item.get("case_id") or str(hash(item.get("case_name", "")))
            
            case = SCOTUSCase(
                case_id=str(case_id),
                case_name=item.get("case_name", "Unknown"),
                docket_number=item.get("docket_number"),
                decision_date=item.get("decision_date"),
                term=item.get("term"),
                opinion_text=item.get("opinion", "") or item.get("text", ""),
                syllabus=item.get("syllabus", "") or item.get("summary", ""),
                citation=item.get("citation") or item.get("us_citation"),
                metadata={
                    "source": "casesumm",
                    "raw_fields": list(item.keys())
                }
            )
            return case
        except Exception as e:
            print(f"Error parsing item: {e}")
            return None
    
    def load_scdb_metadata(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load Supreme Court Database metadata
        
        Args:
            filepath: Path to SCDB CSV file
            
        Returns:
            DataFrame with case metadata
        """
        if filepath is None:
            filepath = self.cache_dir / "scdb_modern.csv"
            
        if not filepath.exists():
            print(f"SCDB file not found at {filepath}")
            print("Please download from http://scdb.wustl.edu/")
            return pd.DataFrame()
            
        print(f"Loading SCDB metadata from {filepath}...")
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"Loaded {len(df)} cases from SCDB")
        return df
    
    def merge_with_scdb(
        self,
        cases: List[SCOTUSCase],
        scdb_df: pd.DataFrame,
        match_on: str = "docket_number"
    ) -> List[SCOTUSCase]:
        """
        Merge CaseSumm cases with SCDB metadata
        
        Args:
            cases: List of SCOTUSCase objects
            scdb_df: SCDB DataFrame
            match_on: Field to match on
            
        Returns:
            Enriched list of cases
        """
        if scdb_df.empty:
            return cases
            
        print("Merging cases with SCDB metadata...")
        
        # Create lookup dict from SCDB
        scdb_lookup = {}
        for _, row in scdb_df.iterrows():
            key = str(row.get(match_on, ""))
            if key:
                scdb_lookup[key] = row.to_dict()
        
        enriched = []
        for case in tqdm(cases, desc="Enriching cases"):
            match_value = getattr(case, match_on, None)
            if match_value and str(match_value) in scdb_lookup:
                scdb_data = scdb_lookup[str(match_value)]
                
                # Enrich with SCDB fields
                case.legal_provisions = self._parse_provisions(scdb_data)
                case.issue_areas = self._parse_issue_areas(scdb_data)
                case.decision_direction = scdb_data.get("decisionDirection")
                case.majority_votes = scdb_data.get("majVotes")
                case.minority_votes = scdb_data.get("minVotes")
                case.metadata["scdb"] = scdb_data
                
            enriched.append(case)
            
        return enriched
    
    def _parse_provisions(self, scdb_data: Dict) -> List[str]:
        """Extract legal provisions from SCDB data"""
        provisions = []
        for i in range(1, 6):
            prov = scdb_data.get(f"lawType{i}")
            if prov and pd.notna(prov):
                provisions.append(str(prov))
        return provisions
    
    def _parse_issue_areas(self, scdb_data: Dict) -> List[str]:
        """Extract issue areas from SCDB data"""
        issue_area = scdb_data.get("issueArea")
        if issue_area and pd.notna(issue_area):
            # SCDB issue area codes
            issue_map = {
                1: "Criminal Procedure",
                2: "Civil Rights",
                3: "First Amendment",
                4: "Due Process",
                5: "Privacy",
                6: "Attorneys",
                7: "Unions",
                8: "Economic Activity",
                9: "Judicial Power",
                10: "Federalism",
                11: "Interstate Relations",
                12: "Federal Taxation",
                13: "Miscellaneous"
            }
            return [issue_map.get(int(issue_area), f"Issue Area {issue_area}")]
        return []
    
    def save_processed_cases(
        self,
        cases: List[SCOTUSCase],
        filepath: Optional[Path] = None
    ):
        """Save processed cases to JSON"""
        if filepath is None:
            filepath = config.PROCESSED_DATA_DIR / "cases.json"
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = [case.to_dict() for case in cases]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(cases)} cases to {filepath}")
    
    def load_processed_cases(
        self,
        filepath: Optional[Path] = None
    ) -> List[SCOTUSCase]:
        """Load previously processed cases"""
        if filepath is None:
            filepath = config.PROCESSED_DATA_DIR / "cases.json"
            
        if not filepath.exists():
            return []
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [SCOTUSCase.from_dict(item) for item in data]


def download_casesumm_sample(limit: int = 100) -> List[SCOTUSCase]:
    """Download a sample of CaseSumm for testing"""
    loader = DataLoader()
    cases = list(loader.load_casesumm(split="train", limit=limit))
    loader.save_processed_cases(cases)
    return cases
