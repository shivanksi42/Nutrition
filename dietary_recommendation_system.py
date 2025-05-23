import os
import asyncio
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores.utils import filter_complex_metadata
import aiohttp
import asyncio
from urllib.parse import urljoin, urlparse
import time
import json
import hashlib
import re
from difflib import SequenceMatcher

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBMED_API_KEY = os.getenv('PUBMED_API_KEY')
USDA_API_KEY = os.getenv('USDA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@dataclass
class HealthCondition:
    condition: str
    allergies: List[str]
    severity: Optional[str] = None
    
class ConditionMatcher:
    """Helper class to match similar health conditions"""
    
    def __init__(self):
        # Define condition synonyms and related terms
        self.condition_synonyms = {
            'high blood pressure': ['hypertension', 'elevated blood pressure', 'high bp'],
            'diabetes': ['diabetes mellitus', 'type 2 diabetes', 'diabetic', 'blood sugar'],
            'heart disease': ['cardiovascular disease', 'coronary artery disease', 'cardiac'],
            'obesity': ['overweight', 'weight management', 'weight loss'],
            'high cholesterol': ['hypercholesterolemia', 'elevated cholesterol', 'cholesterol'],
            'osteoporosis': ['bone density', 'bone health', 'calcium deficiency'],
            'anemia': ['iron deficiency', 'low iron', 'iron deficiency anemia']
        }
        
        # Reverse mapping for quick lookup
        self.term_to_conditions = {}
        for main_condition, synonyms in self.condition_synonyms.items():
            self.term_to_conditions[main_condition] = main_condition
            for synonym in synonyms:
                self.term_to_conditions[synonym] = main_condition
    
    def normalize_condition(self, condition: str) -> str:
        """Normalize a condition string for better matching"""
        condition = condition.lower().strip()
        
        # Remove common prefixes/suffixes
        condition = re.sub(r'\b(chronic|acute|severe|mild|moderate)\b', '', condition)
        condition = re.sub(r'\s+', ' ', condition).strip()
        
        return condition
    
    def find_similar_conditions(self, target_condition: str, existing_conditions: List[str], 
                              threshold: float = 0.6) -> List[str]:
        """Find existing conditions that are similar to the target condition"""
        target_normalized = self.normalize_condition(target_condition)
        similar_conditions = []
        
        # First, check for exact synonym matches
        if target_normalized in self.term_to_conditions:
            main_condition = self.term_to_conditions[target_normalized]
            for existing in existing_conditions:
                existing_normalized = self.normalize_condition(existing)
                if existing_normalized in self.term_to_conditions:
                    if self.term_to_conditions[existing_normalized] == main_condition:
                        similar_conditions.append(existing)
        
        # Then, check for partial matches and similar strings
        for existing in existing_conditions:
            existing_normalized = self.normalize_condition(existing)
            
            # Check if either condition contains the other
            if (target_normalized in existing_normalized or 
                existing_normalized in target_normalized):
                if existing not in similar_conditions:
                    similar_conditions.append(existing)
                continue
            
            # Check string similarity
            similarity = SequenceMatcher(None, target_normalized, existing_normalized).ratio()
            if similarity >= threshold:
                if existing not in similar_conditions:
                    similar_conditions.append(existing)
        
        return similar_conditions
    
    def get_condition_keywords(self, condition: str) -> Set[str]:
        """Extract key terms from a condition for vector search"""
        condition_normalized = self.normalize_condition(condition)
        keywords = set()
        
        # Add the condition itself
        keywords.add(condition_normalized)
        
        # Add synonyms if available
        if condition_normalized in self.term_to_conditions:
            main_condition = self.term_to_conditions[condition_normalized]
            keywords.add(main_condition)
            if main_condition in self.condition_synonyms:
                keywords.update(self.condition_synonyms[main_condition])
        
        # Add individual words from the condition
        words = condition_normalized.split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                keywords.add(word)
        
        return keywords    
    

class DataSourceManager:
    """Manages different data sources and their API integrations"""
    
    def __init__(self):
        self.sources = {
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'usda_fdc': 'https://api.nal.usda.gov/fdc/v1/',
            'eatright': 'https://www.eatright.org/',
            'harvard_nutrition': 'https://nutritionsource.hsph.harvard.edu/'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def search_pubmed_api(self, query: str, max_results: int = 15) -> List[Document]:
        """Search PubMed using official API with API key"""
        try:
          
            search_url = f"{self.sources['pubmed']}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance',
                'field': 'title/abstract'
            }

            if PUBMED_API_KEY:
                search_params['api_key'] = PUBMED_API_KEY
            
            search_response = self.session.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
                logger.warning(f"No PubMed results found for query: {query}")
                return []
            
            ids = search_data['esearchresult']['idlist']
            fetch_url = f"{self.sources['pubmed']}efetch.fcgi"
            
            documents = []
            batch_size = 10
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml',
                    'rettype': 'abstract'
                }
                
                if PUBMED_API_KEY:
                    fetch_params['api_key'] = PUBMED_API_KEY
                
                fetch_response = self.session.get(fetch_url, params=fetch_params)
                fetch_response.raise_for_status()
                
                soup = BeautifulSoup(fetch_response.content, 'xml')
                articles = soup.find_all('PubmedArticle')
                
                for article in articles:
                    try:
                        title_elem = article.find('ArticleTitle')
                        abstract_elem = article.find('AbstractText')
                        authors_elem = article.find_all('Author')
                        journal_elem = article.find('Title') 
                        pmid_elem = article.find('PMID')
                        pub_date_elem = article.find('PubDate')
                        
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text().strip()
                        abstract = abstract_elem.get_text().strip() if abstract_elem else "No abstract available"
                        pmid = pmid_elem.get_text().strip() if pmid_elem else "Unknown"
                        journal = journal_elem.get_text().strip() if journal_elem else "Unknown Journal"
                        
                        authors = []
                        for author in authors_elem[:3]: 
                            last_name = author.find('LastName')
                            first_name = author.find('ForeName')
                            if last_name:
                                author_name = last_name.get_text()
                                if first_name:
                                    author_name += f", {first_name.get_text()}"
                                authors.append(author_name)
                        
                        authors_str = "; ".join(authors) if authors else "Not available"
                        
                        pub_year = "Unknown"
                        if pub_date_elem:
                            year_elem = pub_date_elem.find('Year')
                            if year_elem:
                                pub_year = year_elem.get_text()
                        
                        doc_content = f"""
Title: {title}

Abstract: {abstract}

Journal: {journal}
Authors: {authors_str}
Publication Year: {pub_year}
PubMed ID: {pmid}
                        """.strip()
                        
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                'source': 'pubmed_api',
                                'pmid': pmid,
                                'title': title,
                                'journal': journal,
                                'authors': authors_str,  
                                'pub_year': pub_year,
                                'query': query,
                                'timestamp': datetime.now().isoformat(),
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing PubMed article: {e}")
                        continue
                
                await asyncio.sleep(0.5)
            
            logger.info(f"Retrieved {len(documents)} articles from PubMed for query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching PubMed API: {e}")
            return []
    
    async def search_usda_food_data(self, food_items: List[str], condition: str = None) -> List[Document]:
        """Search USDA Food Data Central API for nutritional information"""
        try:
            if not USDA_API_KEY:
                logger.warning("USDA API key not available")
                return []
            
            documents = []
            
            for food_item in food_items:
                search_url = f"{self.sources['usda_fdc']}foods/search"
                search_params = {
                    'query': food_item.strip(),
                    'dataType': ['Foundation', 'SR Legacy'], 
                    'pageSize': 10,
                    'api_key': USDA_API_KEY
                }
                
                response = self.session.get(search_url, params=search_params)
                response.raise_for_status()
                data = response.json()
                
                if 'foods' not in data or not data['foods']:
                    continue
                
                for food in data['foods'][:3]:  
                    try:
                        fdc_id = food.get('fdcId')
                        description = food.get('description', 'Unknown food')
                        
                        detail_url = f"{self.sources['usda_fdc']}food/{fdc_id}"
                        detail_params = {
                            'api_key': USDA_API_KEY,
                            'format': 'full'
                        }
                        
                        detail_response = self.session.get(detail_url, params=detail_params)
                        detail_response.raise_for_status()
                        detail_data = detail_response.json()
                        
                        nutrients = {}
                        food_nutrients = detail_data.get('foodNutrients', [])
                        
                        key_nutrients = {
                            'Energy': 'calories',
                            'Protein': 'protein',
                            'Total lipid (fat)': 'total_fat',
                            'Carbohydrate, by difference': 'carbohydrates',
                            'Fiber, total dietary': 'fiber',
                            'Sugars, total including NLEA': 'sugars',
                            'Sodium, Na': 'sodium',
                            'Potassium, K': 'potassium',
                            'Calcium, Ca': 'calcium',
                            'Iron, Fe': 'iron',
                            'Vitamin C, total ascorbic acid': 'vitamin_c',
                            'Cholesterol': 'cholesterol'
                        }
                        
                        for nutrient in food_nutrients:
                            nutrient_name = nutrient.get('nutrient', {}).get('name', '')
                            nutrient_value = nutrient.get('amount', 0)
                            nutrient_unit = nutrient.get('nutrient', {}).get('unitName', '')
                            
                            if nutrient_name in key_nutrients:
                                nutrients[key_nutrients[nutrient_name]] = f"{nutrient_value} {nutrient_unit}"
                        
                        doc_content = f"""
Food Item: {description}

Nutritional Information (per 100g):
{self._format_nutrients(nutrients)}

Food Category: {food.get('foodCategory', 'Not specified')}
Brand: {food.get('brandOwner', 'Generic')}
Data Source: USDA Food Data Central
FDC ID: {fdc_id}

Health Relevance: This nutritional data is relevant for managing {condition if condition else 'various health conditions'} and dietary planning.
                        """.strip()
                        
                        nutrients_str = "; ".join([f"{k}: {v}" for k, v in nutrients.items()])
                        
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                'source': 'usda_fdc_api',
                                'fdc_id': str(fdc_id),
                                'food_description': description,
                                'food_category': food.get('foodCategory', 'Not specified'),
                                'brand': food.get('brandOwner', 'Generic'),
                                'nutrients_summary': nutrients_str, 
                                'query': food_item,
                                'condition': condition or 'general',
                                'timestamp': datetime.now().isoformat(),
                                'url': f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{fdc_id}/nutrients"
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"Error processing USDA food data: {e}")
                        continue
                
                await asyncio.sleep(0.3)
            
            logger.info(f"Retrieved {len(documents)} food items from USDA FDC")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching USDA Food Data Central: {e}")
            return []
    
    def _format_nutrients(self, nutrients: Dict[str, str]) -> str:
        """Format nutrient information for better readability"""
        if not nutrients:
            return "Nutritional information not available"
        
        formatted = []
        for key, value in nutrients.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return '\n'.join(formatted)
    
    async def get_recommended_foods_for_condition(self, condition: str, allergies: List[str]) -> List[str]:
        """Get recommended foods for a specific health condition"""
        condition_foods = {
            'high blood pressure': ['spinach', 'bananas', 'oats', 'salmon', 'blueberries', 'olive oil'],
            'diabetes': ['quinoa', 'sweet potatoes', 'legumes', 'nuts', 'fish', 'leafy greens'],
            'heart disease': ['salmon', 'avocado', 'olive oil', 'nuts', 'berries', 'whole grains'],
            'high cholesterol': ['oats', 'beans', 'nuts', 'fish', 'olive oil', 'apples'],
            'obesity': ['lean proteins', 'vegetables', 'fruits', 'whole grains', 'legumes'],
            'osteoporosis': ['dairy products', 'leafy greens', 'fish', 'fortified foods'],
            'anemia': ['lean meat', 'spinach', 'legumes', 'fortified cereals', 'seafood']
        }
        
        recommended_foods = []
        condition_lower = condition.lower()
        
        for key, foods in condition_foods.items():
            if any(word in condition_lower for word in key.split()):
                recommended_foods.extend(foods)
        
        if allergies:
            allergy_keywords = {
                'dairy': ['milk', 'cheese', 'yogurt', 'dairy'],
                'nuts': ['nuts', 'almonds', 'walnuts', 'peanuts'],
                'shellfish': ['shrimp', 'crab', 'lobster', 'shellfish'],
                'fish': ['salmon', 'tuna', 'fish', 'seafood'],
                'eggs': ['eggs', 'egg'],
                'gluten': ['wheat', 'oats', 'grains'],
                'soy': ['soy', 'tofu', 'soybeans']
            }
            
            for allergy in allergies:
                allergy_lower = allergy.lower().strip()
                if allergy_lower in allergy_keywords:
                    keywords_to_avoid = allergy_keywords[allergy_lower]
                    recommended_foods = [
                        food for food in recommended_foods
                        if not any(keyword in food.lower() for keyword in keywords_to_avoid)
                    ]
        
        if len(recommended_foods) < 5:
            safe_foods = ['vegetables', 'fruits', 'lean proteins', 'whole grains', 'legumes']
            recommended_foods.extend(safe_foods)
        
        return list(set(recommended_foods))  
    
    async def scrape_harvard_nutrition(self, condition: str) -> List[Document]:
        """Scrape Harvard Nutrition Source for condition-specific information"""
        try:
            search_url = f"https://nutritionsource.hsph.harvard.edu/?s={condition.replace(' ', '+')}"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            documents = []
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:5]: 
                if '/nutrition-source/' in link['href']:
                    article_url = urljoin('https://nutritionsource.hsph.harvard.edu/', link['href'])
                    article_content = self._scrape_article_content(article_url)
                    
                    if article_content:
                        doc = Document(
                            page_content=article_content,
                            metadata={
                                'source': 'harvard_nutrition',
                                'url': article_url,
                                'condition': condition,
                                'timestamp': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error scraping Harvard Nutrition: {e}")
            return []
    
    def _scrape_article_content(self, url: str) -> str:
        """Extract main content from an article URL"""
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            content_selectors = ['.entry-content', '.post-content', 'article', '.content']
            content = ""
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    content = content_div.get_text(strip=True)
                    break
            
            return content[:5000] 
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return ""

class VectorDatabaseManager:
    """Enhanced vector database manager with improved condition matching"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.metadata_file = os.path.join(persist_directory, "knowledge_metadata.json")
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.condition_matcher = ConditionMatcher()
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self._initialize_retrievers()
        # Load existing knowledge metadata
        self.knowledge_metadata = self._load_knowledge_metadata()
        
    def set_retriever_strategy(self, strategy: str = "mmr"):
        """Switch between different retriever strategies"""
        strategies = {
            "similarity": self.similarity_retriever,
            "mmr": self.mmr_retriever,
            "threshold": self.threshold_retriever
        }
        
        if strategy in strategies:
            self.active_retriever = strategies[strategy]
            logger.info(f"Switched to {strategy} retriever strategy")
        else:
            logger.warning(f"Unknown strategy {strategy}, keeping current strategy")
    
    def _initialize_retrievers(self):
        """Initialize different retriever strategies for optimal performance"""
        
        # 1. Basic similarity retriever (fastest)
        self.similarity_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        # 2. MMR (Maximal Marginal Relevance) retriever for diversity
        self.mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 12,  # Fetch more candidates for diversity
                "lambda_mult": 0.7  # Balance between similarity and diversity
            }
        )
        
        # 3. Similarity with score threshold for quality control
        self.threshold_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.5,  # Only return results above this similarity
                "k": 8
            }
        )
        
        # 4. Multi-query retriever (will implement custom logic)
        self.active_retriever = self.mmr_retriever 
    
    
    def _load_knowledge_metadata(self) -> Dict:
        """Load metadata about what knowledge has been stored"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata file: {e}")
                return {"conditions": {}, "last_updated": None}
        return {"conditions": {}, "last_updated": None}
    
    def _save_knowledge_metadata(self):
        """Save metadata about stored knowledge"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.knowledge_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata file: {e}")
    
    def _generate_condition_key(self, condition: str, allergies: List[str]) -> str:
        """Generate a unique key for a condition-allergy combination"""
        # Normalize the condition for consistent keying
        normalized_condition = self.condition_matcher.normalize_condition(condition)
        normalized_allergies = sorted([a.lower().strip() for a in allergies])
        
        # Create a hash for the combination
        key_string = f"{normalized_condition}|{','.join(normalized_allergies)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _find_existing_similar_conditions(self, target_condition: str, allergies: List[str]) -> List[Dict]:
        """Find existing conditions that are similar to the target condition"""
        existing_conditions = []
        
        # Get all stored conditions
        for condition_key, metadata in self.knowledge_metadata["conditions"].items():
            existing_conditions.append({
                'key': condition_key,
                'condition': metadata.get('condition', ''),
                'allergies': metadata.get('allergies', []),
                'metadata': metadata
            })
        
        # Extract just the condition names for matching
        condition_names = [item['condition'] for item in existing_conditions]
        
        # Find similar conditions
        similar_condition_names = self.condition_matcher.find_similar_conditions(
            target_condition, condition_names, threshold=0.6
        )
        
        # Filter to only include those with matching allergies
        matching_conditions = []
        target_allergies_set = set(allergy.lower().strip() for allergy in allergies)
        
        for item in existing_conditions:
            if item['condition'] in similar_condition_names:
                existing_allergies_set = set(allergy.lower().strip() for allergy in item['allergies'])
                
                # Check if allergies match (either exact match or target is subset)
                if target_allergies_set == existing_allergies_set or target_allergies_set.issubset(existing_allergies_set):
                    matching_conditions.append(item)
        
        return matching_conditions
    
    def has_sufficient_knowledge(self, condition: str, allergies: List[str], 
                               min_documents: int = 10, max_age_days: int = 7) -> bool:
        """Enhanced method to check for sufficient knowledge including similar conditions"""
        
        # First, check for exact match
        condition_key = self._generate_condition_key(condition, allergies)
        
        if condition_key in self.knowledge_metadata["conditions"]:
            if self._check_condition_knowledge(condition_key, min_documents, max_age_days):
                logger.info(f"Found exact match for condition: {condition}")
                return True
        
        # Then, check for similar conditions
        similar_conditions = self._find_existing_similar_conditions(condition, allergies)
        
        for similar_condition in similar_conditions:
            condition_key = similar_condition['key']
            if self._check_condition_knowledge(condition_key, min_documents, max_age_days):
                logger.info(f"Found similar condition match: '{similar_condition['condition']}' for target: '{condition}'")
                return True
        
        # Finally, check vector store directly with enhanced search
        if self._check_vector_store_knowledge(condition, allergies, min_documents):
            logger.info(f"Found sufficient knowledge in vector store for: {condition}")
            return True
        
        logger.info(f"Insufficient knowledge found for condition: {condition}")
        return False
    
    def _check_condition_knowledge(self, condition_key: str, min_documents: int, max_age_days: int) -> bool:
        """Check if a specific condition has sufficient and recent knowledge"""
        if condition_key not in self.knowledge_metadata["conditions"]:
            return False
        
        metadata = self.knowledge_metadata["conditions"][condition_key]
        
        # Check if we have enough documents
        if metadata.get("document_count", 0) < min_documents:
            return False
        
        # Check if the knowledge is recent enough
        last_updated = metadata.get("last_updated")
        if last_updated:
            try:
                last_update_date = datetime.fromisoformat(last_updated)
                age_days = (datetime.now() - last_update_date).days
                if age_days > max_age_days:
                    return False
            except Exception as e:
                logger.warning(f"Error parsing last update date: {e}")
                return False
        
        return True
    
    def _check_vector_store_knowledge(self, condition: str, allergies: List[str], min_documents: int) -> bool:
        """Check vector store directly for relevant documents"""
        try:
            # Get keywords for the condition
            keywords = self.condition_matcher.get_condition_keywords(condition)
            
            # Search with multiple keyword combinations
            search_queries = [
                condition,
                f"{condition} diet nutrition",
                f"{condition} dietary management",
            ]
            
            # Add synonym-based searches
            for keyword in keywords:
                if keyword != condition:
                    search_queries.append(f"{keyword} nutrition")
            
            all_results = set()
            
            for query in search_queries[:3]:  # Limit to prevent too many searches
                try:
                    results = self.vectorstore.similarity_search(query, k=5)
                    for result in results:
                        # Use a combination of content and metadata for uniqueness
                        result_id = hashlib.md5(
                            f"{result.page_content[:100]}{result.metadata.get('source', '')}{result.metadata.get('title', '')}".encode()
                        ).hexdigest()
                        all_results.add(result_id)
                except Exception as e:
                    logger.warning(f"Error in vector search for query '{query}': {e}")
                    continue
            
            unique_results_count = len(all_results)
            
            if unique_results_count >= min_documents:
                logger.info(f"Found {unique_results_count} relevant documents in vector store for: {condition}")
                return True
            else:
                logger.info(f"Only found {unique_results_count} relevant documents in vector store for: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking vector store knowledge: {e}")
            return False
    
    def add_documents(self, documents: List[Document], condition: str = None, 
                     allergies: List[str] = None) -> None:
        """Add documents to the vector database and update metadata"""
        if not documents:
            return
        
        valid_documents = []
        for doc in documents:
            if isinstance(doc, Document):
                valid_documents.append(doc)
            else:
                logger.warning(f"Skipping non-Document object: {type(doc)}")
        
        if not valid_documents:
            logger.warning("No valid Document objects to add")
            return
        
        split_docs = self.text_splitter.split_documents(valid_documents)
        
        filtered_docs = []
        for doc in split_docs:
            try:
                if not isinstance(doc, Document):
                    logger.warning(f"Skipping non-Document in split_docs: {type(doc)}")
                    continue

                safe_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        safe_metadata[key] = str(value)  
                    elif value is None:
                        safe_metadata[key] = "None"
                    else:
                        safe_metadata[key] = str(value) 
                
                filtered_doc = Document(
                    page_content=doc.page_content,
                    metadata=safe_metadata
                )
                filtered_docs.append(filtered_doc)
                
            except Exception as e:
                logger.warning(f"Error filtering document metadata: {e}")
                continue
        
        if not filtered_docs:
            logger.warning("No documents remaining after filtering")
            return
        
        try:
            self.vectorstore.add_documents(filtered_docs)
            logger.info(f"Added {len(filtered_docs)} document chunks to vector database")
            
            # Update metadata if condition and allergies are provided
            if condition is not None:
                if allergies is None:
                    allergies = []
                self._update_knowledge_metadata(condition, allergies, len(valid_documents))
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    def _update_knowledge_metadata(self, condition: str, allergies: List[str], 
                                  document_count: int):
        """Update metadata about stored knowledge"""
        condition_key = self._generate_condition_key(condition, allergies)
        
        if condition_key not in self.knowledge_metadata["conditions"]:
            self.knowledge_metadata["conditions"][condition_key] = {
                "condition": condition,
                "allergies": allergies,
                "document_count": 0,
                "sources": [],
                "first_added": datetime.now().isoformat(),
                "last_updated": None
            }
        
        metadata = self.knowledge_metadata["conditions"][condition_key]
        metadata["document_count"] += document_count
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Ensure sources is a list
        if not isinstance(metadata.get("sources"), list):
            metadata["sources"] = []
        
        self.knowledge_metadata["last_updated"] = datetime.now().isoformat()
        self._save_knowledge_metadata()
    
    def search_similar(self, query: str, k: int = 6, strategy: str = None) -> List[Document]:
        """Enhanced search using retrievers - much faster than manual similarity search"""
        try:
            # Use specified strategy or default
            retriever = self.active_retriever
            if strategy:
                temp_strategies = {
                    "similarity": self.similarity_retriever,
                    "mmr": self.mmr_retriever,
                    "threshold": self.threshold_retriever
                }
                retriever = temp_strategies.get(strategy, self.active_retriever)
            
            if k != 6:
                retriever.search_kwargs["k"] = k
            
            results = retriever.invoke(query)
            
            if k != 6:
                retriever.search_kwargs["k"] = 6
            
            logger.debug(f"Retrieved {len(results)} documents using retriever")
            return results
            
        except Exception as e:
            logger.error(f"Error in retriever search: {e}")
            return []
        
    def multi_query_search(self, base_query: str, k: int = 8) -> List[Document]:
        """Enhanced multi-query search using condition matching and retrievers"""
        try:
            # Generate multiple query variations
            keywords = self.condition_matcher.get_condition_keywords(base_query)
            
            # Create focused query variations
            queries = [
                base_query,
                f"{base_query} diet nutrition",
                f"{base_query} dietary management treatment"
            ]
            
            # Add keyword-based queries (limit to avoid too many calls)
            main_keywords = list(keywords)[:2]
            for keyword in main_keywords:
                if keyword.lower() != base_query.lower():
                    queries.append(f"{keyword} nutrition recommendations")
            
            # Use retriever for each query (still faster than manual similarity search)
            all_results = []
            seen_content_hashes = set()
            
            for query in queries[:4]:  # Limit to 4 queries max
                try:
                    results = self.active_retriever.get_relevant_documents(query)
                    
                    # Deduplicate based on content hash
                    for doc in results:
                        content_hash = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
                        if content_hash not in seen_content_hashes:
                            seen_content_hashes.add(content_hash)
                            all_results.append(doc)
                            
                except Exception as e:
                    logger.warning(f"Error in multi-query search for '{query}': {e}")
                    continue
            
            # Return top k results
            return all_results[:k]
            
        except Exception as e:
            logger.error(f"Error in multi-query search: {e}")
            return self.search_similar(base_query, k) 
        
    def search_with_filters(self, query: str, metadata_filters: Dict = None, k: int = 6) -> List[Document]:
        """Search with metadata filters using retriever"""
        try:
            if metadata_filters:
                # Create a custom retriever with filters
                filtered_retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": k,
                        "filter": metadata_filters
                    }
                )
                return filtered_retriever.get_relevant_documents(query)
            else:
                return self.search_similar(query, k)
                
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return self.search_similar(query, k)
        
    def get_retriever(self, search_type: str = "mmr", **kwargs) -> object:
        """Get a configured retriever for use in chains"""
        default_kwargs = {
            "similarity": {"k": 6},
            "mmr": {"k": 6, "fetch_k": 12, "lambda_mult": 0.7},
            "threshold": {"score_threshold": 0.5, "k": 8}
        }
        
        search_kwargs = default_kwargs.get(search_type, {"k": 6})
        search_kwargs.update(kwargs)
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
    def benchmark_retrievers(self, test_queries: List[str]) -> Dict:
        """Benchmark different retriever strategies"""
        results = {}
        strategies = ["similarity", "mmr", "threshold"]
        
        for strategy in strategies:
            start_time = time.time()
            
            for query in test_queries:
                self.search_similar(query, strategy=strategy)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / len(test_queries)
            
            results[strategy] = {
                "avg_time_per_query": avg_time,
                "total_time": end_time - start_time
            }
        
        return results
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about stored knowledge"""
        stats = {
            "total_conditions": len(self.knowledge_metadata["conditions"]),
            "conditions": []
        }
        
        for condition_key, metadata in self.knowledge_metadata["conditions"].items():
            condition_stats = {
                "condition": metadata["condition"],
                "allergies": metadata.get("allergies", []),
                "document_count": metadata.get("document_count", 0),
                "last_updated": metadata.get("last_updated"),
                "age_days": None
            }
            
            if metadata.get("last_updated"):
                try:
                    last_update = datetime.fromisoformat(metadata["last_updated"])
                    condition_stats["age_days"] = (datetime.now() - last_update).days
                except:
                    pass
            
            stats["conditions"].append(condition_stats)
        
        return stats
    
    def clear_old_knowledge(self, max_age_days: int = 30):
        """Clear knowledge older than specified days"""
        to_remove = []
        
        for condition_key, metadata in self.knowledge_metadata["conditions"].items():
            last_updated = metadata.get("last_updated")
            if last_updated:
                try:
                    last_update_date = datetime.fromisoformat(last_updated)
                    age_days = (datetime.now() - last_update_date).days
                    if age_days > max_age_days:
                        to_remove.append(condition_key)
                        logger.info(f"Marking old knowledge for removal: {metadata['condition']} ({age_days} days old)")
                except Exception as e:
                    logger.warning(f"Error parsing date for cleanup: {e}")
        
        for condition_key in to_remove:
            del self.knowledge_metadata["conditions"][condition_key]
        
        if to_remove:
            self._save_knowledge_metadata()
            logger.info(f"Removed metadata for {len(to_remove)} old conditions")
                        
class DietaryRecommendationSystem:
    """Enhanced system with better knowledge persistence"""
    
    def __init__(self):
        self.data_source_manager = DataSourceManager()
        self.vector_db_manager = VectorDatabaseManager()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.2,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self._initialization_done = False
        self._qa_chain = None
        self._recommendation_chain = None
        
    
    def _setup_chains(self):
        """Setup retrieval chains for faster processing"""
        
        # Create a specialized prompt for dietary recommendations
        recommendation_prompt = PromptTemplate(
            template="""
            You are a nutrition expert and registered dietitian providing evidence-based dietary recommendations.
            
            Patient Information:
            - Health Condition: {condition}
            - Food Allergies/Intolerances: {allergies}
            
            Based on the following scientific literature and nutritional data:
            {context}
            
            Please provide comprehensive, personalized dietary recommendations that include:
            
            1. **Foods to Include** (with specific examples and portions):
               - List specific foods that are beneficial for this condition
               - Include serving sizes and frequency recommendations
               - Explain WHY each food is beneficial (nutritional mechanisms)
            
            2. **Foods to Avoid** (considering both condition and allergies):
               - Foods that may worsen the condition
               - All allergenic foods mentioned by the patient
               - Explain the reasoning behind each restriction
            
            3. **Sample Daily Meal Plan**:
               - Breakfast, lunch, dinner, and snack suggestions
               - Ensure all meals are allergy-safe
               - Include approximate portions and nutritional highlights
            
            4. **Key Nutritional Targets**:
               - Specific nutrients to focus on (with daily targets if applicable)
               - Explain how these nutrients help manage the condition
            
            5. **Practical Implementation Tips**:
               - Meal prep suggestions
               - Grocery shopping tips
               - Dining out strategies
            
            6. **Important Considerations**:
               - Potential interactions with medications (general advice)
               - When to consult healthcare providers
               - Monitoring recommendations
            
            Ensure all recommendations are:
            - Completely safe for someone with the stated allergies
            - Specifically tailored to managing the health condition
            - Based on the scientific evidence provided
            - Practical and actionable for daily implementation
            - Include specific food brands or alternatives when helpful
            
            Format your response in clear sections with bullet points for easy reading.
            """,
            input_variables=["condition", "allergies", "context"]
        )
        
        # # Setup retrieval QA chain with MMR retriever for best results
        retriever = self.vector_db_manager.get_retriever(
            search_type="mmr",
            k=8,
            fetch_k=16,
            lambda_mult=0.7
        )
        
        # self._qa_chain = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=True,
        #     chain_type_kwargs={
        #         "prompt": recommendation_prompt
        #     }
        # )
        
        from langchain.chains import LLMChain
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser
        
        self._recommendation_chain = (
            {
                "context": retriever,
                "condition": RunnablePassthrough(),
                "allergies": RunnablePassthrough()
            }
            | recommendation_prompt
            | self.llm
            | StrOutputParser()
        )
    
    
    
    async def initialize(self):
        """Initialize the system with base knowledge - ONLY when needed"""
        if not self._initialization_done:
            # Setup chains first
            self._setup_chains()
            
            # Only initialize base knowledge if vector store is completely empty
            stats = self.vector_db_manager.get_knowledge_stats()
            if stats['total_conditions'] == 0:
                logger.info("Vector store is empty, initializing with base knowledge...")
                await self._initialize_base_knowledge()
            else:
                logger.info(f"Vector store already contains {stats['total_conditions']} conditions")
            
            self._initialization_done = True
            
    async def get_dietary_recommendations_fast(self, health_condition: HealthCondition) -> str:
        """Fast dietary recommendations using retrieval chains"""
        
        await self.initialize()
        
        try:
            # Check if we have sufficient knowledge
            has_knowledge = self.vector_db_manager.has_sufficient_knowledge(
                health_condition.condition, 
                health_condition.allergies,
                min_documents=5,
                max_age_days=7
            )
            
            if not has_knowledge:
                logger.info(f"Fetching new knowledge for {health_condition.condition}")
                await self._fetch_and_store_knowledge(
                    health_condition.condition, 
                    health_condition.allergies
                )
                # Recreate chains with updated vector store
                self._setup_chains()
            
            # Use retrieval chain for faster processing
            query = f"""
            Dietary recommendations for {health_condition.condition}
            Patient allergies: {', '.join(health_condition.allergies)}
            Nutrition guidelines and food recommendations
            """
            
            # Method 1: Use RetrievalQA chain (simpler but less control)
            result = self._qa_chain.invoke({
                "query": query,
                "condition": health_condition.condition,
                "allergies": ", ".join(health_condition.allergies) if health_condition.allergies else "None"
            })
            
            return result["result"]
            
        except Exception as e:
            logger.error(f"Error in fast recommendations: {e}")
            # Fallback to manual method
            return await self.get_dietary_recommendations(health_condition)
        
    async def get_dietary_recommendations_with_sources(self, health_condition: HealthCondition) -> Dict:
        """Get recommendations with source documents for transparency"""
        
        await self.initialize()
        
        try:
            # Ensure we have knowledge
            has_knowledge = self.vector_db_manager.has_sufficient_knowledge(
                health_condition.condition, 
                health_condition.allergies,
                min_documents=5,
                max_age_days=7
            )
            
            if not has_knowledge:
                await self._fetch_and_store_knowledge(
                    health_condition.condition, 
                    health_condition.allergies
                )
                self._setup_chains()
            
            query = f"""
            Dietary recommendations for {health_condition.condition}
            Patient allergies: {', '.join(health_condition.allergies)}
            Nutrition guidelines and food recommendations
            """
            
            # Use QA chain that returns sources
            result = self._qa_chain.invoke({
                "query": query,
                "condition": health_condition.condition,
                "allergies": ", ".join(health_condition.allergies) if health_condition.allergies else "None"
            })
            
            # Format source information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", "No title"),
                    "url": doc.metadata.get("url", ""),
                    "relevance_preview": doc.page_content[:200] + "..."
                }
                sources.append(source_info)
            
            return {
                "recommendations": result["result"],
                "sources": sources,
                "condition": health_condition.condition,
                "allergies": health_condition.allergies,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations with sources: {e}")
            return {
                "recommendations": f"Error generating recommendations: {str(e)}",
                "sources": [],
                "condition": health_condition.condition,
                "allergies": health_condition.allergies,
                "timestamp": datetime.now().isoformat()
            }
            
    async def get_quick_food_suggestions(self, condition: str, allergies: List[str], meal_type: str = "all") -> Dict:
        """Quick food suggestions using targeted retrieval"""
        
        await self.initialize()
        
        try:
            # Use targeted search with filters
            query = f"{condition} {meal_type} food recommendations nutrition"
            
            # Get retriever with specific settings for quick results
            retriever = self.vector_db_manager.get_retriever(
                search_type="similarity",  # Fastest option
                k=4  # Fewer results for speed
            )
            
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Quick processing with simpler prompt
            quick_prompt = f"""
            Based on this nutritional information: {' '.join([doc.page_content[:300] for doc in relevant_docs])}
            
            Provide quick food suggestions for someone with {condition}.
            Avoid: {', '.join(allergies) if allergies else 'No restrictions'}
            Meal type: {meal_type}
            
            List 5-8 specific foods with brief explanations (1-2 sentences each).
            """
            
            response = self.llm.invoke(quick_prompt)
            
            return {
                "suggestions": response.content,
                "condition": condition,
                "allergies": allergies,
                "meal_type": meal_type,
                "processing_time": "fast"
            }
            
        except Exception as e:
            logger.error(f"Error getting quick suggestions: {e}")
            return {"suggestions": f"Error: {str(e)}", "processing_time": "error"}
    
    def benchmark_retrieval_performance(self, test_conditions: List[str]) -> Dict:
        """Benchmark different retrieval approaches"""
        
        benchmark_results = {}
        
        # Test different retriever strategies
        strategies = ["similarity", "mmr", "threshold"]
        
        for strategy in strategies:
            start_time = time.time()
            
            self.vector_db_manager.set_retriever_strategy(strategy)
            
            for condition in test_conditions:
                query = f"{condition} dietary recommendations nutrition"
                results = self.vector_db_manager.search_similar(query, k=6)
            
            end_time = time.time()
            
            benchmark_results[strategy] = {
                "total_time": end_time - start_time,
                "avg_time_per_query": (end_time - start_time) / len(test_conditions),
                "queries_tested": len(test_conditions)
            }
        
        # Test retrieval chain vs manual approach
        start_time = time.time()
        
        # Manual approach timing would go here
        
        return benchmark_results
    
    async def _initialize_base_knowledge(self):
        """Pre-populate the database with common conditions and dietary guidelines"""
        base_conditions = [
            ("high blood pressure hypertension", []),
            ("diabetes type 2", [])
           
        ]
        
        logger.info("Checking base knowledge...")
        
        stats = self.vector_db_manager.get_knowledge_stats()
        logger.info(f"Current knowledge base contains {stats['total_conditions']} conditions")
        
        for condition, allergies in base_conditions:
            try:
                if not self.vector_db_manager.has_sufficient_knowledge(condition, allergies):
                    logger.info(f"Fetching base knowledge for: {condition}")
                    await self._fetch_and_store_knowledge(condition, allergies)
                else:
                    logger.info(f"Sufficient knowledge already exists for: {condition}")
            except Exception as e:
                logger.error(f"Error initializing knowledge for {condition}: {e}")
                continue
    
    async def _fetch_and_store_knowledge(self, condition: str, allergies: List[str]):
        """Fetch knowledge from multiple sources and store in vector database"""
        all_documents = []
        
        try:
            pubmed_query = f"{condition} diet nutrition dietary management {' '.join(allergies)}"
            pubmed_docs = await self.data_source_manager.search_pubmed_api(pubmed_query, max_results=12)
            all_documents.extend(pubmed_docs)
            
            recommended_foods = await self.data_source_manager.get_recommended_foods_for_condition(
                condition, allergies
            )
            usda_docs = await self.data_source_manager.search_usda_food_data(
                recommended_foods, condition
            )
            all_documents.extend(usda_docs)
            
            harvard_docs = await self.data_source_manager.scrape_harvard_nutrition(condition)
            all_documents.extend(harvard_docs)
            
            if all_documents:
                # Pass condition and allergies to track metadata
                self.vector_db_manager.add_documents(all_documents, condition, allergies)
                logger.info(f"Stored {len(all_documents)} documents for {condition}")
                logger.info(f"  - PubMed: {len(pubmed_docs)} articles")
                logger.info(f"  - USDA: {len(usda_docs)} food items")
                logger.info(f"  - Harvard: {len(harvard_docs)} articles")
            else:
                logger.warning(f"No documents found for {condition}")
                
        except Exception as e:
            logger.error(f"Error fetching and storing knowledge for {condition}: {e}")
    
    async def get_dietary_recommendations(self, health_condition: HealthCondition) -> str:
        """Get dietary recommendations for a given health condition and allergies"""
        
        await self.initialize()
        
        try:
            # Check if we have sufficient knowledge
            has_knowledge = self.vector_db_manager.has_sufficient_knowledge(
                health_condition.condition, 
                health_condition.allergies,
                min_documents=5,  # Require at least 5 documents
                max_age_days=7    # Refresh if older than 7 days
            )
            
            if not has_knowledge:
                logger.info(f"Fetching new knowledge for {health_condition.condition}")
                await self._fetch_and_store_knowledge(
                    health_condition.condition, 
                    health_condition.allergies
                )
            else:
                logger.info(f"Sufficient knowledge found for user request: {health_condition.condition}")
            
            
            query = f"""
            Dietary recommendations for {health_condition.condition}
            Patient allergies: {', '.join(health_condition.allergies)}
            Nutrition guidelines and food recommendations
            """
            
            relevant_docs = self.vector_db_manager.search_similar(query, k=8)
            
            # Your existing prompt template code here...
            prompt_template = """
            You are a nutrition expert and registered dietitian providing evidence-based dietary recommendations.
            
            Patient Information:
            - Health Condition: {condition}
            - Food Allergies/Intolerances: {allergies}
            
            Based on the following scientific literature and nutritional data:
            {context}
            
            Please provide comprehensive, personalized dietary recommendations that include:
            
            1. **Foods to Include** (with specific examples and portions):
               - List specific foods that are beneficial for this condition
               - Include serving sizes and frequency recommendations
               - Explain WHY each food is beneficial (nutritional mechanisms)
            
            2. **Foods to Avoid** (considering both condition and allergies):
               - Foods that may worsen the condition
               - All allergenic foods mentioned by the patient
               - Explain the reasoning behind each restriction
            
            3. **Sample Daily Meal Plan**:
               - Breakfast, lunch, dinner, and snack suggestions
               - Ensure all meals are allergy-safe
               - Include approximate portions and nutritional highlights
            
            4. **Key Nutritional Targets**:
               - Specific nutrients to focus on (with daily targets if applicable)
               - Explain how these nutrients help manage the condition
            
            5. **Practical Implementation Tips**:
               - Meal prep suggestions
               - Grocery shopping tips
               - Dining out strategies
            
            6. **Important Considerations**:
               - Potential interactions with medications (general advice)
               - When to consult healthcare providers
               - Monitoring recommendations
            
            Ensure all recommendations are:
            - Completely safe for someone with the stated allergies
            - Specifically tailored to managing the health condition
            - Based on the scientific evidence provided
            - Practical and actionable for daily implementation
            - Include specific food brands or alternatives when helpful
            
            Format your response in clear sections with bullet points for easy reading.
            """
            
            from langchain.prompts import PromptTemplate
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["condition", "allergies", "context"]
            )
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            if not context:
                context = "Limited scientific data available. Providing general dietary guidelines."
            
            formatted_prompt = prompt.format(
                condition=health_condition.condition,
                allergies=", ".join(health_condition.allergies) if health_condition.allergies else "None",
                context=context
            )
            
            response = self.llm.invoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Error getting dietary recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return self.vector_db_manager.get_knowledge_stats()
    
    def clear_old_knowledge(self, max_age_days: int = 300):
        """Clear knowledge older than specified days"""
        self.vector_db_manager.clear_old_knowledge(max_age_days)
        
class DietaryRecommendationAPI:
    """Enhanced API wrapper with retriever-based system"""
    
    def __init__(self):
        self.system = DietaryRecommendationSystem()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the API system"""
        if not self._initialized:
            await self.system.initialize()
            self._initialized = True
    
    async def get_recommendations(self, condition: str, allergies: List[str], 
                                fast_mode: bool = True) -> Dict:
        """Get dietary recommendations via API call with fast/comprehensive options"""
        try:
            await self.initialize()
            
            health_condition = HealthCondition(
                condition=condition,
                allergies=allergies
            )
            
            
            if fast_mode:
                # Use fast retriever-based method
                recommendations = await self.system.get_dietary_recommendations_fast(health_condition)
                method_used = "fast_retriever"
            else:
                # Use comprehensive method (backward compatible)
                recommendations = await self.system.get_dietary_recommendations(health_condition)
                method_used = "comprehensive"
            
            
            return {
                "status": "success",
                "condition": condition,
                "allergies": allergies,
                "recommendations": recommendations,
                "method": method_used,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
   
                     
# async def main():
#     """Enhanced main function with multiple usage examples"""
    
#     print("🍎 Dietary Recommendation System - Enhanced with Retrievers")
#     print("=" * 60)
    
#     # Initialize the API
#     api = DietaryRecommendationAPI()
    
#     # Example 1: Fast recommendations (default)
#     print("\n📊 Example 1: Fast Recommendations")
#     print("-" * 40)
    
#     result1 = await api.get_recommendations(
#         condition="high blood pressure",
#         allergies=["dairy", "nuts"],
#         fast_mode=True
#     )
    
#     print(f"Status: {result1['status']}")
#     print(f"Processing Time: {result1.get('processing_time_seconds', 'N/A')} seconds")
#     print(f"Method Used: {result1.get('method', 'N/A')}")
#     print("\nRecommendations Preview:")
#     print(result1["recommendations"][:500] + "...\n")
    
#     # Example 2: Comprehensive recommendations
#     print("\n📚 Example 2: Comprehensive Recommendations")
#     print("-" * 40)
    
#     result2 = await api.get_recommendations(
#         condition="diabetes type 2",
#         allergies=["gluten"],
#         fast_mode=False
#     )
    
#     print(f"Status: {result2['status']}")
#     print(f"Processing Time: {result2.get('processing_time_seconds', 'N/A')} seconds")
#     print(f"Method Used: {result2.get('method', 'N/A')}")
#     print("\nRecommendations Preview:")
#     print(result2["recommendations"][:500] + "...\n")
    
#     # Example 3: Recommendations with sources
#     print("\n🔗 Example 3: Recommendations with Sources")
#     print("-" * 40)
    
#     result3 = await api.get_recommendations_with_sources(
#         condition="heart disease",
#         allergies=["shellfish"]
#     )
    
#     print(f"Status: {result3['status']}")
#     print(f"Processing Time: {result3.get('processing_time_seconds', 'N/A')} seconds")
#     print(f"Number of Sources: {len(result3.get('sources', []))}")
    
#     if result3.get('sources'):
#         print("\nTop Sources:")
#         for i, source in enumerate(result3['sources'][:3], 1):
#             print(f"  {i}. {source.get('source', 'Unknown')} - {source.get('title', 'No title')}")
    
#     print("\nRecommendations Preview:")
#     print(result3["recommendations"][:400] + "...\n")
    
#     # Example 4: Quick food suggestions
#     print("\n⚡ Example 4: Quick Food Suggestions")
#     print("-" * 40)
    
#     result4 = await api.get_quick_suggestions(
#         condition="high cholesterol",
#         allergies=["eggs"],
#         meal_type="breakfast"
#     )
    
#     print(f"Status: {result4['status']}")
#     print(f"Processing Time: {result4.get('processing_time_seconds', 'N/A')} seconds")
#     print(f"Meal Type: {result4.get('meal_type', 'N/A')}")
#     print("\nSuggestions:")
#     print(result4.get("suggestions", "No suggestions available"))
#     print()
    
#     # Example 5: System statistics
#     print("\n📈 Example 5: System Statistics")
#     print("-" * 40)
    
#     stats = api.get_system_stats()
    
#     if stats['status'] == 'success':
#         knowledge_stats = stats['knowledge_stats']
#         print(f"Total Conditions in Knowledge Base: {knowledge_stats['total_conditions']}")
#         print(f"System Initialized: {stats['system_initialized']}")
        
#         if knowledge_stats['conditions']:
#             print("\nKnowledge Base Contents:")
#             for condition in knowledge_stats['conditions'][:5]:  # Show first 5
#                 age = condition.get('age_days', 'Unknown')
#                 print(f"  - {condition['condition']}: {condition['document_count']} docs, {age} days old")
    
#     # Example 6: Performance benchmark
#     print("\n🏃 Example 6: Performance Benchmark")
#     print("-" * 40)
    
#     benchmark_result = await api.benchmark_performance([
#         "obesity",
#         "osteoporosis"
#     ])
    
#     if benchmark_result['status'] == 'success':
#         results = benchmark_result['benchmark_results']
#         print("Retriever Performance Comparison:")
        
#         for strategy, metrics in results.items():
#             avg_time = metrics.get('avg_time_per_query', 0)
#             total_time = metrics.get('total_time', 0)
#             print(f"  - {strategy.capitalize()}: {avg_time:.3f}s avg, {total_time:.3f}s total")
    
#     print("\n✅ All examples completed!")


# if __name__ == "__main__":
#     asyncio.run(main())