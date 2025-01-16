from langchain_ollama import OllamaLLM
import json
import re
import random
from collections import defaultdict
from itertools import combinations
import pandas as pd
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from nltk.tokenize import word_tokenize
import json
from gensim.corpora.dictionary import Dictionary
import re
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import random
from langchain_ollama import OllamaLLM

class Preprocessor:
    def __init__(self, custom_stopwords=None, min_word_length=3):
        """
        Initialize the Preprocessor with optional custom stopwords and word length filter.
        """
        download('stopwords')
        download('punkt')
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        self.min_word_length = min_word_length

    def preprocess_text(self, text):
        """
        Preprocess a single text string.
        :param text: String to preprocess.
        :return: Preprocessed string or 'placeholder' if the text is invalid.
        """
        try:
            # Remove extra spaces and newlines
            text = re.sub(r'\s+', ' ', text.strip())

            # Expand contractions (e.g., "I've" -> "I have")
            contractions = {
                "it's": "it is", "I've": "I have", "didn't": "did not", "doesn't": "does not",
                "can't": "cannot", "won't": "will not", "isn't": "is not", "aren't": "are not"
            }
            for contraction, expanded in contractions.items():
                text = text.replace(contraction, expanded)

            # Remove punctuation and numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\d+', '', text)  # Remove numbers

            # Tokenize
            tokens = word_tokenize(text.lower())

            # Filter tokens based on stopwords and length
            filtered_tokens = [
                token for token in tokens
                if token not in self.stop_words and len(token) >= self.min_word_length
            ]

            return " ".join(filtered_tokens) if filtered_tokens else "placeholder"

        except Exception as e:
            print(f"Error processing text: {e}")
            return "placeholder"

    def preprocess_documents(self, documents):
        """
        Preprocess a list of documents.
        :param documents: List of strings (documents) to preprocess.
        :return: List of preprocessed documents.
        """
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list of strings.")

        processed_docs = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                print(f"Document {i} is not a valid string. Skipping...")
                processed_docs.append("placeholder")
                continue

            try:
                processed_docs.append(self.preprocess_text(doc))
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                processed_docs.append("placeholder")

        return processed_docs


@dataclass
class LLMResponse:
    """Data class to store LLM responses with enhanced fields."""
    content: str
    score: float = None
    topic: str = None
    confidence: float = None
    error: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Document:
    """Data class to store document information."""
    content: str
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = None

class RAGSystem:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initializes the RAG system:
          Loads the specified embedding model.
          Creates an empty document store and sets the FAISS index to None.
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_store = []
        self.index = None

    def add_documents(self, documents: List[str]):
        """
        Adds documents to the system and prepares them for retrieval.
        Converts each document into a vector embedding using the SentenceTransformer model.
        Each document is wrapped in a Document object containing:
            content: The text of the document.
            embedding: The corresponding embedding (as a NumPy array).
        These objects are appended to document_store.
        """
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
        for doc, emb in zip(documents, embeddings):
            self.document_store.append(Document(
                content=doc,
                embedding=emb.numpy()
            ))
        self._update_index()

    def _update_index(self):
        """
        Updates the FAISS index with embeddings from all stored documents.
        """
        embeddings = np.vstack([doc.embedding for doc in self.document_store])
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def retrieve_relevant_docs(self, query: str, k: int = 3):
        """
        Retrieves the top-k documents most relevant to a given query.
        """
        query_embedding = self.embedding_model.encode([query])[0]
        D, I = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        return [self.document_store[i] for i in I[0]]

def extract_fallback_score(response):
    """
    Improved fallback method to extract scores with better accuracy
    """
    # First try to find a score explicitly stated
    print(response)
    score_patterns = [
        r'SCORE:\s*(\d+)',
        r'score of (\d+)',
        r'rated (\d+)',
        r'(\d+) points'
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response.lower())
        if match:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score
            elif 0 <= score <= 100:
                return round(score / 10)  # Convert 0-100 to 0-10 scale

    # Keyword-based scoring with more nuanced approach
    keywords = {
        10: ['perfect', 'excellent', 'outstanding'],
        8: ['very good', 'strong', 'highly relevant'],
        6: ['good', 'relevant', 'adequate'],
        4: ['fair', 'partially relevant', 'moderate'],
        2: ['poor', 'weak', 'irrelevant']
    }

    response_lower = response.lower()
    for score, terms in keywords.items():
        if any(term in response_lower for term in terms):
            return score

    return 5  # Default middle score


class EnhancedLLM:
    def __init__(self, documents, model="llama3.2"):
        """
        Initialize the EnhancedLLM class.

        :param documents: List of documents to process.
        :param model: Name of the model to use with OllamaLLM.
        """
        self.documents = documents
        self.model = model
        try:
            # Initialize the LLM
            self.llm = OllamaLLM(model=model)

            # Test connection by running a simple query
            test_prompt = "System check: Are you ready?"
            response = self.llm.generate([test_prompt])
            if response.generations and response.generations[0][0].text:
                print(f"Connected to model '{model}' successfully!")
            else:
                raise ValueError("The model did not return a valid response.")
        except Exception as e:
            print(f"Failed to connect to model '{model}': {e}")
            self.llm = None  # Reset llm to None to indicate failure

    class EnhancedLLM:
        def __init__(self, documents: List[str], model: str):
            """
            Initialize the EnhancedLLM class.

            :param documents: List of documents to process.
            :param model: Name of the model to use with OllamaLLM.
            """
            self.documents = documents
            self.model = model
            self.llm = OllamaLLM(model=model)
            print(self.llm)

    def summarize_document(self, text, max_tokens=200) -> str:
        """
        Enhanced summarization with OllamaLLM, focusing on domain-specific terminology and key concepts.
        """
        # Preprocess to limit length
        max_input_length = 4000
        if len(text) > max_input_length:
            text = text[:max_input_length] + "..."

        # Create the prompt
        prompt = f"""
        You are a specialized topic identification system. Your task is to:
        1. Extract and emphasize domain-specific terminology.
        2. Identify technical concepts and subject matter.
        3. Recognize topic patterns and themes.
        4. Remove general discussion and focus on subject identifiers.

        Return a structured summary in this format:
        MAIN_TOPIC: [primary subject area]
        KEY_TERMS: [comma-separated domain-specific terms]
        SUMMARY: [2-3 sentences focusing on technical content]

        Analyze this text for technical topic identification: {text}
        """

        try:
            # Initialize the model


            # Generate the response
            response = self.llm.generate([prompt])

            # Extract the generated text
            generated_text = response.generations[0][0].text

            # Ensure the response contains the expected structure
            if not all(section in generated_text for section in ["MAIN_TOPIC:", "KEY_TERMS:", "SUMMARY:"]):
                # Fallback to simple truncation
                return "The output is not structured properly. Here is the first 200 characters:\n" + text[:200] + "..."

            return generated_text.strip()

        except Exception as e:
            print(f"Error in enhanced summarization: {str(e)}")
            return "An error occurred. Here is the first 200 characters:\n" + text[:200] + "..."

    def improved_evaluate_groups_with_llm(self,group_documents, other_groups, model="llama3.2"):
        """
        Enhanced evaluation function using OllamaLLM for scoring document groups.
        """
        # Combine group documents and context
        all_docs_text = "\n".join(group_documents)

        # Create summaries for the group and other groups
        group_summary = "\n".join([f"Document {i + 1}: {doc[:500]}..." for i, doc in enumerate(group_documents)])
        other_groups_summary = "\n".join(
            [f"Group {i + 1}: {' '.join(group[:3])}..." for i, group in enumerate(other_groups)])

        # Prepare the prompt
        prompt = f"""You are a specialized content coherence evaluator.

TARGET GROUP:
{group_documents}

COMPARISON GROUPS:
{other_groups}

Evaluation Criteria:
1. INTERNAL COHERENCE (50%)
- How consistently do the documents align in topic and terminology?
- Do they share a common technical vocabulary?
- Is there thematic continuity?

2. EXTERNAL DISTINCTIVENESS (50%)
- How clearly separated is this group from others?
- Are there unique technical markers?
- Is there minimal topic overlap with other groups?

Provide your evaluation in this EXACT format:
COHERENCE_SCORE: [1-10]
DISTINCTIVENESS_SCORE: [1-10]
FINAL_SCORE: [average of above, rounded to nearest whole number]
STRONG_POINTS: [bullet list of group's strongest cohesion markers]
DISTINGUISHING_FEATURES: [key elements that separate this group]"""

        try:
            # Initialize the model


            # Generate the response
            response = self.llm.generate([prompt])

            # Extract the generated text
            generated_text = response.generations[0][0].text

            # Parse the response for SCORE and BRIEF_JUSTIFICATION
            score_match = re.search(r"SCORE:\s*(\d+)", generated_text)
            justification_match = re.search(r"BRIEF_JUSTIFICATION:\s*(.+)", generated_text)

            if score_match:
                score = int(score_match.group(1))
                justification = justification_match.group(1) if justification_match else "No justification provided."
                return {
                    "score": min(10, max(1, score)),  # Ensure score is between 1 and 10
                    "justification": justification.strip()
                }

            # Fallback if the structure isn't as expected
            return {
                "score": 5,  # Default score
                "justification": "Could not extract a valid response."
            }

        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {
                "score": 5,  # Default score
                "justification": f"Error during evaluation: {str(e)}"
            }

    def classify_document(self, text) -> dict:
        """
        Classify the topic of a document using OllamaLLM.

        :param text: The text to classify.
        :return: A dictionary with classification results.
        """
        # Create the prompt
        prompt = f"""
        You are an expert topic classifier focusing on technical and academic content.

        Document for classification:
        {text}

        Analyze this document following these steps:
        1. Identify primary technical domain
        2. Extract key technical terminology
        3. Recognize methodological approaches
        4. Note any cross-domain elements

        Provide classification in this EXACT format:
        PRIMARY_TOPIC: [single specific technical field]
        SUBTOPICS: [3-4 related technical areas]
        TECHNICAL_INDICATORS: [key technical terms that influenced classification]
        CROSS_DOMAIN_ELEMENTS: [any interdisciplinary aspects]
        CONFIDENCE: [0-1 score with brief justification]
        """

        try:
            response = self.llm.generate([prompt])

            # Access response text based on structure
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get("text", "").strip()
            else:
                raise ValueError("Response structure is invalid or empty.")

            # Parse response into dictionary
            result = {
                "PRIMARY_TOPIC": self._extract_field(generated_text, "PRIMARY_TOPIC"),
                "SUBTOPICS": self._extract_field(generated_text, "SUBTOPICS"),
                "TECHNICAL_INDICATORS": self._extract_field(generated_text, "TECHNICAL_INDICATORS"),
                "CROSS_DOMAIN_ELEMENTS": self._extract_field(generated_text, "CROSS_DOMAIN_ELEMENTS"),
                "CONFIDENCE": self._extract_field(generated_text, "CONFIDENCE"),
            }

            return result

        except Exception as e:
            print(f"Error in topic classification: {str(e)}")
            return {"error": str(e)}

    def _extract_field(self, text: str, field_name: str) -> str:
        """
        Extract a specific field from the generated text.
        :param text: The generated text.
        :param field_name: The field name to extract.
        :return: The extracted field value.
        """
        pattern = rf"{field_name}:\s*(.*)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""


def calculate_coherence_scores(groups, dictionary, measure="c_v"):
    scores = []
    for group in groups:
        try:
            # Create "topics" as a list of the most frequent terms in the group
            topics = [[word for word, freq in dictionary.doc2bow(doc)] for doc in group]

            # Create a CoherenceModel for the group
            coherence_model = CoherenceModel(
                topics=topics,
                texts=group,
                dictionary=dictionary,
                coherence=measure
            )

            # Calculate the coherence score
            score = coherence_model.get_coherence()
            scores.append(score)
        except Exception as e:
            print(f"Error calculating coherence for group: {e}")
            scores.append(0.0)

    return scores

# def perform_lda_analysis(documents, n_topics=5):
#     if not documents or not isinstance(documents, list):
#         return {"assigned_topics": [], "topics_keywords": {}}
#
#     try:
#         vectorizer = CountVectorizer(
#             stop_words='english',
#             max_df=0.95,
#             min_df=2,
#             token_pattern=r'(?u)\b\w+\b'
#         )
#
#         X = vectorizer.fit_transform(documents)
#
#         lda_model = LatentDirichletAllocation(
#             n_components=n_topics,
#             random_state=42,
#             max_iter=20,
#             learning_method='batch'
#         )
#
#         lda_model.fit(X)
#         feature_names = vectorizer.get_feature_names_out()
#
#         topics = {}
#         for topic_idx, topic in enumerate(lda_model.components_):
#             top_keywords_idx = topic.argsort()[-10:][::-1]
#             top_keywords = [feature_names[i] for i in top_keywords_idx]
#             topics[topic_idx] = top_keywords
#
#         topic_assignments = lda_model.transform(X)
#         assigned_topics = np.argmax(topic_assignments, axis=1)
#
#         return {
#             "assigned_topics": assigned_topics.tolist(),
#             "topics_keywords": topics
#         }
#
#     except Exception as e:
#         print(f"Error in LDA analysis: {str(e)}")
#         return {"assigned_topics": [], "topics_keywords": {}}



def create_groups_from_data(data, topic1, topic2, num_docs=None, mixed_ratio=0.5):
    """
    Create three groups from the dataset:
    1. Homogeneous group with only documents from `topic1`.
    2. Homogeneous group with only documents from `topic2`.
    3. Mixed group with a specified proportion of documents from both topics.

    Args:
        data (list): List of documents with `text` and `label`.
        topic1 (str): The label for the first homogeneous group.
        topic2 (str): The label for the second homogeneous group.
        num_docs (int or None): Number of documents to include in each group (default: all available documents).
        mixed_ratio (float): Proportion of documents in the mixed group (default: 50% from each).

    Returns:
        tuple: Three lists representing the three groups (group1, group2, mixed_group).
    """
    # Separate documents by label
    grouped_by_label = defaultdict(list)
    for doc in data:
        grouped_by_label[doc["label"]].append(doc["text"])

    # Validate topics exist in the data
    if topic1 not in grouped_by_label or topic2 not in grouped_by_label:
        raise ValueError(f"One or both topics '{topic1}' and '{topic2}' are not present in the dataset.")

    # Create homogeneous groups
    group1 = grouped_by_label[topic1]
    group2 = grouped_by_label[topic2]

    # If num_docs is specified, limit the size of the groups
    if num_docs:
        group1 = random.sample(group1, min(num_docs, len(group1)))
        group2 = random.sample(group2, min(num_docs, len(group2)))

    # Create a mixed group
    size_mixed = int(mixed_ratio * len(group1))  # Number of documents from each group in the mixed group
    mixed_group = random.sample(group1, size_mixed) + random.sample(group2, size_mixed)
    random.shuffle(mixed_group)  # Shuffle to mix documents randomly

    return group1, group2, mixed_group


# def get_balanced_dataset(newsgroups, category_groups, docs_per_category=3):
#
#     group_docs = []
#     category_counts = {}
#
#     for group_categories in category_groups:
#         group_data = []
#         group_total = 0
#
#         for category in group_categories:
#             category_indices = [i for i in range(len(newsgroups.target))
#                               if newsgroups.target_names[newsgroups.target[i]] == category]
#
#             # Get and preprocess documents
#             category_docs = [preprocess_text(newsgroups.data[i])
#                            for i in category_indices[:docs_per_category]]
#             group_data.extend(category_docs)
#
#             category_counts[category] = len(category_docs)
#             group_total += len(category_docs)
#
#         group_docs.append(group_data)
#
#         print(f"\nGroup with categories {group_categories}:")
#         print(f"Total documents: {group_total}")
#         for category in group_categories:
#             print(f"  - {category}: {category_counts[category]} documents")
#
#     return group_docs, category_counts


if __name__ == "__main__":
    try:
        # Load the 20 Newsgroups dataset
        newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

        # Convert the dataset into a list of dictionaries
        dataset = [{"text": text, "label": newsgroups_data.target_names[label]}
                   for text, label in zip(newsgroups_data.data, newsgroups_data.target)]

        # Create groups (homogeneous and mixed)
        group1, group2, mixed_group = create_groups_from_data(dataset,
                                                              topic1='rec.sport.hockey',
                                                              topic2='comp.sys.ibm.pc.hardware',
                                                              num_docs=10,
                                                              mixed_ratio=0.5)
        # tokenized_group1 = [[word for word in doc.split()] for doc in group1]
        # tokenized_group2 = [[word for word in doc.split()] for doc in group2]
        # tokenized_mixed_group = [[word for word in doc.split()] for doc in mixed_group]
        #
        # dictionary = Dictionary(tokenized_group1 + tokenized_group2 + tokenized_mixed_group)
        #
        # coherence_scores = calculate_coherence_scores([tokenized_group1, tokenized_group2, tokenized_mixed_group],
        #                                               dictionary)
        #
        # print("Coherence Scores:", coherence_scores)
        #

        # Initialize preprocessor
        # preprocessor = Preprocessor()
        #
        # # Preprocess each group
        # preprocessed_group1 = preprocessor.preprocess_documents(group1)
        # preprocessed_group2 = preprocessor.preprocess_documents(group2)
        # preprocessed_mixed_group = preprocessor.preprocess_documents(mixed_group)
        #
        # # Save processed groups into a dictionary
        # processed_data = {
        #     "group1": preprocessed_group1,
        #     "group2": preprocessed_group2,
        #     "mixed_group": preprocessed_mixed_group
        # }
        #
        # print("\nProcessed Groups:")
        # for group_name, documents in processed_data.items():
        #     print(f"\n{group_name} ({len(documents)} documents):")
        #     for doc in documents:
        #         print(f"- {doc}")

        # Summarize documents in each group
        docs_to_summarize = group1[:10]

        # Initialize the EnhancedLLM with documents and model
        enhanced_llm = EnhancedLLM(documents=docs_to_summarize, model="llama3.2")
        x = enhanced_llm.improved_evaluate_groups_with_llm(group1,group2)
        print(x)
        # # Summarize each document

        # for i, doc in enumerate(docs_to_summarize, 1):
        #     summary = enhanced_llm.summarize_document(doc, max_tokens=200)
        #     print(f"Document {i}: {doc}\nSummary: {summary}\n")

        # Output summarized groups
        # print("\nSummarized Groups:")
        # for group_name, summaries in summarized_data.items():
        #     print(f"\n{group_name} Summaries:")
        #     for idx, summary_data in enumerate(summaries):
        #         print(f"\nDocument {idx + 1}:")
        #         print(f"Original: {summary_data['original']}")
        #         print(f"Summary: {summary_data['summary']}")

    except Exception as e:
        print(f"Unexpected error: {e}")

# #
# if __name__ == "__main__":
#     try:
#         # Load the 20 Newsgroups dataset
#         newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
#
#         # Initialize preprocessor
#         preprocessor = Preprocessor()
#
#         # Group documents by target names
#         grouped_documents = {}
#         for text, target in zip(newsgroups_data.data, newsgroups_data.target):
#             target_name = newsgroups_data.target_names[target]
#             if target_name not in grouped_documents:
#                 grouped_documents[target_name] = []
#             grouped_documents[target_name].append(text)
#
#         # Apply preprocessing to each group
#         processed_groups = {
#             target: preprocessor.preprocess_documents(docs)
#             for target, docs in grouped_documents.items()
#         }
#
#         # Initialize RAG system
#         rag_system = RAGSystem()
#
#         # Flatten all processed documents and add them to RAG system
#         all_documents = [doc for docs in processed_groups.values() for doc in docs]
#         rag_system.add_documents(all_documents)
#
#         # Test retrieval
#         test_query = "satellite technology"
#         retrieved_docs = rag_system.retrieve_relevant_docs(test_query, k=3)
#
#         # Output results
#         print("\nProcessed Groups:")
#         print(json.dumps(processed_groups, indent=4))
#
#         print("\nRetrieved Documents:")
#         for doc in retrieved_docs:
#             print(f"- {doc.content}")
#
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#
#
#
#
#
# # Example usage
# if __name__ == "__main__":
#     # from nltk import download
#     # from nltk.corpus import stopwords
#     # from nltk.tokenize import word_tokenize
#     #
#     # # Ensure necessary NLTK resources are downloaded
#     # download('stopwords')
#     # download('punkt')
#
#     # Example input documents
#     documents = [
#         "The satellite was launched into orbit.",
#         "Space exploration requires advanced propulsion systems."
#     ]
#
#     # Preprocessing step
#     class Preprocessor:
#         def __init__(self):
#             self.stop_words = set(stopwords.words('english'))
#             self.custom_stops = {'would', 'could', 'should', 'said', 'like', 'also'}
#             self.stop_words.update(self.custom_stops)
#
#         def preprocess_text(self, text):
#             text = text.lower()
#             text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
#             text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#
#             tokens = word_tokenize(text)
#
#             # Filter tokens based on conditions
#             filtered_tokens = []
#             for token in tokens:
#                 if (token not in self.stop_words and
#                         len(token) > 2 and
#                         not token.isnumeric() and
#                         not all(c in '0123456789.-' for c in token)):
#
#                     if token.isupper() and len(token) <= 5:
#                         filtered_tokens.append(token)
#                     else:
#                         filtered_tokens.append(token.lower())
#
#             cleaned_text = " ".join(filtered_tokens)
#             return cleaned_text if cleaned_text.strip() else "placeholder"
#
#         def preprocess_documents(self, documents):
#             processed_docs = []
#             for doc in documents:
#                 try:
#                     if isinstance(doc, str):
#                         processed_docs.append(self.preprocess_text(doc))
#                     else:
#                         processed_docs.append("placeholder")  # Handle non-string entries
#                 except Exception as e:
#                     print(f"Error preprocessing document: {e}")
#                     processed_docs.append("placeholder")
#
#             return processed_docs
#
#     # Preprocess documents
#     preprocessor = Preprocessor()
#     processed_docs = preprocessor.preprocess_documents(documents)
#
#     # Initialize RAG system and add documents
#     rag_system = RAGSystem()
#     rag_system.add_documents(processed_docs)
#
#     # Query the system
#     query = "satellite technology"
#     relevant_docs = rag_system.retrieve_relevant_docs(query, k=2)
#
#     # Output results
#     print("\nQuery:", query)
#     print("\nRelevant Documents:")
#     for doc in relevant_docs:
#         print(f"- {doc.content}")
#
#
#
#
#
#
# def improved_summarize_with_ollama(text, model="llama3.2", max_tokens=200):
#     """
#     Enhanced summarization with OllamaLLM, focusing on domain-specific terminology and key concepts.
#     """
#     # Preprocess to limit length
#     max_input_length = 4000
#     if len(text) > max_input_length:
#         text = text[:max_input_length] + "..."
#
#     # Create the prompt
#     prompt = f"""
#     You are a specialized topic identification system. Your task is to:
#     1. Extract and emphasize domain-specific terminology.
#     2. Identify technical concepts and subject matter.
#     3. Recognize topic patterns and themes.
#     4. Remove general discussion and focus on subject identifiers.
#
#     Return a structured summary in this format:
#     MAIN_TOPIC: [primary subject area]
#     KEY_TERMS: [comma-separated domain-specific terms]
#     SUMMARY: [2-3 sentences focusing on technical content]
#
#     Analyze this text for technical topic identification: {text}
#     """
#
#     try:
#         # Initialize the model
#         llm = OllamaLLM(model=model)
#         print(1)
#         # Generate the response
#         response = llm.generate([prompt])
#
#         # Extract the generated text
#         generated_text = response.generations[0][0].text
#
#         # Ensure the response contains the expected structure
#         if not all(section in generated_text for section in ["MAIN_TOPIC:", "KEY_TERMS:", "SUMMARY:"]):
#             # Fallback to simple truncation
#             return "The output is not structured properly. Here is the first 200 characters:\n" + text[:200] + "..."
#
#         return generated_text.strip()
#
#     except Exception as e:
#         print(f"Error in enhanced summarization: {str(e)}")
#         return "An error occurred. Here is the first 200 characters:\n" + text[:200] + "..."
#
#
# input_file = "newsgroups_data_preprocessed.json"
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)
