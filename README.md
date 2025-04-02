# Evaluating Subjects Through LLM

This project focuses on developing a novel assessment measure for evaluating document divisions using Large Language Models (LLMs). By leveraging the capabilities of LLMs, the project aims to create a scoring mechanism that closely aligns with human evaluations of document grouping. The proposed approach compares the new measure with existing evaluation techniques in the field of Topic Detection.

## Project Workflow
The project follows a structured workflow divided into the following key stages:

### 1. Preprocessing
In this stage, raw text data is cleaned and prepared for further analysis. The preprocessing steps include:
- Tokenization
- Stop-word removal
- Lemmatization
- Data formatting and structuring

### 2. Summarization
Key information from documents is extracted and summarized to create a concise representation. This step ensures that the document's core content is retained while reducing unnecessary noise.

### 3. Retrieval-Augmented Generation (RAG)
The project employs Retrieval-Augmented Generation (RAG) to retrieve relevant information from the document corpus. The RAG pipeline consists of:
- Document retrieval
- Query-based summarization
- Enhanced content understanding using LLMs

### 4. LLM Evaluation
Finally, the summarized and retrieved content is evaluated using a Large Language Model. The evaluation focuses on:
- Internal coherence within document groups
- External distinctiveness between different groups
- Generating a scoring metric that mimics human judgment

## Datasets Used
The project utilizes the following datasets:
- **Newsgroups**: A dataset of approximately 20,000 documents categorized into 20 topics, consisting of long-form text.
- **CAVES**: A Twitter dataset focusing on COVID-19 vaccine hesitancy, containing 9,921 tweets labeled into 10 categories of vaccine concerns.

## Goals and Objectives
The primary goal of this project is to design and implement a novel measure for evaluating document divisions. The specific objectives include:

- **Measure Development**: Create a scoring mechanism based on LLMs that evaluates how well a document division reflects human-like categorization.
- **Comparative Analysis**: Benchmark the new measure against existing techniques in Topic Detection.
- **Scalability**: Ensure the approach works effectively across various datasets with different characteristics (e.g., Newsgroups vs. CAVES).

## Team
This project is conducted by:
- **Eyal Shubeli**
- **Nadav Toledo**
- **Ido Villa**

### Supervisors
- **Prof. Rami Pozis**
- **Dr. Aviad Elyashar**
- **Noa Tal**

## Tools and Technologies
The project utilizes the following tools and technologies:
- **Python**: For preprocessing, summarization, and evaluation.
- **LLMs**: To generate scores and evaluate document groupings.
- **RAG Pipelines**: For retrieval and generation of document summaries.
- **Data Visualization**: Tools like Matplotlib and Seaborn for presenting results.
- **Neptune.ml**: For tracking experiments and performance metrics.

## Installation and Usage
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-repo/evaluating-subjects-through-llm.git
cd evaluating-subjects-through-llm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'

# Install dependencies
pip install -r requirements.txt
```

## Running the Project
To preprocess and evaluate document divisions, run:

```bash
python main.py
```

## Contributions
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch-name`).
3. Commit your changes.
4. Push to your fork and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
Feel free to reach out with any questions or suggestions!

