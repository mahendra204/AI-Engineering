# 🚀 AI Engineer - Complete Portfolio Repository

A comprehensive collection of **production-ready AI/ML projects**, Databricks learning materials, and PySpark examples. This repository showcases full-stack engineering capabilities in data processing, machine learning, and advanced analytics.

---

## 📋 Table of Contents

- [Projects Overview](#-projects-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Project Details](#-project-details)
- [Technologies & Stack](#-technologies--stack)
- [Setup & Installation](#-setup--installation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Projects Overview

### 1. **RAG Application** — Retrieval-Augmented Generation
**Status:** ✅ Production-Ready  
**Path:** `./rag_app/`

A modern RAG system that combines document retrieval with LLM reasoning:
- 📄 Multi-format document ingestion (PDF, DOCX, TXT)
- 🔍 Hybrid retrieval (semantic + BM25 search)
- 💬 FastAPI REST backend + Streamlit UI
- 🧠 OpenAI LLM integration (GPT-4o-mini)
- 💾 ChromaDB vector store

**Key Features:**
- Real-time document chunking & embedding
- Context-aware Q&A system
- Production-grade error handling & logging
- Comprehensive test suite

**Quick Start:**
```bash
cd rag_app
pip install -r requirements.txt
python main.py  # FastAPI server
streamlit run ui.py  # Chat UI
```

[📖 Full Documentation](./rag_app/README.md)

---

### 2. **Sentiment Analysis API**
**Status:** ✅ Complete  
**Path:** `./Sentiment Analyzer/`

A FastAPI-based sentiment analysis service with web UI:
- 🎭 Real-time sentiment classification
- 📊 Multi-model support (BERT, transformer-based)
- 🎨 Beautiful web interface
- ⚡ Fast inference with GPU acceleration

**Use Cases:**
- Social media monitoring
- Customer feedback analysis
- Review sentiment classification

**Quick Start:**
```bash
cd "Sentiment Analyzer"
pip install -r requirements.txt
python src/api.py
python src/ui.py  # Web UI
```

[📖 Full Documentation](./Sentiment%20Analyzer/README.md)

---

### 3. **Databricks Reskilling Course**
**Status:** ✅ Complete Comprehensive Course  
**Path:** `./databricks_reskill/`

An enterprise-grade learning path covering:
- 📘 7 progressive skill modules (Beginner → Advanced)
- 💾 30+ hands-on PySpark scripts
- 🏗️ Data engineering, Delta Lake, SQL optimization
- 📊 ML, streaming, and analytics
- 🔒 Security & production best practices

**Module Breakdown:**
| Module | Topics | Duration |
|--------|--------|----------|
| 01 Intro | Architecture, workspace setup | 1-2 hrs |
| 02 Fundamentals | Data concepts, storage | 2-3 hrs |
| 03 Spark Basics | DataFrames, RDDs, APIs | 3-4 hrs |
| 04 Spark SQL | Queries, window functions | 3-4 hrs |
| 05 Performance | Optimization, tuning | 3-4 hrs |
| 06 Advanced | Streaming, ML, governance | 4-5 hrs |
| 07 Production | Security, CI/CD, monitoring | 3-4 hrs |

**Quick Start:**
```bash
cd databricks_reskill
# Start with the learning path
cat 00_START_HERE.md
cat COURSE_JOURNEY.md

# Run example scripts
python scripts/01_basic_setup.py
python scripts/02_data_transformation.py
```

[📖 Course Guide](./databricks_reskill/00_START_HERE.md) | [🗺️ Journey Map](./databricks_reskill/COURSE_JOURNEY.md)

---

### 4. **PySpark Examples & Setup**
**Status:** ✅ Verified & Working  
**Path:** `./spark/`

A complete PySpark environment with working examples:
- ✅ PySpark 4.1.1 installed & verified
- 📝 Multiple working example scripts
- 📊 DataFrame operations, SQL, transformations
- 🔧 Setup verification & diagnostics

**Verified Components:**
- Python 3.13.5 + Java 25 + Windows 11 compatible
- DataFrame creation & transformations
- Spark SQL operations
- Schema management

**Quick Start:**
```bash
cd spark
# Run verified working example
python pyspark_working_example.py

# Or check setup
python PYSPARK_SETUP_COMPLETE.py
```

[📖 Full Setup Guide](./spark/PYSPARK_README.md)

---

### 5. **Synthetic Data Generation Scripts**
**Status:** ✅ Complete  
**Path:** `./Synthetic_Data_Gen_Scripts-Claude/`

Enterprise-grade synthetic data generators:
- 👥 Customer, employee, transaction data
- 📦 Product catalogs, order histories
- 🎫 Support tickets with realistic patterns
- 📊 Ready for testing, ML training, demos

**Generated Datasets:**
- Customers (demographic data)
- Employees (HR data)
- Orders & transactions
- Products & inventory
- Support tickets

**Quick Start:**
```bash
cd Synthetic_Data_Gen_Scripts-Claude
# Generate all datasets
python generate_all.py

# Or specific datasets
python generate_customers.py
python generate_orders.py
```

[📖 Full Documentation](./Synthetic_Data_Gen_Scripts-Claude/readme.md)

---

## 📁 Repository Structure

```
AIEngineer/
├── README.md                              # This file
│
├── rag_app/                               # 🔍 RAG Application
│   ├── app/                               # Core RAG modules
│   ├── scripts/                           # Ingestion & query CLI
│   ├── tests/                             # Test suite
│   ├── static/                            # UI assets
│   ├── main.py                            # FastAPI server
│   ├── ui.py                              # Streamlit interface
│   └── requirements.txt
│
├── Sentiment Analyzer/                    # 🎭 Sentiment Analysis API
│   ├── src/                               # Main application code
│   ├── requirements.txt
│   └── README.md
│
├── databricks_reskill/                    # 📚 Learning Course
│   ├── 00_START_HERE.md                   # Entry point
│   ├── COURSE_JOURNEY.md                  # Learning path
│   ├── 01_Introduction.md                 # Module 1
│   ├── 03_spark_basics.md                 # Module 3
│   ├── 04_spark_sql.md                    # Module 4
│   ├── 05_performance_tuning.md           # Module 5
│   ├── 06_advanced_features.md            # Module 6
│   ├── 07_production_ready.md             # Module 7
│   └── scripts/                           # 30+ example scripts
│
├── spark/                                 # ⚡ PySpark Examples
│   ├── PYSPARK_README.md                  # Setup guide
│   ├── pyspark_working_example.py         # ✅ Verified working
│   ├── pyspark_demo_working.py
│   └── [multiple example files]
│
└── Synthetic_Data_Gen_Scripts-Claude/    # 📊 Data Generators
    ├── generate_all.py                    # Master generator
    ├── generate_customers.py
    ├── generate_orders.py
    └── [more generators]
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+ (tested with 3.13.5)
- pip / conda
- Java 11+ (for Spark projects)
- Git

### Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd AIEngineer

# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies for specific project
cd <project-folder>
pip install -r requirements.txt
```

### Run Specific Projects

**RAG Application:**
```bash
cd rag_app
python main.py                    # API server on http://localhost:8000
streamlit run ui.py              # UI on http://localhost:8501
```

**Sentiment Analysis:**
```bash
cd "Sentiment Analyzer"
python src/api.py
python src/ui.py
```

**Databricks Course:**
```bash
cd databricks_reskill
# Read documentation first
cat 00_START_HERE.md

# Run example scripts progressively
python scripts/01_basic_setup.py
python scripts/02_data_transformation.py
```

**PySpark:**
```bash
cd spark
python pyspark_working_example.py
```

**Generate Test Data:**
```bash
cd Synthetic_Data_Gen_Scripts-Claude
python generate_all.py
```

---

## 📊 Project Details

### RAG Application Architecture
```
┌──────────────────────────────────────────┐
│         User Query (Web/API)              │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Retriever         │
        │ (Semantic + BM25)   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Context Building   │
        │  + Prompt           │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   LLM (GPT-4o-mini) │
        │   Response Gen      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Return Answer      │
        │  + Context          │
        └─────────────────────┘
```

### Databricks Learning Path
```
Beginner ─→ Fundamentals ─→ Spark Basics ─→ Spark SQL
    ↓           ↓               ↓              ↓
Performance ─→ Advanced ─→ Production ─→ Expert
```

---

## 🛠️ Technologies & Stack

### Core Technologies
| Component | Technology | Version |
|-----------|-----------|---------|
| **Python** | Python | 3.9+ |
| **Spark** | PySpark | 4.1.1 |
| **Web Framework** | FastAPI | Latest |
| **UI Framework** | Streamlit | Latest |
| **Vector Store** | ChromaDB | Latest |
| **LLM** | OpenAI GPT-4o | Latest |
| **ML Models** | HuggingFace Transformers | Latest |

### Key Libraries
```
LangChain          # LLM orchestration
FastAPI            # REST APIs
Streamlit          # Web UIs
ChromaDB           # Vector database
OpenAI             # LLM & embeddings
PyMuPDF, python-docx, BeautifulSoup  # Document parsing
Pandas, NumPy      # Data manipulation
PySpark            # Distributed computing
PyTorch, Transformers  # ML models
Pytest             # Testing
```

---

## 📋 Setup & Installation

### Environment Variables
Most projects require API keys. Create `.env` file in project directories:

```bash
# For RAG App & Sentiment Analyzer
OPENAI_API_KEY=sk_...
```

### Docker Support
Several projects include Docker support:

```bash
# Build & run with Docker
docker-compose up -d

# Or individual containers
docker build -t rag-app .
docker run -p 8000:8000 -p 8501:8501 rag-app
```

### Testing
```bash
# RAG App tests
cd rag_app
pytest tests/ -v

# Sentiment Analyzer tests
cd "Sentiment Analyzer"
pytest -v
```

---

## 🚀 Advanced Usage

### Databricks Integration
Connect the scripts to a live Databricks workspace:
```python
from databricks.sql import sql

connection = sql.connect(
    server_hostname="<workspace>.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/<warehouse-id>",
    personal_access_token="<token>"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM my_table")
```

### Custom RAG Pipeline
Build your own RAG system:
```python
from rag_app.app.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)

pipeline.ingest_documents("path/to/docs")
answer = pipeline.query("Your question here")
```

---

## 📚 Learning Resources

### Inside This Repository
- **Databricks Course:** 7 modules, 25+ hours of content
- **PySpark Guide:** Complete setup with 5+ working examples
- **API Examples:** FastAPI best practices across 2 projects
- **RAG System:** Production-grade implementation

### Recommended External Resources
- [Apache Spark Documentation](https://spark.apache.org/docs/)
- [Databricks Academy](https://academy.databricks.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Guide](https://docs.trychroma.com/)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 for Python code
- Add tests for new features
- Update documentation
- Ensure all tests pass

---

## 📝 Project Documentation

Each project has detailed documentation:

| Project | Documentation |
|---------|--------------|
| RAG App | [rag_app/README.md](./rag_app/README.md) |
| Sentiment Analyzer | [Sentiment Analyzer/README.md](./Sentiment%20Analyzer/README.md) |
| Databricks Course | [databricks_reskill/00_START_HERE.md](./databricks_reskill/00_START_HERE.md) |
| PySpark | [spark/PYSPARK_README.md](./spark/PYSPARK_README.md) |
| Synthetic Data | [Synthetic_Data_Gen_Scripts-Claude/readme.md](./Synthetic_Data_Gen_Scripts-Claude/readme.md) |

---

## 🐛 Troubleshooting

### Common Issues

**PySpark not found:**
```bash
# Ensure Java is installed
java -version

# Install/upgrade PySpark
pip install --upgrade pyspark
```

**OpenAI API errors:**
```bash
# Check API key is set correctly
echo $OPENAI_API_KEY  # Should not be empty
```

**ChromaDB connection issues:**
```bash
# Clear and reinitialize
rm -rf vectorstore/
python scripts/ingest_docs.py
```

**Port already in use:**
```bash
# Change port in config
# RAG: app/config.py
# Sentiment: src/config.py
```

---

## 📞 Support

For issues, questions, or suggestions:

1. **Check documentation** - Most answers are in project READMEs
2. **Search existing issues** - Your question may be answered
3. **Create an issue** - Include error message, steps to reproduce
4. **Email** - Contact project maintainers

---

## 📄 License

This repository is provided as-is for educational and professional purposes.

---

## 🎓 Summary

This repository represents a **complete AI engineering portfolio** covering:
- ✅ **Modern RAG systems** (LLMs + retrieval)
- ✅ **ML/NLP applications** (sentiment analysis)
- ✅ **Big data processing** (PySpark, Databricks)
- ✅ **API development** (FastAPI)
- ✅ **Frontend integration** (Streamlit)
- ✅ **Data generation** (synthetic datasets)

**Start with:** [00_START_HERE.md in databricks_reskill](./databricks_reskill/00_START_HERE.md)

---

**Last Updated:** April 2026  
**Repository Status:** ✅ Active & Maintained  
**Difficulty Level:** Beginner to Advanced  
**Time Investment:** 50+ hours of learning & implementation

---

<div align="center">

**Happy Learning & Building! 🚀**

[⬆ back to top](#-ai-engineer---complete-portfolio-repository)

</div>
