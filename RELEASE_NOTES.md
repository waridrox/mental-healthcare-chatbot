# Changelog

## [v0.2.0]
### Overview:
This update introduces significant enhancements to the chatbot's retrieval-augmented generation (RAG) functionality, primarily focusing on the data used for building the FAISS vector store.

### Updated Data Sources:
The chatbot now incorporates curated datasets from the **National Institute of Mental Health (NIMH)**. The NIMH is a leading federal agency for research on mental disorders, providing valuable resources and data aimed at improving mental health care and advancing the understanding of mental illnesses. These datasets include comprehensive mental health information, facilitating research and insights into effective treatments and interventions.

### Key Changes:
- **Data Integration:** Curated datasets from the NIMH have been incorporated into the system to build the FAISS vector store, enhancing the quality and relevance of the information processed by the chatbot.
- **Data Storage:** The previous `data/` directory has been removed as it has been replaced by a single serialized file, `faiss_vectorstore.pkl`, which now contains the FAISS vectors derived from the NIMH data.
- **Performance Improvement:** The new storage format improves loading times and simplifies data management.
- **Backward Compatibility:** Please note that this change may affect existing setups. Users will need to migrate to the new `faiss_vectorstore.pkl` format.

### Migration Steps:
1. Remove the existing `data/` directory as it has been deprecated.
2. Use the new `data.pickle` file for FAISS vector storage.

### Additional Notes:
We recommend backing up existing data before upgrading. For detailed information on migration and usage, please refer to the updated documentation.

## [v0.1.0]
### Overview:
This initial release of the chatbot utilizes the FAISS vector store to handle mental health queries.

### Data Source:
In version **v0.1.0**, the chatbot relied on the PDF titled **[Defining Mental Health and Mental Illness](https://www.researchgate.net/publication/328248529_Defining_mental_health_and_mental_illness)**. This document provided foundational insights into mental health and mental illness, which were essential for processing relevant queries effectively.

### Key Features:
- **FAISS Vector Store:** Implemented for efficient retrieval of mental health-related information based on the provided PDF.
- **Query Handling:** Enabled the chatbot to respond to user inquiries using the indexed content from the PDF.

### Notes:
This version served as a foundation for subsequent enhancements, laying the groundwork for future updates that will incorporate more extensive data sources and improved functionalities.
