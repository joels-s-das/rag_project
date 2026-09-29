from rag02 import (
    dotenv_load,
    initialize_components,
    load_data,
    create_vectorstore,
    get_qa_chain,
    split_docs
)
# PDF + CSV → Load → Split → 
# Embedding → Chroma → Groq → Answer

# Step 1: Load env
dotenv_load()

# Step 2: Init LLM
initialize_components()

# Step 3: Load data
urls = [
    "https://www.cnbc.com/2026/04/22/tesla-tsla-q1-2026-earnings-report.html",
    "https://www.cnbc.com/2026/01/28/tesla-tsla-2025-q4-earnings.html"
]


documents = load_data(urls)   # ✅ MUST exist
print("Loaded docs:", len(documents))  # debug

# Step 4 (chunking)
documents = split_docs(documents)   # ✅ safe now

print("After split:", len(documents))  # debug

# Step 5
vectorstore = create_vectorstore(documents)



# Step 6: QA chain
qa_chain = get_qa_chain(vectorstore)

# Step 7: Query
#query = "What are Tesla's earnings?"
query = "Summarize Tesla earnings by year and by quarter from the given articles. Include Q4 2025 and Q1 2026 details."
response = qa_chain.run(query)

print("\nAnswer:\n", response)

'''
Answer:
 The article does not provide a comprehensive summary of Tesla's earnings by year and by quarter. However, it does mention the following information:

1. **Annual declines**: Tesla has recorded annual declines in the past two years.
2. Q1 2026:
    1. **Vehicle deliveries**: 358,023
    2. **Revenue**: $22.39 billion (vs. $22.64 billion expected)
    3. **Earnings per share**: 41 cents adjusted (vs. 37 cents expected)
    4. **Net income**: $477 million (vs. $409 million a year earlier)
    5. **Automotive gross margins**: 19.2%
3. Year-ago quarter (Q1 2025):
    1. **Revenue**: $19.3 billion
    2. **Auto segment revenue**: $14 billion
    3. **Net income**: $409 million
4. No information is provided about Q4 2025.

Note that the article does not provide a complete picture of Tesla's earnings
 by year and by quarter. If you need more information, you may need to consult additional sources.
'''

