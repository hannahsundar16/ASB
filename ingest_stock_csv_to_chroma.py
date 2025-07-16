
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables (must include OPENAI_API_KEY)
load_dotenv()

def prepare_documents_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    documents = []

    for i, row in df.iterrows():
        metadata = {
            "id": row["id"],
            "symbol": row["symbol"],
            "name": f"{row['first_name']} {row['last_name']}",
            "brokerage": row["brokerage"],
            "trade_date": row["trade_date"]
        }

        content = f"""
        Trade: {row['transaction_type']} {row['quantity']} shares of {row['symbol']} ({row['company_name']})
        Price: ${row['price']} | Total: ${row['total_value']}
        Date: {row['trade_date']} at {row['trade_time']}
        Portfolio: ${row['portfolio_value']} | Risk: {row['risk_tolerance']} | Goal: {row['investment_goal']}
        """

        documents.append(Document(page_content=content.strip(), metadata=metadata))

    return documents

def ingest_to_chroma(documents, persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"✅ Ingested {len(documents)} records into Chroma at '{persist_directory}'")

if __name__ == "__main__":
    csv_path = "synthetic_stock_dataset_small.csv"
    persist_directory = "vector_db_stock"

    docs = prepare_documents_from_csv(csv_path)
    ingest_to_chroma(docs, persist_directory)