import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd

# LangChain + Chroma
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

from src.prompts.rag_emotion_prompt import (
    RAG_EMOTION_PROMPT_TEMPLATE_SYSTEM,
    RAG_EMOTION_PROMPT_TEMPLATE_USER
)

from scripts.run_mistral import run_mistral_inference
from scripts.run_gpt4o import run_gpt4o_inference
from scripts.run_qwen import run_qwen_inference
from scripts.run_llama import run_llama_inference
from scripts.run_mentallama import run_mentallama_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        choices=["mistral","gpt4o","qwen2.5","llama","mental_llama"],
                        help="Which model to use for emotion annotation")
    parser.add_argument("--annotator_ids", type=str, required=True,
                        help="Comma-separated list of annotator IDs (e.g. '0,1,2')")
    parser.add_argument("--split_folder", default="data/annotator_split",
                        help="Folder that contains 'annotation_data_annotator_{id}.csv' files")
    parser.add_argument("--text_csv", default="data/original/StudEmo_text_data.csv",
                        help="CSV file that has 'text_id' and 'text' columns")
    parser.add_argument("--output_folder", default="data/personalization",
                        help="Folder to store final RAG-based results")
    parser.add_argument("--rag_size", type=int, default=50,
                        help="Number of lines to use as RAG reference (excluding header)")
    parser.add_argument("--k_retrieval", type=int, default=3,
                        help="Number of docs to retrieve for each new text")
    args = parser.parse_args()

    if args.model == "mistral":
        inference_func = run_mistral_inference
    elif args.model == "gpt4o":
        inference_func = run_gpt4o_inference
    elif args.model == "qwen2.5":
        inference_func = run_qwen_inference
    elif args.model == "llama":
        inference_func = run_llama_inference
    elif args.model == "mental_llama":
        inference_func = run_mentallama_inference
    else:
        raise ValueError(f"Unknown model: {args.model}")

    text_df = pd.read_csv(args.text_csv)
    if "text_id" not in text_df.columns or "text" not in text_df.columns:
        raise ValueError("text_csv must have at least 'text_id' and 'text' columns.")

    annotator_ids = [int(a.strip()) for a in args.annotator_ids.split(",")]

    os.makedirs(args.output_folder, exist_ok=True)

    for annot_id in annotator_ids:
        in_csv = os.path.join(args.split_folder, f"annotation_data_annotator_{annot_id}.csv")
        if not os.path.exists(in_csv):
            print(f"File not found: {in_csv}")
            continue

        df = pd.read_csv(in_csv)
        if "text_id" not in df.columns:
            print(f"No 'text_id' column in {in_csv}, skipping.")
            continue

        # Convert the first rag_size lines (excluding header) as RAG reference
        rag_df = df.head(args.rag_size).copy()
        test_df = df.iloc[args.rag_size:].copy()
        print(f"Annotator {annot_id}: Using {len(rag_df)} lines for RAG, {len(test_df)} lines for test.")

        # Merge RAG reference with text CSV
        rag_merged = pd.merge(rag_df, text_df[["text_id","text"]], on="text_id", how="left")

        docs = []
        for i, row in rag_merged.iterrows():
            annotation_info = row.drop(["text_id","text"]).to_dict()
            info_str = ", ".join([f"{k}={v}" for k,v in annotation_info.items()])
            doc_content = f"({info_str})\nFullText: {row['text']}"
            doc = Document(page_content=str(doc_content), metadata={"row_index": i})
            docs.append(doc)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embeddings)

        # Merge test lines with text CSV
        test_merged = pd.merge(test_df, text_df[["text_id","text"]], on="text_id", how="left")
        outputs = []

        for idx, row in test_merged.iterrows():
            new_text = str(row["text"])

            # Retrieve docs
            retrieved_docs = vectorstore.similarity_search(new_text, k=args.k_retrieval)
            rag_context = "\n\n".join(
                f"PrevDoc({d.metadata['row_index']}): {d.page_content}" for d in retrieved_docs
            )

            # Build final prompt
            system_prompt = RAG_EMOTION_PROMPT_TEMPLATE_SYSTEM
            user_prompt = RAG_EMOTION_PROMPT_TEMPLATE_USER.format(
                rag_context=rag_context,
                post_text=new_text
            )

            raw_response = inference_func(system_prompt=system_prompt, user_prompt=user_prompt)
            outputs.append(raw_response)

            if (idx + 1) % 50 == 0:
                print(f"Annotator {annot_id}, processed {idx+1} test lines...")

        test_merged["model_annotations"] = outputs

        out_csv = os.path.join(args.output_folder, f"{args.model}_rag_{args.rag_size}_{args.k_retrieval}_result_{annot_id}.csv")
        test_merged.to_csv(out_csv, index=False)
        print(f"Saved {len(test_merged)} RAG-based results to {out_csv}")

if __name__ == "__main__":
    main()
