import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd

# LangChain + Chroma
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

from src.prompts.rag_emotion_prompt import RAG_EMOTION_PROMPT_TEMPLATE_SYSTEM, RAG_EMOTION_PROMPT_TEMPLATE_USER

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
    parser.add_argument("--prev_folder", default="data/split_prev",
                        help="Folder that contains 'annotator_{id}_prev.csv' files")
    parser.add_argument("--test_folder", default="data/split_test",
                        help="Folder that contains 'annotator_{id}_test.csv' files")
    parser.add_argument("--output_folder", default="data/personalization",
                        help="Folder to store final RAG-based results")
    parser.add_argument("--text_csv", default="data/original/StudEmo_text_data.csv",
                        help="CSV file that has 'text_id' and 'text' columns")
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
        prev_csv = os.path.join(args.prev_folder, f"annotator_{annot_id}_prev.csv")
        test_csv = os.path.join(args.test_folder, f"annotator_{annot_id}_test.csv")
        
        if not os.path.exists(prev_csv):
            print(f"File not found: {prev_csv}")
            continue
        if not os.path.exists(test_csv):
            print(f"File not found: {test_csv}")
            continue
        
        prev_df = pd.read_csv(prev_csv)
        if "text_id" not in prev_df.columns:
            print(f"No 'text_id' column in {prev_csv}, skipping.")
            continue

        merged_prev = pd.merge(prev_df, text_df[["text_id","text"]], on="text_id", how="left")

        docs = []
        for i, row in merged_prev.iterrows():
            annotation_info = row.drop(["text_id","text"]).to_dict()
            info_str = ", ".join([f"{k}={v}" for k,v in annotation_info.items()])
            doc_content = f"({info_str})\nFullText: {row['text']}"
            
            doc = Document(page_content=str(doc_content), metadata={"row_index": i})
            docs.append(doc)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(docs, embeddings)

        test_df = pd.read_csv(test_csv)
        if "text_id" not in test_df.columns:
            print(f"No 'text_id' column in {test_csv}, skipping.")
            continue

        merged_test = pd.merge(test_df, text_df[["text_id","text"]], on="text_id", how="left")

        outputs = []
        for idx, row in merged_test.iterrows():
            new_text = str(row["text"])
            
            # Retrieve docs
            retrieved_docs = vectorstore.similarity_search(new_text, k=args.k_retrieval)
            rag_context = "\n\n".join([f"PrevDoc({d.metadata['row_index']}): {d.page_content}"
                                       for d in retrieved_docs])

            # Build final prompt
            system_prompt = RAG_EMOTION_PROMPT_TEMPLATE_SYSTEM
            user_prompt = RAG_EMOTION_PROMPT_TEMPLATE_USER.format(rag_context=rag_context, post_text=new_text)

            raw_response = inference_func(system_prompt=system_prompt, user_prompt=user_prompt)
            outputs.append(raw_response)

            if (idx + 1) % 50 == 0:
                print(f"Annotator {annot_id}, processed {idx+1} lines...")

        merged_test[f"{args.model}_rag_output"] = outputs

        out_path = os.path.join(args.output_folder, f"rag_result_{annot_id}.csv")
        merged_test.to_csv(out_path, index=False)
        print(f"Saved {len(merged_test)} RAG-based results to {out_path}")

if __name__ == "__main__":
    main()
