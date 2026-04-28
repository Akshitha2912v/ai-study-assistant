from rag_pipeline import process_pdf_and_answer

# Change this to any PDF you have on your Mac
pdf_path = "uploads/sql-basics-cheat-sheet-a4.pdf"
question = "What is this document about?"

answer, sources = process_pdf_and_answer(pdf_path, question)

print("\n=== ANSWER ===")
print(answer)
print(f"\nSources: Pages {sources}")