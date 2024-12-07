
from qa_tool import QATool

def main():

    banner = """
    ╔══════════════════════════════════════╗
    ║           PDF Q&A Tool               ║
    ║        -------------------           ║
    ║   AI-powered document assistant      ║
    ╚══════════════════════════════════════╝
        """
    print(banner)

    # Ask user for PDF path 
    pdf_path = input("Enter the path to the PDF file: ") 

    print()
    print('============================================')
    print('Initializing Q&A Tool please wait...')
    qa_tool = QATool(pdf_path)

    # Ask User to input queries until they enter an empty string 
    print()
    print('============================================')
    print()
    print("Enter your queries, and enter an empty string to exit.")
    print()

    queries = []
    idx=1
    while True:
        query = input(f"{idx}. Enter your query: ")
        if query == "":
            break 
        queries.append(query)
        # print(f'============================================')
        idx += 1 

    print("Answering Queries...")
    qa_dict = {} 
    for query in queries:
        answer = qa_tool.answer_query(query)
        qa_dict[query] = answer

    # Print Q&A pairs with clear separation and formatting
    print("\n" + "="*80 + "\n")  # Top border
    for query, answer in qa_dict.items():
        print(f"Question:\n{'-'*10}\n{query}\n")
        print(f"Answer:\n{'-'*10}\n{answer}\n")
        print("="*80 + "\n")  #f Separator between Q&A pairs

    # Delete the pinecode index 
    qa_tool.delete_pinecone_index() 

if __name__ == "__main__":
    main()