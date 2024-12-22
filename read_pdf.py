import PyPDF2

# Path to your PDF file (same folder as the script)
pdf_path = "FACstructure.pdf"

try:
    # Open the PDF file in read-binary mode
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        pdf_text = ""

        # Extract text from each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()

    # Check if pdf_text is populated
    if not pdf_text:
        print("No text extracted from the PDF.")
    else:
        print("Text extracted successfully!")

    # Split the extracted text into chunks by new lines
    text_chunks = pdf_text.split('\n')

    # Print first few text chunks for verification
    print(text_chunks[:5])

except Exception as e:
    print(f"An error occurred: {e}")
