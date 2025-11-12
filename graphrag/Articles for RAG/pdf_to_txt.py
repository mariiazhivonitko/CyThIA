import os
import PyPDF2

def pdf_to_txt(input, output):
    # Make sure output folder exists
    os.makedirs(output, exist_ok=True)

    # Loop over all PDF files in the input folder
    for filename in os.listdir(input):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output, txt_filename)

            try:
                with open(pdf_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""  # extract page text

                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(text)

                print(f"Converted: {filename} to {txt_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input = "input"    # change to your folder path
output = "output"  # folder where txts will be saved
pdf_to_txt(input, output)
