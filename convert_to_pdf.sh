#!/bin/bash
# Convert markdown files to PDF using pandoc
# Run this script from the tabula-rasa-core directory

echo "Converting markdown files to PDF..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "pandoc not found. Installing..."
    # Try different package managers
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y pandoc texlive-xetex texlive-fonts-recommended
    elif command -v brew &> /dev/null; then
        brew install pandoc basictex
    else
        echo "Please install pandoc manually: https://pandoc.org/installing.html"
        exit 1
    fi
fi

# Convert description
echo "Converting Project Description..."
pandoc Main_High_TRCore_Description.md -o Main_High_TRCore_Description.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article

# Convert AI Usage
echo "Converting AI Usage Explanation..."
pandoc Main_High_TRCore_AI_Usage.md -o Main_High_TRCore_AI_Usage.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article

# Convert Presentation (using beamer)
echo "Converting Presentation..."
pandoc Main_High_TRCore_Presentation.md -o Main_High_TRCore_Presentation.pdf \
    --pdf-engine=xelatex \
    -t beamer \
    -V theme:metropolis \
    -V colortheme:dark \
    -V aspectratio:169

echo "Done! PDF files created:"
ls -la *.pdf