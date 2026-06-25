#!/bin/bash
# Install texlive-full and compile the paper
# Run once to get the PDF

echo "Installing texlive (may take several minutes)..."
sudo apt-get install -y texlive-full 2>&1

cd "$(dirname "$0")"
echo "Compiling paper..."
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
echo "Done. PDF: docs/paper.pdf"
