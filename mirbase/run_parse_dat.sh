SGE_Batch -c "cat miRNA.dat | perl -ne 's/<p>//g; s/<\/p>//g; s/<br>/\n/g; print;' > miRNA.dat.txt" -r parse_dat -q nucleotide
