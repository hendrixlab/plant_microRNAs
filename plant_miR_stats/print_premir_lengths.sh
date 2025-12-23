#!/bin/bash
cat plant_data2.txt | grep -v '>' | grep -v 0 | grep -v '(' | perl -ne 'chomp; print length($_), "\n"' | sort -n | uniq -c
