#!/usr/bin/env bash
dim=$1
outfn=$2
python supervised.ppdb.py --embed 1 --dim $dim --test_baseline 1 --dim_divide 4 > $outfn
python supervised.ppdb.py --embed 1 --dim $dim --test_baseline 0 --dim_divide 10 >> $outfn.10
