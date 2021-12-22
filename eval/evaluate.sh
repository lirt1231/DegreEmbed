#!/usr/bin/env bash

usage()
{
    echo "usage: sysinfo_page [[[-d dataset ] [-i]] | [-h]]"
}

dataset=""
topk=(1 3 10)

while [ "$1" != "" ]; do
	case $1 in
		-d | --dataset )        shift
                                dataset="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
	esac
	shift                                
done

echo "evaluating $dataset dataset model results"

for i in "${topk[@]}"
do
	echo "Top $i results: " >> saved/"$dataset"/topk.txt
	echo "Top $i results: " >> saved/"$dataset"/topk_raw.txt
	python eval/evaluate.py --dataset=$dataset --include_reverse --top_k=$i --filter >> saved/"$dataset"/topk.txt
	python eval/evaluate.py --dataset=$dataset --include_reverse --top_k=$i >> saved/"$dataset"/topk_raw.txt
done






