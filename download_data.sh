#!/bin/bash

mkdir -p data/iwildcam_v2.0/
mkdir -p ckpts/

fileid="1l3o4TL0Acq_xmmIUEJRejrxZEmQthG-U"
filename="data/iwildcam_v2.0/dataset_subtree.csv"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


fileid="19cMXfFew4c9RzwG8iU2GFPQdC6xbryoQ"
filename="ckpts/species_class_model.pt"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}

