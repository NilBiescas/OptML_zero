#!/bin/bash
wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar
mv data original
rm datasets.tar
