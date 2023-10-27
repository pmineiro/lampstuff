#! /bin/bash

for x in huggingface torch
do
    amlt storage upload ~/.cache/${x} uploads/cache/
done

amlt storage list uploads/cache
