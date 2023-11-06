#! /bin/bash

# pip install -U amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo

for x in train dev
do
    for y in outputs questions
    do
        amlt storage upload ../personalized_scholarly/${x}_${y}.json.gz uploads/data/personalized_scholarly/
    done
done

amlt storage list uploads/data/personalized_scholarly

# amlt storage upload ../personalized_product_rating/User_keq4_t5base_step1_iter3_loratruncaug uploads/data/personalized_product_rating/
