#! /bin/bash

for x in train dev
do
    for y in outputs questions
    do
        amlt storage upload ../personalized_product_rating/${x}_${y}.json.gz uploads/data/personalized_product_rating/
    done
done

amlt storage list uploads/data/personalized_product_rating

# amlt storage upload ../personalized_product_rating/User_keq4_t5base_step1_iter3_loratruncaug uploads/data/personalized_product_rating/
