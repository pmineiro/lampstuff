#! /bin/bash

# pip install -U amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo

for x in train dev
do
    for y in outputs questions
    do
        amlt storage upload ../personalized_news_cat/${x}_${y}.json.gz uploads/data/personalized_news_cat/
    done
done

amlt storage list uploads/data/personalized_news_cat

# amlt storage upload ../personalized_news_cat/User_keq4_t5base_step1_iter3_loratruncaug uploads/data/personalized_news_cat/
