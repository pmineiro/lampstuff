cd personalized_news_cat
cp /mnt/default/uploads/data/personalized_news_cat/*.json.gz .
(mkdir -p ~/.cache; cp -a /mnt/default/uploads/cache/* ~/.cache/)
env AUGMENT=$1 python ../runnbcell.py -y User_keq4_t5base.ipynb 0 "$AMLT_OUTPUT_DIR"/User_keq4_t5base.ipynb
