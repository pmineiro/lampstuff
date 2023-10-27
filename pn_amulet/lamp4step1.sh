cd personalized_news
cp /mnt/default/uploads/data/personalized_news/*.json.gz .
(mkdir -p ~/.cache; cp -a /mnt/default/uploads/cache/* ~/.cache/)
env AUGMENT=$1 python ../runnbcell.py -y User_keq4_t5base.ipynb 0 "$AMLT_OUTPUT_DIR"/User_keq4_t5base.ipynb
