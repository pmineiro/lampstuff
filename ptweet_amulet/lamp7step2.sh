cd personalized_tweet
cp /mnt/default/uploads/data/personalized_tweet/*.json.gz .
cp -a /mnt/default/uploads/data/personalized_tweet/User_keq4_t5base_step1_iter* .
(mkdir -p ~/.cache; cp -a /mnt/default/uploads/cache/* ~/.cache/)
env AUGMENT=$1 GAMMA=$2 python ../runnbcell.py -y User_keq4_t5base.ipynb 1 "$AMLT_OUTPUT_DIR"/User_keq4_t5base.ipynb
