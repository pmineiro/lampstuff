cd personalized_product_rating
cp /mnt/default/uploads/data/personalized_product_rating/*.json.gz .
cp -a /mnt/default/uploads/data/personalized_product_rating/User_keq4_t5xxl_step1_iter* .
(mkdir -p ~/.cache; cp -a /mnt/default/uploads/cache/* ~/.cache/)
env AUGMENT=$1 python ../runnbcell.py -y User_keq4_t5xxl.ipynb 1 "$AMLT_OUTPUT_DIR"/User_keq4_t5xxl.ipynb
