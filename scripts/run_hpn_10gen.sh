CUDA_VISIBLE_DEVICES="0" /home/lwq/anaconda3/envs/pymarl3/bin/python ./src/main.py \
    --config=hpn_qmix \
    --env-config=sc2_v2_terran \
    with \
    obs_agent_id=True \
    obs_last_action=False \
    runner=parallel \
    batch_size_run=8 \
    batch_size=128 \
    buffer_size=5000 \
    t_max=10050000 \
    epsilon_anneal_time=100000 \
    td_lambda=0.6 \
    mixer=qmix \