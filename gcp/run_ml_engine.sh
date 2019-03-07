JOBNAME=rl_job_hp_$(date -u +%y%m%d_%H%M%S)
REGION=europe-north1
BUCKET=viktor_rl_test_bucket

gcloud ml-engine jobs submit training $JOBNAME \
        --package-path=$PWD/rl_on_gcp \
        --module-name=run_lunar_lander \
        --region=$REGION \
        --staging-bucket=gs://$BUCKET \
        --config=gcp/hyperparameters.yaml \
        --runtime-version=1.10 \
        --\
        --steps=10000\
        --start_train=10000\
        --buffer_size=10000\
        --model_dir='gs://viktor_rl_test_bucket/rl_on_gcp'