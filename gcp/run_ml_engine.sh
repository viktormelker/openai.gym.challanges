JOBNAME=rl_job_hp_$(date -u +%y%m%d_%H%M%S)
REGION=europe-west1
BUCKET=viktor_rl_test_bucket

gcloud ml-engine jobs submit training $JOBNAME \
        --package-path=$PWD/app \
        --module-name=app.run_lunar_lander \
        --region=$REGION \
        --staging-bucket=gs://$BUCKET \
        --config=gcp/hyperparameters.yaml \
        --runtime-version=1.10 \
        --python-version=3.5