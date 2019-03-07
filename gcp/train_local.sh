JOBNAME=rl_job_hp_$(date -u +%y%m%d_%H%M%S)
REGION=europe-west1
BUCKET=viktor_rl_test_bucket

gcloud ml-engine local train --module-name app.run_lunar_lander \
          --package-path=$PWD/app \
          --distributed \
          --parameter-server-count 4 \
          --worker-count 8