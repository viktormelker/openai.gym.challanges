gcloud ml-engine local train --module-name app.run_lunar_lander \
          --package-path=$PWD/app \
          --distributed \
          --parameter-server-count 4 \
          --worker-count 8