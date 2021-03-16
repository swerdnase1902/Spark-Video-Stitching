#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: ./create_cluster.sh  bucket-name  region  zone"
    exit
fi

PROJECT=$(gcloud config get-value project)
BUCKET=$1
REGION=$2
ZONE=$3
INSTALL=gs://$BUCKET/pip-install.sh

#upload install 
gsutil cp pip-install.sh $INSTALL

gcloud beta dataproc clusters create \
   --num-workers=2 \
   --worker-machine-type=n1-standard-1 \
   --master-machine-type=n1-standard-1 \
   --image-version=1.5-ubuntu18 \
   --enable-component-gateway \
   --optional-components=ANACONDA,JUPYTER \
   --zone=$ZONE \
   --region=$REGION \
   --metadata 'PIP_PACKAGES=numpy scipy pandas sklearn pillow matplotlib imutils opencv-python==4.2.0.34 python-resize-image' \
   --initialization-actions=$INSTALL \
   --initialization-action-timeout 5m \
   cluster1
   