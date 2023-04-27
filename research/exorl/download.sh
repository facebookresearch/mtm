#!/bin/bash

S3_URL=https://dl.fbaipublicfiles.com/exorl
DOMAIN=${1:-walker}
ALGO=${2:-proto}

DIR=./datasets/${DOMAIN}
mkdir -p ${DIR}/${ALGO}

URL=${S3_URL}/${DOMAIN}/${ALGO}.zip

echo "downloading ${ALGO} dataset for ${DOMAIN} from ${URL}..."

wget ${URL} -P ${DIR}

echo "unzipping ${ALGO}.zip into ${DIR}/${ALGO}..."

unzip -q ${DIR}/${ALGO}.zip -d ${DIR}/${ALGO}

rm ${DIR}/${ALGO}.zip
