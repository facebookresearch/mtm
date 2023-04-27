#!/bin/bash

# Add license header by running the following command
# ./oss_utils/add_license_header.sh

for i in $(find * -name '*.py');
do
  if ! grep -q Copyright $i
  then
    cat oss_utils/copyright.txt $i >$i.new && mv $i.new $i
  fi
done
