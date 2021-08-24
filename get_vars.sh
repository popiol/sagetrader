#!/bin/bash

export APP_VER=`echo "$CI_COMMIT_BRANCH" | tr '[:upper:]' '[:lower:]'`

if [[ "$APP_VER" == "master" || "$APP_VER" == "$CI_DEFAULT_BRANCH" ]]; then
	export TEMP_DEPLOY="false"
else
	export TEMP_DEPLOY="true"
fi

envsubst < config.tfvars > config.tfvars.new
mv config.tfvars.new config.tfvars

envsubst < main.tf > main.tf.new
mv main.tf.new main.tf

#cd terraform/lambda
#dirs=`ls -d1 */ | sed "s/\/$//" | tr '\n' ' '`
#for dir in $dirs
#do
#	cd $dir
#	zip -r ../$dir.zip .
#	cd ..
#done
#cd ../..
