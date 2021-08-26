#!/bin/bash

export APP_VER=`echo "$CI_COMMIT_BRANCH" | tr '[:upper:]' '[:lower:]'`

if [ -z "$APP_VER" ]; then
    export APP_VER=`git branch --show-current`
fi

if [[ "$APP_VER" == "main" || "$APP_VER" == "$CI_DEFAULT_BRANCH" ]]; then
	export TEMP_DEPLOY="false"
else
	export TEMP_DEPLOY="true"
fi

envsubst < config.tfvars.templ > config.tfvars
envsubst < main.tf.templ > main.tf

#cd terraform/lambda
#dirs=`ls -d1 */ | sed "s/\/$//" | tr '\n' ' '`
#for dir in $dirs
#do
#	cd $dir
#	zip -r ../$dir.zip .
#	cd ..
#done
#cd ../..
