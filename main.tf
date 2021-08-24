terraform {
	backend "s3" {
		bucket = "${STATEFILE_BUCKET}"
		key = "${APP_NAME}/${APP_VER}/tfstate"
		region = "${AWS_REGION}"
	}
}

provider "aws" {
	region = var.inp.aws_region
}

data "aws_caller_identity" "current" {}

module "main_bucket" {
	source = "./terraform/bucket"
	bucket_name = "bucket"
	inp = var.inp
}

module "sagemaker_role" {
	source = "./terraform/role"
	role_name = "sagemaker"
	service = "sagemaker"
	attached_policies = ["AmazonSageMakerFullAccess", "AmazonS3FullAccess"]
	inp = var.inp
}

module "firehose_role" {
	source = "./terraform/role"
	role_name = "firehose"
	service = "firehose"
	attached_policies = ["AmazonKinesisFirehoseFullAccess", "AmazonS3FullAccess"]
	inp = var.inp
}

module "hist_stream" {
	source = "./terraform/firehose"
	name = "hist_stream"
    role_arn = module.firehose_role.arn
	service = "firehose"
	attached_policies = ["AmazonKinesisFirehoseFullAccess", "AmazonS3FullAccess"]
	inp = var.inp
}

# output

output "main_bucket" {
  value = module.main_bucket.bucket_name
}


output "main_bucket_arn" {
  value = module.main_bucket.arn
}

output "sagemaker_role_arn" {
  value = module.sagemaker_role.arn
}
