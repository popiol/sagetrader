# inputs

variable "bucket_name" {
	type = string
}

variable "inp" {
	type = map(string)
}

variable "archived_paths" {
    type = list(string)
    default = []
}

# definition

resource "aws_s3_bucket" "main" {
	bucket = "${var.inp.aws_user}.${replace(var.inp.app_id,"_","-")}-${var.bucket_name}"
	acl = "private"
	tags = var.inp.app
	force_destroy = var.inp.temporary

	dynamic "lifecycle_rule" {
		for_each = toset(var.archived_paths)

		content {
			id = "archive${replace(lifecycle_rule.key,"/","_")}"
			enabled = true
			prefix = lifecycle_rule.key

			transition {
				days = 30
				storage_class = "STANDARD_IA"
			}

			transition {
				days = 360
				storage_class = "GLACIER"
			}
		}
	}
}

data "aws_iam_policy_document" "access" {
	policy_id = "${var.inp.app_id}_${var.bucket_name}_s3"

	statement {
		actions = [
			"s3:GetObject",
			"s3:PutObject",
			"s3:DeleteObject"
		]
		resources = [
			"${aws_s3_bucket.main.arn}/*"
		]
	}

	statement {
		actions = [
			"s3:ListBucket"
		]
		resources = [
			"${aws_s3_bucket.main.arn}"
		]
	}
}

# output

output "access_policy" {
  value = data.aws_iam_policy_document.access.json
}

output "name" {
  value = aws_s3_bucket.main.bucket
}

output "arn" {
  value = aws_s3_bucket.main.arn
}
