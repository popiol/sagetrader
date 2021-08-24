# input

variable "inp" {
	type = map(string)
}

variable "name" {
	type = string
}

variable "role_arn" {
	type = string
}

variable "bucket_arn" {
	type = string
    default = ""
}

variable "bucket_prefix" {
	type = string
    default = "data/"
}

variable "http_url" {
	type = string
    default = ""
}

# definition

resource "aws_kinesis_firehose_delivery_stream" "main" {
	name = "${var.inp.app_id}_${var.name}"
	destination = "http_endpoint"

    dynamic "s3_configuration" {
        for_each = toset(var.bucket_arn == "" ? [] : [var.bucket_arn])

        content {
            role_arn = var.role_arn
            bucket_arn = var.bucket_arn
            buffer_size = 10
            buffer_interval = 600
            compression_format = "GZIP"
            prefix = var.bucket_prefix
        }
    }

    dynamic "http_endpoint_configuration" {
        for_each = toset(var.http_url == "" ? [] : [var.http_url])
        
        content {
            url = var.http_url
            buffering_size = 1
            buffering_interval = 60
            role_arn = var.role_arn
            request_configuration {
                content_encoding = "GZIP"
            }
        }
	}
}

#output

output "arn" {
	value = aws_kinesis_firehose_delivery_stream.main.arn
}
