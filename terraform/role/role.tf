# input

variable "inp" {
	type = map(string)
}

variable "role_name" {
	type = string
}

variable "service" {
	type = string
}

variable "attached_policies" {
	type = list(string)
    default = []
}

#definitions

resource "aws_iam_role" "main" {
	name = "${var.inp.app_id}_${var.role_name}"
	assume_role_policy = data.aws_iam_policy_document.main.json
}

data "aws_iam_policy_document" "main" {
	statement {
		actions = [
			"sts:AssumeRole"
		]
		principals {
			type = "Service"
			identifiers = [
				"${var.service}.amazonaws.com"
			]
		}
	}
}

resource "aws_iam_role_policy_attachment" "main" {
    for_each = toset(var.attached_policies)
	role = aws_iam_role.main.name
	policy_arn = "arn:aws:iam::aws:policy/${each.key}"
}

#output

output "arn" {
  value = aws_iam_role.main.arn
}

output "name" {
  value = aws_iam_role.main.name
}
