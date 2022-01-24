from cvutils import Validator


def validate_label(label):
    v = Validator("rw")
    return v.validate(label)
