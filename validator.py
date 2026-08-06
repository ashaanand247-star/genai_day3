from pydantic import ValidationError


def validate_response(response, output_model):
    try:
        validated_data = output_model.model_validate(response)
        return validated_data, None

    except ValidationError as e:
        return None, str(e)