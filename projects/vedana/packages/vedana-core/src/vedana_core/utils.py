def cast_dtype(value, value_name: str, dtype: str):
    """
    умнейшая конвертация строковых переменных по заданному в дата-модели типу данных
    """

    try:
        if dtype == "int" or dtype == "float":
            if isinstance(value, str):
                if len(value.strip(",").split(",")) == 2:
                    value = value.replace(",", ".")
                elif len(value.strip(",").split(",")) > 2:  # assume "," is 000 delimeter
                    value = value.replace(",", "")
                elif len(value.strip(".").split(".")) > 2:  # assume "." is 000 delimeter
                    value = value.replace(".", "")

            value = float(value)
            if dtype == "int":
                value = int(value)

        elif dtype == "bool":
            return str(value).strip().lower() in ["1", "true", "да", "есть"]
        elif dtype == "str":
            return str(value)

        else:
            return value  # Fallback

    except (ValueError, TypeError):
        print(f"cast_dtype error for value {value_name} {value} (expected dtype {dtype})")

    return value
