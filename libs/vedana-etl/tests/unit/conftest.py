import pytest

import pandas as pd


@pytest.fixture
def dm_anchors_df():
    return pd.DataFrame(
        [
            {"noun": "Article", "description": "", "id_example": "a1", "query": ""},
            {"noun": "Author", "description": "", "id_example": "u1", "query": ""},
        ]
    )


@pytest.fixture
def dm_attributes_df():
    return pd.DataFrame(
        [
            {
                "attribute_name": "title",
                "anchor": "Article",
                "description": "",
                "link": None,
                "data_example": "",
                "embeddable": True,
                "query": "",
                "dtype": "str",
                "embed_threshold": 0.0,
            },
            {
                "attribute_name": "year",
                "anchor": "Article",
                "description": "",
                "link": None,
                "data_example": "",
                "embeddable": False,
                "query": "",
                "dtype": "int",
                "embed_threshold": 0.0,
            },
        ]
    )


@pytest.fixture
def dm_links_df():
    # has_direction=False -> допустима и обратная пара узлов
    return pd.DataFrame(
        [
            {
                "anchor1": "Author",
                "anchor2": "Article",
                "sentence": "WROTE",
                "description": "",
                "query": "",
                "anchor1_link_column_name": None,
                "anchor2_link_column_name": None,
                "has_direction": False,
            },
        ]
    )
