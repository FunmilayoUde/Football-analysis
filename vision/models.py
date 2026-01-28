from __future__ import annotations
from inference import get_model


def load_roboflow_models(api_key: str, player_model_id: str, field_model_id: str):
    player_model = get_model(model_id=player_model_id, api_key=api_key)
    field_model = get_model(model_id=field_model_id, api_key=api_key)
    return player_model, field_model
