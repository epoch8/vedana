import os
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyQuery
from pydantic import BaseModel



class TriggerIntentModel(BaseModel):
    name: str
    entities: dict[str, Any] | None = None


_query_scheme = APIKeyQuery(name="token_stell", auto_error=False)

stell_api_token = os.getenv("STELL_API_TOKEN", "")


def verify_token(
    token: str | None = Security(_query_scheme),
) -> None:
    if stell_api_token and token != stell_api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
Auth = Annotated[None, Depends(verify_token)]


def get_stell_api() -> FastAPI:
    api = FastAPI()


    @api.post("/tracker/conversations/{conversation_id}/trigger_intent", tags=["Tracker"], status_code=status.HTTP_200_OK)
    async def conversation_trigger_intent(
        _: Auth,
        conversation_id: str,
        request: TriggerIntentModel,
    ) -> int:
        return status.HTTP_200_OK

    @api.get("/tracker/conversations/{conversation_id}", tags=["Tracker"], status_code=status.HTTP_200_OK)
    async def get_tracker(_: Auth, conversation_id: str) -> None:
        pass

    return api


stell_api = get_stell_api()
