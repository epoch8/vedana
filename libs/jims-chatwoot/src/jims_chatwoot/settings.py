from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatwootSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="CHATWOOT_"
    )

    url: str
    account_id: str
    agent_bot_access_token: str
    admin_access_token: str

    audio_content_intent: str = "/audio_message"
    media_content_intent: str = "/media_message"

class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="JIMS_CHATWOOT_"
    )

    token: str


chatwoot_settings = ChatwootSettings()  # type: ignore
api_settings = ApiSettings()  # type: ignore
