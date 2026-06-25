import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpAuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enable_mcp: bool = False
    api_key: str | None = None
    authentik_url: str | None = None
    authentik_app_slug: str | None = None


class VedanaTokenVerifier(TokenVerifier):
    def __init__(self, settings: McpAuthSettings) -> None:
        super().__init__()
        self.api_key = settings.api_key
        self.authentik_url = settings.authentik_url.rstrip("/") if settings.authentik_url else None
        self.authentik_app_slug = settings.authentik_app_slug

    async def verify_token(self, token: str) -> AccessToken | None:
        if self.api_key is not None and token == self.api_key:
            return AccessToken(token=token, client_id="api-key", scopes=[])

        if self.authentik_url and self.authentik_app_slug:
            url = f"{self.authentik_url}/api/v3/core/applications/{self.authentik_app_slug}/check_access/"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            logger.debug("Authentik check_access: status={} body={}", resp.status_code, resp.text)
            if resp.status_code == 200 and resp.json().get("passing") is True:
                return AccessToken(token=token, client_id="authentik", scopes=[])
            logger.warning("Authentik check_access failed: status={} passing={}", resp.status_code, resp.json() if resp.status_code == 200 else resp.text)
        else:
            logger.warning("Authentik not configured: authentik_url={} authentik_app_slug={}", self.authentik_url, self.authentik_app_slug)

        return None


def make_token_verifier() -> VedanaTokenVerifier | None:
    settings = McpAuthSettings()
    if not settings.api_key and not settings.authentik_url:
        return None
    return VedanaTokenVerifier(settings)
