from pydantic_settings import BaseSettings, SettingsConfigDict


class K8sConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="K8S",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enable_jobs: bool = True
    job_cpu_limit: str = "2"
    job_memory_limit: str = "4Gi"
    job_cpu_request: str = "500m"
    job_memory_request: str = "2Gi"


k8s_config = K8sConfig()  # type: ignore

