import logging
import os
import time
from typing import Any, Callable, Iterator

from kubernetes import client, config
from kubernetes.client import (
    BatchV1Api,
    CoreV1Api,
    V1Container,
    V1EnvVar,
    V1Job,
    V1JobSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1PodTemplateSpec,
    V1ResourceRequirements,
)
from vedana_core.settings import settings as core_settings
from vedana_etl.settings import settings as etl_settings

from vedana_backoffice.k8s.config import K8sConfig, k8s_config

logger = logging.getLogger(__name__)


class K8sJobLauncher:
    """Manages Kubernetes job lifecycle for ETL pipeline execution."""

    def __init__(self, k8s_config: K8sConfig = k8s_config):
        """Initialize the job launcher with configuration."""
        self.config = k8s_config

        if os.getenv("KUBERNETES_SERVICE_HOST"):  # when running inside k8s
            config.load_incluster_config()
        else:  # outside (container, local development)
            config.load_kube_config()

        self._k8s_client: BatchV1Api = BatchV1Api()
        self._k8s_core: CoreV1Api = CoreV1Api()

        self.namespace: str | None = None
        self.image: str | None = None
        self._detect_pod_info()

        self.env_vars = self._get_env_vars()

    def _detect_pod_info(self) -> tuple[str, str]:
        """Auto-detect namespace and image from backoffice pod.

        Returns:
            Tuple of (namespace, image)
        """
        if self.namespace and self.image:
            return self.namespace, self.image

        # Get namespace and image
        if os.getenv("KUBERNETES_SERVICE_HOST"):  # when running inside k8s
            pod_name = os.environ["HOSTNAME"]  # Get pod name from HOSTNAME (standard in K8s)
            namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
            pod = self._k8s_core.read_namespaced_pod(name=pod_name, namespace=namespace)
            image = pod.spec.containers[0].image  # todo check which image

        else:  # running in docker / locally - development/demo/testing etc.
            namespace = os.environ["K8S_NAMESPACE"]
            image = os.environ["K8S_IMAGE"]

        self.namespace = namespace
        self.image = image
        return namespace, image

    def _get_env_vars(self) -> list[V1EnvVar]:
        """Environment variables to pass to the job."""
        env_vars: list[V1EnvVar] = []
        env_dict = {key.upper(): value for key, value in etl_settings.model_dump().items()}

        # this is not represented in etl yet. todo fix?
        env_dict["MEMGRAPH_URI"] = core_settings.memgraph_uri
        env_dict["MEMGRAPH_USER"] = core_settings.memgraph_user
        env_dict["MEMGRAPH_PWD"] = core_settings.memgraph_pwd

        return env_vars

    def build_job_manifest(
        self,
        job_name: str,
        command: list[str],
        labels: dict[str, str] | None = None,
    ) -> V1Job:
        """Build a Kubernetes Job (V1Job) manifest.

        Args:
            job_name: Unique name for the job
            command: Command and arguments to run
            labels: Optional labels to add to the job

        Returns:
            V1Job manifest
        """

        # Build labels
        job_labels = {
            "app": "vedana-etl",  # todo
            "managed-by": "vedana-backoffice",
        }
        if labels:
            job_labels.update(labels)

        # Build container
        container = V1Container(
            name="etl",
            image=self.image,
            command=command,
            env=self.env_vars,
            resources=V1ResourceRequirements(
                requests={"cpu": self.config.job_cpu_request, "memory": self.config.job_memory_request},
                limits={"cpu": self.config.job_cpu_limit, "memory": self.config.job_memory_limit},
            ),
        )

        # Build pod spec
        pod_spec = V1PodSpec(
            containers=[container],
            restart_policy="Never",
        )

        # Build job spec
        job_spec = V1JobSpec(
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels=job_labels),
                spec=pod_spec,
            ),
            backoff_limit=0,  # Don't retry failed jobs automatically
        )

        # Build job
        job = V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=V1ObjectMeta(name=job_name, namespace=self.namespace, labels=job_labels),
            spec=job_spec,
        )

        return job

    def create_job(
        self,
        job_name: str,
        command: list[str],
        labels: dict[str, str] | None = None,
    ) -> V1Job:
        job = self.build_job_manifest(job_name, command, labels)
        created_job = self._k8s_client.create_namespaced_job(namespace=self.namespace, body=job)
        logger.info(f"Created job {job_name} in namespace {self.namespace}")
        return created_job

    def get_job_status(self, job_name: str) -> dict[str, Any]:
        """Get the status of a Kubernetes job.

        Args:
            job_name: Name of the job

        Returns:
            Dictionary with status information
        """
        try:
            job = self._k8s_client.read_namespaced_job(name=job_name, namespace=self.namespace)
            status = job.status

            result = {
                "job_name": job_name,
                "namespace": self.namespace,
                "active": status.active or 0,
                "succeeded": status.succeeded or 0,
                "failed": status.failed or 0,
                "conditions": [c.to_dict() for c in (status.conditions or [])],
            }

            # Determine overall status
            if status.succeeded:
                result["status"] = "completed"
            elif status.failed:
                result["status"] = "failed"
            elif status.active:
                result["status"] = "running"
            else:
                result["status"] = "pending"

            return result
        except client.exceptions.ApiException as e:
            if e.status == 404:
                return {"job_name": job_name, "status": "not_found", "error": "Job not found"}
            raise

    def get_job_pod_name(self, job_name: str) -> str | None:
        """Get the name of the pod created by a job.

        Args:
            job_name: Name of the job

        Returns:
            Pod name or None if not found
        """
        try:
            # List pods with label selector for the job
            pods = self._k8s_core.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                return pods.items[0].metadata.name
            return None
        except Exception as e:
            logger.error(f"Failed to get pod name for job {job_name}: {e}")
            return None

    def stream_job_logs(
        self,
        job_name: str,
        log_callback: Callable[[str], None] | None = None,
        follow: bool = True,
    ) -> Iterator[str]:
        """Stream logs from a job's pod.

        Args:
            job_name: Name of the job
            log_callback: Optional callback function to call for each log line
            follow: Whether to follow logs (stream in real-time)

        Yields:
            Log lines as strings
        """
        # Wait for pod to be created
        pod_name = None
        max_wait = 60  # Wait up to 60 seconds
        wait_interval = 1
        waited = 0

        while not pod_name and waited < max_wait:
            pod_name = self.get_job_pod_name(job_name)
            if not pod_name:
                time.sleep(wait_interval)
                waited += wait_interval

        if not pod_name:
            error_msg = f"Pod for job {job_name} not found after {max_wait} seconds"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            return

        # Stream logs
        try:
            log_stream = self._k8s_core.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                follow=follow,
                _preload_content=False,
            )

            for line in log_stream:
                line_str = line.decode("utf-8").rstrip()
                if log_callback:
                    log_callback(line_str)
                yield line_str

        except Exception as e:
            error_msg = f"Failed to stream logs from pod {pod_name}: {e}"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            raise

    def delete_job(self, job_name: str) -> None:
        """Delete a Kubernetes job.

        Args:
            job_name: Name of the job to delete
        """
        try:
            self._k8s_client.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
            logger.info(f"Deleted job {job_name} from namespace {self.namespace}")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.warning(f"Job {job_name} not found (may have been already deleted)")
            else:
                raise
