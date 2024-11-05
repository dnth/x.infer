from ray import serve

from xinfer.serve import serve_model


def test_serve_model():
    serve_model("vikhyatk/moondream2", blocking=False)

    serve.shutdown()


def test_serve_model_custom_deployment():
    """Test model serving with custom deployment options"""
    deployment_kwargs = {"num_replicas": 2, "ray_actor_options": {"num_cpus": 2}}
    handle = serve_model(
        "vikhyatk/moondream2", deployment_kwargs=deployment_kwargs, blocking=False
    )
    assert handle.deployment_id.name == "XInferModel"
    serve.shutdown()
