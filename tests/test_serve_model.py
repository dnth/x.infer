from ray import serve

from xinfer.serve import serve_model


def test_serve_model():
    serve_model("vikhyatk/moondream2", blocking=False, open_api_docs=False)

    serve.shutdown()


def test_serve_model_custom_deployment():
    """Test model serving with custom deployment options"""
    deployment_kwargs = {"num_replicas": 1, "ray_actor_options": {"num_cpus": 1}}
    handle = serve_model(
        "vikhyatk/moondream2",
        deployment_kwargs=deployment_kwargs,
        blocking=False,
        open_api_docs=False,
    )
    assert handle.deployment_id.name == "XInferModel"
    serve.shutdown()
